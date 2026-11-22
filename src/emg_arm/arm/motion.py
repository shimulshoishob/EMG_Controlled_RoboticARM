"""Smooth, continuous joint motion for the xArm 1S.

The xArm servos take "go to P within T ms" commands, and each new command
restarts the move from wherever the servo is. The old approach sent a tiny step
from the GUI timer on every tick. The step size and timing then depended on GUI
load, and the blocking USB writes ran on the GUI thread, so motion was jerky and
laggy.

MotionStreamer runs in its own thread at a fixed rate. Callers only set a
desired velocity per servo (units/s). Every cycle it:
  * ramps each servo's velocity toward the desired one along an S-curve:
    acceleration is limited, and so is how fast the acceleration itself
    changes (jerk), so starts, stops and reversals have no sudden kick,
  * brakes ahead of a soft limit so the joint glides into it instead of
    stopping dead at full speed,
  * moves a float setpoint by velocity * dt, clamped to the soft limits,
  * sends ONE packet that aims every moving servo `lookahead` ahead of its
    setpoint, with duration = lookahead. The servo then moves at the commanded
    speed and never comes to rest between packets.
A servo whose velocity reaches zero gets one last command to its setpoint.
If the GUI stops sending commands (frozen, crashed tick), the arm slows to a stop.

All limits and the hard speed cap are still enforced by SafeArm.
"""

import math
import threading
import time

from .driver import MIN_WRITE_GAP_S, ArmSafetyError


def _approach(value, goal, max_delta):
    if value < goal:
        return min(goal, value + max_delta)
    return max(goal, value - max_delta)


class MotionStreamer:
    def __init__(self, arm, arm_cfg, clock=time.monotonic):
        self.arm = arm
        self.clock = clock
        self.period = float(arm_cfg.get("stream_period_ms", 40)) / 1000.0
        self.lookahead = float(arm_cfg.get("lookahead_ms", 120)) / 1000.0
        self.accel = float(arm_cfg.get("accel_units_per_s2", 500))
        self.decel = float(arm_cfg.get("decel_units_per_s2", 1000))
        self.jerk = float(arm_cfg.get("jerk_units_per_s3", 6000))
        self.command_timeout = float(arm_cfg.get("command_timeout_s", 0.25))
        # Anti-jitter (all live-tunable via apply_tuning):
        #  min_speed:  slowest commanded speed; slower goals are raised to it, because a servo asked to
        #              creep a fraction of a unit per packet stutters (stop-go) instead of gliding.
        #  min_lead:   the target is always at least this many units ahead of the setpoint. At low speed
        #              the lookahead is stretched to achieve it, so integer rounding of the target
        #              (+-0.5 unit) stays a small fraction of the commanded step.
        self.min_speed = float(arm_cfg.get("min_speed_units_per_s", 8))
        self.min_lead = float(arm_cfg.get("min_lead_units", 3))
        self.max_lookahead = 0.5
        self.match_duration = bool(arm_cfg.get("match_duration", True))
        if self.period < MIN_WRITE_GAP_S:
            raise ArmSafetyError(f"stream_period_ms must be at least {MIN_WRITE_GAP_S * 1000:.0f}")
        if not self.period <= self.lookahead <= 0.5:
            raise ArmSafetyError("lookahead_ms must be between stream_period_ms and 500")
        if self.accel <= 0 or self.decel <= 0 or self.jerk <= 0:
            raise ArmSafetyError("accel/decel/jerk must be positive")

        self._goal_lock = threading.Lock()  # guards _goal / _goal_t (cheap, GUI-facing)
        self._step_lock = threading.Lock()  # guards velocities and all streaming I/O
        self._goal = {}
        self._goal_t = -math.inf
        self._vel = {}
        self._acc = {}
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ GUI side
    def command(self, velocities, now=None):
        """Set the desired signed velocity (units/s) per servo. Servos left
        out slow down to a stop. Call this regularly; see command_timeout_s."""
        cap = self.arm.hard_max_speed
        goal = {int(s): max(-cap, min(cap, float(v))) for s, v in velocities.items() if v}
        with self._goal_lock:
            self._goal = goal
            self._goal_t = self.clock() if now is None else now

    def apply_tuning(self, **values):
        """Change smoothing parameters while running (from the Settings page). Invalid values are
        refused so a typo can never make the arm unsafe."""
        new = {"lookahead": self.lookahead, "accel": self.accel, "decel": self.decel, "jerk": self.jerk,
               "min_speed": self.min_speed, "min_lead": self.min_lead}
        new.update({k: float(v) for k, v in values.items() if k in new})
        if not self.period <= new["lookahead"] <= self.max_lookahead:
            raise ArmSafetyError("lookahead must be between the stream period and 500 ms")
        if min(new["accel"], new["decel"], new["jerk"]) <= 0 or new["min_speed"] < 0 or new["min_lead"] < 0:
            raise ArmSafetyError("accel/decel/jerk must be positive; min speed and lead cannot be negative")
        with self._step_lock:
            for key, value in new.items():
                setattr(self, key, value)

    def halt(self):
        """Stop at once (emergency stop / disarm): every moving servo is told to
        stay at its current setpoint."""
        with self._goal_lock:
            self._goal = {}
        with self._step_lock:
            moving = list(self._vel)
            self._vel.clear()
            self._acc.clear()
            if moving and self.error is None:
                self.arm.hold(moving)

    def is_moving(self):
        return bool(self._vel)

    def clear_error(self):
        self.error = None

    # ------------------------------------------------------------- stepping
    def step(self, dt, now=None):
        """One streaming cycle. Runs in the thread; tests call it directly."""
        now = self.clock() if now is None else now
        dt = min(max(dt, 0.0), 0.1)  # a stalled thread must never cause a jump
        with self._goal_lock:
            goal = dict(self._goal) if now - self._goal_t <= self.command_timeout else {}
        with self._step_lock:
            if self.error is not None:
                return
            setpoints, vels = {}, {}
            for sid in set(goal) | set(self._vel):
                v = self._vel.get(sid, 0.0)
                g = goal.get(sid, 0.0)
                if g == 0.0 and sid not in self._vel:
                    continue  # idle and staying idle
                lo, hi = self.arm.limits[sid]
                p = self.arm.setpoint(sid)
                if g:
                    g = math.copysign(max(abs(g), self.min_speed), g)   # no creeping
                    # Never go faster than the S-curve can stop from within the
                    # room left, so the joint eases into its soft limit.
                    room = max(0.0, hi - p if g > 0 else p - lo)
                    g = math.copysign(min(abs(g), self._speed_to_stop_within(room, dt)), g)
                v, a = self._s_curve(v, self._acc.get(sid, 0.0), g, dt)
                p += v * dt
                if (p >= hi - 0.5 and v > 0) or (p <= lo + 0.5 and v < 0):
                    # Reached a soft limit (within half a unit: the braking
                    # curve only creeps over that last bit).
                    p, v, a = (hi if v > 0 else lo), 0.0, 0.0
                setpoints[sid] = p
                vels[sid] = v
                if v == 0.0 and a == 0.0:
                    self._vel.pop(sid, None)  # this packet is its final "stop here"
                    self._acc.pop(sid, None)
                else:
                    self._vel[sid] = v
                    self._acc[sid] = a
            if not setpoints:
                return
            lookahead = self._lookahead_for(vels.values())
            leads = {sid: v * lookahead for sid, v in vels.items()}
            if self.match_duration and len(vels) == 1:
                lookahead, leads = self._match_duration(next(iter(vels)), setpoints, vels, lookahead, leads)
            try:
                self.arm.stream(setpoints, leads, lookahead * 1000.0)
            except Exception as exc:
                self.error = exc
                self._vel.clear()
                self._acc.clear()
                with self._goal_lock:
                    self._goal = {}

    def _match_duration(self, sid, setpoints, vels, lookahead, leads):
        """One servo moving (the usual case): the arm sends integer targets, so the aimed-at lead is
        rounded to a whole unit. Choose the packet duration from the rounded lead so the commanded
        speed is exactly v, instead of v +- rounding noise (several percent at low speed)."""
        v, p = vels[sid], setpoints[sid]
        lo, hi = self.arm.limits[sid]
        target = round(min(hi, max(lo, p + leads[sid])))
        lead = target - p
        if abs(lead) < 1.0 or lead * v <= 0:
            return lookahead, leads                  # at a limit or stopping: keep the plain packet
        duration = lead / v
        if not 0.5 * lookahead <= duration <= min(self.max_lookahead, 1.5 * lookahead):
            return lookahead, leads
        return duration, {sid: lead}

    def _lookahead_for(self, velocities):
        """Lookahead for this packet: the configured one, stretched when the slowest moving servo
        would otherwise be aimed less than min_lead units ahead."""
        slowest = min((abs(v) for v in velocities if v), default=0.0)
        if slowest <= 0.0 or self.min_lead <= 0.0:
            return self.lookahead
        return min(self.max_lookahead, max(self.lookahead, self.min_lead / slowest))

    def _stop_distance(self, speed):
        """Distance an S-curve stop from `speed` (starting at zero acceleration) takes."""
        if speed <= self.decel ** 2 / self.jerk:  # never reaches full decel
            return speed * math.sqrt(speed / self.jerk)
        return speed ** 2 / (2.0 * self.decel) + speed * self.decel / (2.0 * self.jerk)

    def _speed_to_stop_within(self, room, dt):
        """Highest speed that can still stop within `room` (two cycles of margin)."""
        lo, hi = 0.0, self.arm.hard_max_speed
        if self._stop_distance(hi) + 2.0 * hi * dt <= room:
            return hi
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if self._stop_distance(mid) + 2.0 * mid * dt <= room:
                lo = mid
            else:
                hi = mid
        return lo

    def _s_curve(self, v, a, goal, dt):
        """Jerk-limited step of velocity v (acceleration a) toward goal.
        The acceleration aims at the most that can still ramp back to zero,
        one jerk * dt step per cycle, by the time the velocity arrives:
        a * (a + jerk * dt) / (2 * jerk) <= |error|. So it eases into the goal
        speed instead of overshooting."""
        error = goal - v
        if error == 0.0 and a == 0.0:
            return v, a
        speeding_up = goal * v >= 0 and abs(goal) > abs(v)
        a_max = self.accel if speeding_up else self.decel
        j_step = self.jerk * dt
        a_fit = (math.sqrt(j_step ** 2 + 8.0 * self.jerk * abs(error)) - j_step) / 2.0
        a_goal = math.copysign(min(a_max, a_fit), error)
        a = _approach(a, a_goal, self.jerk * dt)
        v_next = v + a * dt
        if (goal - v_next) * error <= 0.0:
            return goal, 0.0  # arrived at (or would pass) the goal speed
        return v_next, a

    # --------------------------------------------------------------- thread
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="arm-motion", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self.halt()
        except Exception:
            pass

    def _run(self):
        last = next_t = time.monotonic()
        while not self._stop.is_set():
            next_t += self.period
            wait = next_t - time.monotonic()
            if wait > 0:
                if self._stop.wait(wait):
                    break
            else:
                next_t = time.monotonic()  # fell behind: don't try to catch up
            now = time.monotonic()
            self.step(now - last)
            last = now
