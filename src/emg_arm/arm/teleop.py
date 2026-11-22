"""Turns noisy gesture predictions and held keys into safe arm motion.

Pure Python (no Qt) so the safety logic can be unit-tested.

    GestureFilter (vote / hysteresis / stale guard)      -> a stable gesture ID
    control.commands.CommandGenerator + MappingStore    -> moves / one-shot actions
    Teleop (this file)                                  -> per-servo velocities -> MotionStreamer -> SafeArm
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Callable, Optional

from ..control.commands import CommandGenerator, OneShot
from ..control.gestures import DIRECTION_GESTURES, resolve_gesture
from ..control.mapping import VERTICAL, MappingStore

MOTIONS = DIRECTION_GESTURES          # kept for older imports
CONTROL_MODES = ("hybrid", "gestures", "keyboard")


def canonical_gesture(label):
    """Backwards-compatible name for ``resolve_gesture`` (default word rules, no overrides)."""
    return resolve_gesture(label)


class GestureFilter:
    """Majority vote with hysteresis + confidence gate + stale / lockout guards.

    It only decides *which gesture is stable and for how long*. What that gesture does is the
    business of the mapping, so the same filter serves all ten gestures.
    """

    def __init__(self, cfg, resolver: Callable[[str], Optional[str]] = resolve_gesture,
                 is_continuous: Callable[[Optional[str]], bool] = lambda g: g in DIRECTION_GESTURES,
                 conf_for: Optional[Callable[[Optional[str], float], float]] = None):
        self.min_conf = float(cfg["min_confidence"])
        self.vote_window = int(cfg["vote_window"])
        self.stable_votes = int(cfg["stable_votes"])
        self.hold_votes = min(int(cfg.get("hold_votes", self.stable_votes)), self.stable_votes)
        self.grace_s = float(cfg.get("dropout_grace_s", 0.0))
        self.timeout_s = float(cfg["prediction_timeout_s"])
        self.max_move_s = float(cfg["max_continuous_move_s"])
        self.resolver = resolver
        self.is_continuous = is_continuous
        self.conf_for = conf_for or (lambda gid, default: default)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.vote_window)
        self.last_prediction_t = None
        self.stable = None
        self.stable_since = 0.0
        self.locked_out = False
        self._dropped = None  # (gesture, since, locked_out, when) for the flicker grace

    def add_prediction(self, label, confidence, now):
        gid = self.resolver(label)
        gesture = gid if gid and confidence >= self.conf_for(gid, self.min_conf) else None
        self.history.append(gesture)
        self.last_prediction_t = now
        counts = Counter(g for g in self.history if g is not None)
        if self.stable is not None and counts[self.stable] >= self.hold_votes:
            stable = self.stable  # hysteresis: easier to stay than to enter
        else:
            best = counts.most_common(1)
            stable = best[0][0] if best and best[0][1] >= self.stable_votes else None
        if stable != self.stable:
            self._change(stable, now)
        return gesture

    def _change(self, new, now):
        old = self.stable
        since, locked = now, False
        dropped = self._dropped
        self._dropped = None
        if self.is_continuous(old) and new is None:
            self._dropped = (old, self.stable_since, self.locked_out, now)
        elif dropped and new == dropped[0] and now - dropped[3] <= self.grace_s:
            # A brief misclassification inside a held motion: carry on with the
            # same speed ramp and lockout timer instead of starting over.
            since, locked = dropped[1], dropped[2]
        self.stable, self.stable_since, self.locked_out = new, since, locked

    def is_fresh(self, now):
        return self.last_prediction_t is not None and now - self.last_prediction_t <= self.timeout_s

    def held_for(self, now):
        return max(0.0, now - self.stable_since)


class Teleop:
    """One control tick: choose a motion source, turn it into per-servo
    velocities and hand them to the MotionStreamer."""

    def __init__(self, arm, gesture_filter, config, streamer, mapping: Optional[MappingStore] = None):
        self.arm = arm
        self.filter = gesture_filter
        self.streamer = streamer
        self.mapping = mapping or MappingStore(config)
        fcfg = config["filter"]
        self.commands = CommandGenerator(self.mapping, float(fcfg.get("grip_cooldown_s", 1.5)))
        self.gesture_map = config["gestures"]          # keyboard Left/Right/Up/Down
        self.vertical_joints = config["vertical_joints"]
        self.vertical = next(iter(self.vertical_joints))
        self.ramp_start = float(fcfg["ramp_start"])
        self.ramp_time = float(fcfg["ramp_time_s"])
        self.max_move_s = float(fcfg["max_continuous_move_s"])
        self.reversal_guard_s = float(fcfg.get("reversal_guard_s", 0.2))
        self.mode = "hybrid"
        self.armed = False
        self._last_dir = {}        # servo -> (direction, time it was last driven by a gesture)

    # ------------------------------------------------------------ servo lookup
    def joint_servo(self, joint_name):
        j = self.vertical_joints[joint_name]
        return int(j["servo"]), int(j["up_direction"])

    def servo_for(self, motion):
        """left/right/up/down (keyboard; up/down use the selected joint) or
        '<joint>_up' / '<joint>_down' (direct keyboard keys)."""
        if motion in self.gesture_map:
            entry = self.gesture_map[motion]
            direction = int(entry["direction"])
            servo = entry["servo"]
            if servo != VERTICAL:
                return int(servo), direction
            joint = self.vertical_joints[self.vertical]
        else:
            name, _, way = motion.rpartition("_")
            joint = self.vertical_joints[name]
            direction = 1 if way == "up" else -1
        return int(joint["servo"]), direction * int(joint["up_direction"])

    def gesture_servo(self, servo, direction):
        """Servo and sign for a gesture mapping (``vertical`` follows the selected joint)."""
        if servo == VERTICAL:
            sid, up = self.joint_servo(self.vertical)
            return sid, direction * up
        return int(servo), direction

    def switch_vertical(self):
        names = list(self.vertical_joints)
        self.vertical = names[(names.index(self.vertical) + 1) % len(names)]
        return self.vertical

    def ramp(self, held_s):
        if self.ramp_time <= 0:
            return 1.0
        return self.ramp_start + (1.0 - self.ramp_start) * min(1.0, held_s / self.ramp_time)

    # ------------------------------------------------------------------- tick
    def tick(self, now, key_motions=None, key_grip=False, emg_ok=True):
        """key_motions: {motion: seconds held} for every motion key held.
        Returns a status dict for the UI. Raises on hardware errors."""
        status = {"motion": None, "source": None, "gripper": None, "switched": None, "home": False,
                  "goto": None, "gesture": None, "hint": "", "note": "", "velocities": {}}
        if not self.armed:
            self.streamer.command({}, now)
            self.commands.reset()
            return status

        if self.mode == "gestures":
            key_motions, key_grip = None, False
        elif self.mode == "keyboard":
            emg_ok = False

        # (servo, direction, speed override or 0, held_s, label)
        intents, shots = [], []
        if key_motions or key_grip:
            source = "keyboard"
            moves = {m: h for m, h in (key_motions or {}).items() if h <= self.max_move_s}
            if len(moves) < len(key_motions or {}):
                status["note"] = "Key held too long - release it"
            for motion, held in moves.items():
                servo, direction = self.servo_for(motion)
                intents.append((servo, direction, 0.0, held, motion))
            if key_grip:
                shots.append(OneShot("keyboard", "gripper_toggle", None))
        elif emg_ok:
            source = "emg"
            cmds = self.commands.generate(self.filter, now)
            status["gesture"], status["hint"], status["note"] = cmds.gesture_id, cmds.hint, cmds.note
            for mv in cmds.moves:
                servo, direction = self.gesture_servo(mv.servo, mv.direction)
                if self._reversal_blocked(servo, direction, now):
                    status["note"] = "Direction change guard - hold the new gesture a moment"
                    continue
                intents.append((servo, direction, mv.speed, mv.held_s, mv.gesture_id))
            shots = cmds.oneshots
        else:
            self.streamer.command({}, now)
            self.commands.reset()
            status["note"] = "EMG stream lost - holding position"
            return status

        for shot in shots:
            self._run_oneshot(shot, status, source)

        velocities, names = {}, []
        for servo, direction, speed_override, held, name in intents:
            if self.arm.at_limit(servo, direction):
                status["note"] = f"{self.arm.names[servo]} at its safe limit - move the other way"
                continue
            base = speed_override if speed_override > 0 else self.arm.speed
            cap = max(self.arm.speed, speed_override)
            v = velocities.get(servo, 0.0) + direction * base * self.ramp(held)
            velocities[servo] = max(-cap, min(cap, v))
            names.append(name)
            if source == "emg":
                self._last_dir[servo] = (direction, now)
        self.streamer.command(velocities, now)
        if names:
            status["motion"] = " + ".join(sorted(set(names)))
            status["source"] = source
        status["velocities"] = velocities
        return status

    # --------------------------------------------------------------- helpers
    def _reversal_blocked(self, servo, direction, now):
        """A gesture that would instantly reverse a servo that a gesture was just driving the other
        way is held back: the arm brakes first. Prevents left/right chatter shaking the base."""
        last = self._last_dir.get(servo)
        return bool(last and last[0] != direction and now - last[1] < self.reversal_guard_s)

    def _run_oneshot(self, shot, status, source):
        action = shot.action
        if action == "gripper_toggle":
            status["gripper"] = "closed" if self.arm.toggle_gripper() else "open"
        elif action == "gripper_open":
            self.arm.set_gripper(False)
            status["gripper"] = "open"
        elif action == "gripper_close":
            self.arm.set_gripper(True)
            status["gripper"] = "closed"
        elif action == "switch_joint":
            status["switched"] = self.switch_vertical()
        elif action == "go_home":
            status["home"] = True
        elif action == "move_to":
            m = shot.mapping
            self.streamer.halt()
            self.arm.move_to(int(m.servo), m.target_position, m.duration_ms)
            status["goto"] = (int(m.servo), m.target_position)
        else:
            return
        status["source"] = source
