#!/usr/bin/env python3
"""Software-side evidence about base-servo jitter (no hardware needed).

Streams a constant-speed jog of the base servo through the real MotionStreamer into a simulated arm,
then replays the packets through a simple servo model (linear move to each target over its duration,
new packet = new move from the current position) and reports how steady the *commanded* servo speed is.
It measures the software's contribution only: it cannot see mechanical backlash, power sag or the
servo's own control loop. Compare "legacy" (min speed / min lead off) with the shipped settings.

    python tools/jitter_report.py
"""

import copy
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emg_arm.arm.driver import SafeArm, SimulatedBackend, load_config  # noqa: E402
from emg_arm.arm.motion import MotionStreamer  # noqa: E402

BASE = 6


class RecordingBackend(SimulatedBackend):
    def __init__(self, start):
        super().__init__(start)
        self.now = 0.0
        self.log = []                      # (time, {servo: target}, duration_s)

    def set_positions(self, positions, duration_ms):
        self.log.append((self.now, dict(positions), duration_ms / 1000.0))
        super().set_positions(positions, duration_ms)


def servo_speed_samples(log, servo, start_pos, t_end, dt=0.001, warmup=1.0):
    """Speed of an ideal servo that obeys every packet: (target - position) / duration."""
    x, goal, t_goal_end, v = float(start_pos), None, 0.0, 0.0
    i, t, samples = 0, 0.0, []
    while t < t_end:
        while i < len(log) and log[i][0] <= t:
            if servo in log[i][1]:
                goal = log[i][1][servo]
                duration = max(log[i][2], 1e-3)
                v = (goal - x) / duration
                t_goal_end = log[i][0] + duration
            i += 1
        if goal is not None and t < t_goal_end:
            x += v * dt
        else:
            v = 0.0
        if t >= warmup:
            samples.append(v)
        t += dt
    return samples


def measure(speed, config, seconds=None, period=0.04):
    seconds = seconds or min(6.0, 300.0 / speed)        # stay clear of the base's soft limit
    backend = RecordingBackend({1: 100, 2: 500, 4: 820, 5: 700, 6: 500})
    arm = SafeArm(config, backend)
    arm.read_positions()
    arm.go_home()
    backend.log.clear()
    streamer = MotionStreamer(arm, config["arm"])
    t = 0.0
    for _ in range(int(seconds / period)):
        backend.now = t
        streamer.command({BASE: speed}, t)
        t += period
        streamer.step(period, t)
    samples = servo_speed_samples(backend.log, BASE, 500, t)
    return samples, len(backend.log)


def summarise(samples, target_speed):
    cruise = [v for v in samples if v != 0.0]
    stopped = 1.0 - len(cruise) / len(samples)
    spread = statistics.pstdev(samples) / target_speed if samples else 0.0
    return {"relative_speed_spread": spread, "time_stopped": stopped}


def main():
    shipped = load_config()
    legacy = copy.deepcopy(shipped)
    legacy["arm"].update(min_speed_units_per_s=0, min_lead_units=0, match_duration=False)   # the original behaviour
    print(f"{'speed u/s':>10} | {'setting':8} | {'speed spread (std/target)':>26} | {'time with servo idle':>20}")
    for speed in (10, 20, 50, 100):
        for name, cfg in (("legacy", legacy), ("shipped", shipped)):
            samples, _ = measure(speed, cfg)
            r = summarise(samples, speed)
            print(f"{speed:>10} | {name:8} | {r['relative_speed_spread']:>26.3f} | {r['time_stopped'] * 100:>18.1f} %")


if __name__ == "__main__":
    main()
