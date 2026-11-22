"""Base-servo jitter mitigation. Simulated (software-side) measurements only: real servo vibration
must still be validated on the arm (see README, hardware checklist)."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from emg_arm.arm.driver import ArmSafetyError, SafeArm, SimulatedBackend, load_config  # noqa: E402
from emg_arm.arm.motion import MotionStreamer  # noqa: E402
import jitter_report  # noqa: E402

CONFIG = load_config()


def legacy_config():
    cfg = copy.deepcopy(CONFIG)
    cfg["arm"].update(min_speed_units_per_s=0, min_lead_units=0, match_duration=False)
    return cfg


class JitterTests(unittest.TestCase):
    def test_low_speed_commanded_speed_is_steadier_than_the_original(self):
        for speed in (10, 20):
            old, _ = jitter_report.measure(speed, legacy_config())
            new, _ = jitter_report.measure(speed, CONFIG)
            old_spread = jitter_report.summarise(old, speed)["relative_speed_spread"]
            new_spread = jitter_report.summarise(new, speed)["relative_speed_spread"]
            self.assertLess(new_spread, 0.02, f"{speed} u/s")
            self.assertLess(new_spread * 5, old_spread, f"{speed} u/s: expected a large reduction")

    def test_normal_speed_is_not_degraded(self):
        samples, _ = jitter_report.measure(100, CONFIG)
        self.assertLess(jitter_report.summarise(samples, 100)["relative_speed_spread"], 0.02)

    def test_movement_range_is_not_reduced(self):
        """The base still reaches its configured limit; smoothing must not shorten the range."""
        backend = SimulatedBackend({1: 100, 2: 500, 4: 820, 5: 700, 6: 500})
        arm = SafeArm(CONFIG, backend)
        arm.read_positions()
        arm.go_home()
        arm.set_speed(100)
        streamer = MotionStreamer(arm, CONFIG["arm"])
        t = 0.0
        for _ in range(400):                             # 16 s towards +limit
            streamer.command({6: 100}, t)
            t += 0.04
            streamer.step(0.04, t)
        hi = CONFIG["arm"]["servos"]["6"]["max"]
        self.assertGreaterEqual(arm.targets[6], hi - 1)
        self.assertLessEqual(arm.targets[6], hi)

    def test_no_duplicate_packets(self):
        samples, packets = jitter_report.measure(10, CONFIG, seconds=5.0)
        self.assertLessEqual(packets, 5.0 / 0.04 + 5)      # never more than one packet per stream period
        backend = SimulatedBackend({1: 100, 2: 500, 4: 820, 5: 700, 6: 500})
        arm = SafeArm(CONFIG, backend)
        arm.read_positions()
        arm.go_home()
        n = len(backend.packets)
        arm._send(6, arm.targets[6], 100)                  # same target again
        self.assertEqual(len(backend.packets), n, "identical target is not re-sent")

    def test_min_speed_floor_prevents_creeping(self):
        backend = SimulatedBackend({1: 100, 2: 500, 4: 820, 5: 700, 6: 500})
        arm = SafeArm(CONFIG, backend)
        arm.read_positions()
        arm.go_home()
        streamer = MotionStreamer(arm, CONFIG["arm"])
        streamer.command({6: 1.0}, 0.0)                    # absurdly slow request
        for i in range(60):
            streamer.command({6: 1.0}, i * 0.04)
            streamer.step(0.04, (i + 1) * 0.04)
        self.assertGreater(abs(arm.setpoint(6) - 500), 8 * 1.5 * 0.5, "moves at least at the minimum speed")

    def test_tuning_is_validated_and_live(self):
        arm = SafeArm(CONFIG, SimulatedBackend({1: 100, 2: 500, 4: 820, 5: 700, 6: 500}))
        streamer = MotionStreamer(arm, CONFIG["arm"])
        streamer.apply_tuning(lookahead=0.2, jerk=3000)
        self.assertEqual((streamer.lookahead, streamer.jerk), (0.2, 3000))
        with self.assertRaises(ArmSafetyError):
            streamer.apply_tuning(lookahead=0.01)
        with self.assertRaises(ArmSafetyError):
            streamer.apply_tuning(jerk=0)
        self.assertEqual(streamer.lookahead, 0.2, "a rejected change leaves the previous values")


if __name__ == "__main__":
    unittest.main()
