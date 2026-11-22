"""Gesture mapping: resolution, defaults, validation, persistence, import/export, execution.

    python -m unittest tests.test_mapping -v        (no hardware, no Qt)
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emg_arm.arm.driver import SafeArm, SimulatedBackend, load_config  # noqa: E402
from emg_arm.arm.motion import MotionStreamer  # noqa: E402
from emg_arm.arm.teleop import GestureFilter, Teleop  # noqa: E402
from emg_arm.config_bundle import effective_config  # noqa: E402
from emg_arm.control.gestures import GESTURE_IDS, resolve_gesture  # noqa: E402
from emg_arm.control.mapping import (GestureMapping, MappingError, MappingStore, default_mappings, errors,  # noqa: E402
                                     validate)

CONFIG = load_config()


class FakeSettings(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class GestureResolutionTests(unittest.TestCase):
    def test_ten_gestures(self):
        self.assertEqual(len(GESTURE_IDS), 10)
        self.assertEqual(len(set(GESTURE_IDS)), 10)

    def test_study_names_resolve_to_their_own_id(self):
        names = {"Left": "left", "Right": "right", "Up": "up", "Down": "down", "Closed Fist": "closed_fist",
                 "Double Closed Fist": "double_fist", "Victory Sign": "victory", "Thumbs Up": "thumbs_up",
                 "Pinch using thumb and index finger": "pinch", "Rest": "rest"}
        for label, expected in names.items():
            self.assertEqual(resolve_gesture(label), expected, label)

    def test_double_fist_is_not_read_as_fist(self):
        self.assertEqual(resolve_gesture("double_fist_close"), "double_fist")
        self.assertEqual(resolve_gesture("fist_close"), "closed_fist")

    def test_override_wins(self):
        self.assertIsNone(resolve_gesture("squeeze"))
        self.assertEqual(resolve_gesture("squeeze", {"squeeze": "pinch"}), "pinch")


class DefaultsAndValidationTests(unittest.TestCase):
    def test_defaults_cover_all_gestures_and_are_valid(self):
        m = default_mappings(CONFIG)
        self.assertEqual(set(m), set(GESTURE_IDS))
        self.assertEqual(errors(validate(m, CONFIG)), [])

    def test_defaults_reproduce_the_previous_behaviour(self):
        m = default_mappings(CONFIG)
        self.assertEqual((m["left"].servo, m["left"].direction), (6, 1))
        self.assertEqual((m["right"].servo, m["right"].direction), (6, -1))
        self.assertEqual((m["up"].servo, m["up"].direction), ("vertical", 1))
        self.assertEqual(m["closed_fist"].action, "gripper_toggle")
        self.assertEqual(m["rest"].action, "hold")

    def test_dangerous_defaults_are_off(self):
        self.assertFalse(default_mappings(CONFIG)["double_fist"].enabled)

    def _bad(self, **change):
        m = default_mappings(CONFIG)
        for k, v in change.items():
            setattr(m["victory"], k, v)
        return errors(validate(m, CONFIG))

    def test_rejects_unknown_servo_and_action(self):
        self.assertTrue(self._bad(action="move_servo", servo=99))
        self.assertTrue(self._bad(action="teleport"))

    def test_rejects_speed_above_cap_and_negative(self):
        self.assertTrue(self._bad(action="move_servo", servo=2, speed=CONFIG["arm"]["hard_max_speed_units_per_s"] + 1))
        self.assertTrue(self._bad(action="move_servo", servo=2, speed=-5))

    def test_rejects_target_outside_servo_range(self):
        self.assertTrue(self._bad(action="move_to", servo=6, target_position=900))     # base max is 850
        self.assertFalse(self._bad(action="move_to", servo=6, target_position=800, duration_ms=2000))

    def test_rejects_gripper_as_continuous_joint(self):
        self.assertTrue(self._bad(action="move_servo", servo=1))

    def test_rejects_conflicting_continuous_mappings(self):
        issues = self._bad(action="move_servo", servo=6, direction=1)     # Left already drives servo 6, +1
        self.assertTrue(any("conflicts" in i.message for i in issues))

    def test_disabled_gesture_does_not_conflict(self):
        m = default_mappings(CONFIG)
        m["victory"] = GestureMapping("victory", False, "move_servo", 6, 1)
        self.assertEqual(errors(validate(m, CONFIG)), [])

    def test_rejects_duplicate_model_label(self):
        m = default_mappings(CONFIG)
        m["victory"].model_label = "x"
        m["pinch"].model_label = "X"
        self.assertTrue(errors(validate(m, CONFIG)))

    def test_long_hold_must_exceed_hold(self):
        self.assertTrue(self._bad(action="gripper_toggle", long_action="switch_joint", hold_s=2.0, long_hold_s=1.0))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "map.json"

    def test_persists_across_restart(self):
        a = MappingStore(CONFIG, self.path)
        new = copy.deepcopy(a.mappings)
        new["victory"] = GestureMapping("victory", True, "move_servo", 2, -1, speed=40)
        new["pinch"].enabled = False
        a.set_all(new)
        b = MappingStore(CONFIG, self.path)                 # "restart"
        self.assertEqual(b.mappings["victory"].action, "move_servo")
        self.assertEqual(b.mappings["victory"].speed, 40)
        self.assertFalse(b.mappings["pinch"].enabled)
        self.assertEqual(b.load_warning, "")

    def test_invalid_set_changes_nothing(self):
        a = MappingStore(CONFIG, self.path)
        bad = copy.deepcopy(a.mappings)
        bad["victory"] = GestureMapping("victory", True, "move_servo", 99, 1)
        with self.assertRaises(MappingError):
            a.set_all(bad)
        self.assertEqual(a.mappings["victory"].action, "gripper_open")
        self.assertFalse(self.path.exists())

    def test_reset_to_defaults(self):
        a = MappingStore(CONFIG, self.path)
        new = copy.deepcopy(a.mappings)
        new["pinch"].enabled = False
        a.set_all(new)
        a.reset_defaults()
        self.assertTrue(a.mappings["pinch"].enabled)
        self.assertTrue(MappingStore(CONFIG, self.path).mappings["pinch"].enabled)

    def test_corrupt_file_falls_back_to_defaults_with_warning(self):
        self.path.write_text("{not json", encoding="utf-8")
        a = MappingStore(CONFIG, self.path)
        self.assertEqual(a.mappings["left"].action, "move_servo")
        self.assertIn("ignored", a.load_warning)

    def test_invalid_saved_values_fall_back(self):
        data = MappingStore(CONFIG).to_json()
        data["mappings"]["left"]["servo"] = 99
        self.path.write_text(json.dumps(data), encoding="utf-8")
        a = MappingStore(CONFIG, self.path)
        self.assertEqual(a.mappings["left"].servo, 6)
        self.assertTrue(a.load_warning)

    def test_export_import_roundtrip_and_bad_import(self):
        a = MappingStore(CONFIG, self.path)
        new = copy.deepcopy(a.mappings)
        new["victory"] = GestureMapping("victory", True, "move_to", 6, target_position=700, duration_ms=2500)
        a.set_all(new)
        out = self.dir / "export.json"
        a.export_to(out)
        b = MappingStore(CONFIG, self.dir / "other.json")
        b.import_from(out)
        self.assertEqual(b.mappings["victory"].target_position, 700)
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"mappings": {"victory": {"action": "move_to", "servo": 6, "target_position": 5}}}))
        with self.assertRaises(MappingError):
            b.import_from(bad)
        self.assertEqual(b.mappings["victory"].target_position, 700, "failed import leaves the mapping untouched")

    def test_missing_gestures_in_file_keep_defaults(self):
        self.path.write_text(json.dumps({"version": 1, "mappings": {"pinch": {"action": "none"}}}))
        a = MappingStore(CONFIG, self.path)
        self.assertEqual(a.mappings["pinch"].action, "none")
        self.assertEqual(a.mappings["left"].action, "move_servo")

    def test_other_arm_configuration_gets_its_own_defaults(self):
        cfg = copy.deepcopy(CONFIG)
        cfg["gestures"]["left"] = {"servo": 2, "direction": -1}
        m = default_mappings(cfg)
        self.assertEqual((m["left"].servo, m["left"].direction), (2, -1))

    def test_effective_config_only_narrows_limits(self):
        s = FakeSettings(servo_ranges={"6": [100, 900]})                  # wider than shipped 150-850: ignored
        self.assertEqual(effective_config(CONFIG, s)["arm"]["servos"]["6"]["min"], 150)
        s = FakeSettings(servo_ranges={"6": [200, 800]}, home_overrides={"6": 450})
        eff = effective_config(CONFIG, s)
        self.assertEqual((eff["arm"]["servos"]["6"]["min"], eff["arm"]["servos"]["6"]["max"]), (200, 800))
        self.assertEqual(eff["arm"]["home"]["6"], 450)
        SafeArm(eff, SimulatedBackend({}))                                  # still passes the safety validation


def make_teleop(store):
    arm = SafeArm(CONFIG, SimulatedBackend({1: 100, 2: 500, 4: 820, 5: 700, 6: 500}))
    arm.read_positions()
    arm.go_home()
    flt = GestureFilter(CONFIG["filter"], resolver=store.resolve, is_continuous=store.is_continuous,
                        conf_for=store.min_confidence)
    streamer = MotionStreamer(arm, CONFIG["arm"])
    return arm, flt, streamer, Teleop(arm, flt, CONFIG, streamer, store)


def hold_gesture(flt, teleop, streamer, label, seconds, t0=0.0, tick=0.05):
    """Predictions every 50 ms (like the model) and a control tick each; returns end time and statuses."""
    t, out = t0, []
    for _ in range(int(seconds / tick)):
        flt.add_prediction(label, 0.95, t)
        out.append(teleop.tick(t))
        streamer.step(tick, t + tick)
        t += tick
    return t, out


class MappingExecutionTests(unittest.TestCase):
    def test_remapped_gesture_moves_the_assigned_servo_at_its_speed(self):
        store = MappingStore(CONFIG)
        store.mappings["victory"] = GestureMapping("victory", True, "move_servo", 2, 1, speed=40)
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        start = arm.targets[2]
        _t, statuses = hold_gesture(flt, teleop, streamer, "Victory Sign", 1.5)
        self.assertGreater(arm.targets[2], start)
        top = max(abs(v) for s in statuses for v in s["velocities"].values())
        self.assertLessEqual(top, 40 + 1e-6, "per-gesture speed limit applies")

    def test_gripper_gestures(self):
        store = MappingStore(CONFIG)
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        t, _ = hold_gesture(flt, teleop, streamer, "Pinch", 1.5)           # default: close gripper
        self.assertTrue(arm.gripper_closed)
        t, _ = hold_gesture(flt, teleop, streamer, "Rest", 0.5, t)
        t, _ = hold_gesture(flt, teleop, streamer, "Victory Sign", 2.5, t)   # default: open gripper
        self.assertFalse(arm.gripper_closed)

    def test_move_to_goes_to_a_safe_fixed_position(self):
        store = MappingStore(CONFIG)
        store.mappings["victory"] = GestureMapping("victory", True, "move_to", 6, target_position=700, duration_ms=1500, hold_s=0.4)
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        _t, statuses = hold_gesture(flt, teleop, streamer, "Victory Sign", 1.5)
        self.assertEqual(arm.targets[6], 700)
        self.assertEqual(sum(1 for s in statuses if s["goto"]), 1, "fires once while the gesture stays detected")

    def test_go_home_is_requested_not_executed_by_teleop(self):
        store = MappingStore(CONFIG)
        store.mappings["double_fist"].enabled = True
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        _t, statuses = hold_gesture(flt, teleop, streamer, "Double Closed Fist", 2.5)
        self.assertEqual(sum(1 for s in statuses if s["home"]), 1)

    def test_disabled_gesture_is_inert(self):
        store = MappingStore(CONFIG)
        store.mappings["left"].enabled = False
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        start = dict(arm.targets)
        hold_gesture(flt, teleop, streamer, "Left", 1.5)
        self.assertEqual(arm.targets, start)

    def test_reversal_guard_blocks_an_instant_direction_flip(self):
        store = MappingStore(CONFIG)
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        teleop._last_dir[6] = (1, 10.0)                    # base was just driven positive by a gesture
        self.assertTrue(teleop._reversal_blocked(6, -1, 10.1))
        self.assertFalse(teleop._reversal_blocked(6, -1, 10.5))
        self.assertFalse(teleop._reversal_blocked(6, 1, 10.1))

    def test_modes(self):
        store = MappingStore(CONFIG)
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        teleop.mode = "gestures"
        status = teleop.tick(0.0, key_motions={"left": 1.0}, emg_ok=True)
        self.assertIsNone(status["motion"], "keyboard ignored in gestures-only mode")
        teleop.mode = "keyboard"
        status = teleop.tick(0.1, key_motions={"left": 1.0}, emg_ok=True)
        self.assertEqual(status["source"], "keyboard")

    def test_stale_gesture_after_link_loss_cannot_move_the_arm(self):
        store = MappingStore(CONFIG)
        arm, flt, streamer, teleop = make_teleop(store)
        teleop.armed = True
        t, _ = hold_gesture(flt, teleop, streamer, "Left", 1.0)
        self.assertTrue(streamer.is_moving())
        status = teleop.tick(t, emg_ok=False)              # the window passes emg_ok=False when the link is not streaming
        self.assertIn("lost", status["note"])
        for _ in range(15):
            t += 0.04
            streamer.step(0.04, t)
        self.assertFalse(streamer.is_moving())
        # even if the link flag were wrong, predictions older than the timeout produce no motion
        status = teleop.tick(t + 5.0, emg_ok=True)
        self.assertEqual(status["velocities"], {})


if __name__ == "__main__":
    unittest.main()
