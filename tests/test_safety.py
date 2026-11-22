"""Safety tests that run with no hardware:  python -m unittest discover tests"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emg_arm.arm.driver import ArmSafetyError, SafeArm, SimulatedBackend, load_config  # noqa: E402
from emg_arm.arm.motion import MotionStreamer  # noqa: E402
from emg_arm.arm.teleop import GestureFilter, Teleop, canonical_gesture  # noqa: E402
from emg_arm.control.commands import CommandGenerator  # noqa: E402
from emg_arm.control.mapping import MappingStore  # noqa: E402

CONFIG = load_config()


def make_arm(config=CONFIG, start=None):
    start = start or {1: 100, 2: 500, 4: 820, 5: 700, 6: 500}
    backend = SimulatedBackend(start)
    arm = SafeArm(config, backend)
    arm.read_positions()
    return arm, backend


def run_stream(streamer, velocities, seconds, t0=0.0, dt=0.04):
    """Drive the streamer by hand (no thread), re-commanding like the GUI does."""
    t = t0
    for _ in range(int(round(seconds / dt))):
        if velocities is not None:
            streamer.command(velocities, t)
        t += dt
        streamer.step(dt, t)
    return t


def feed(gesture_filter, label, t0, n, conf=0.95, step=0.05):
    t = t0
    for _ in range(n):
        gesture_filter.add_prediction(label, conf, t)
        t += step
    return t


class ConfigTests(unittest.TestCase):
    def test_shipped_config_passes_validation(self):
        make_arm()

    def test_home_outside_limits_rejected(self):
        bad = copy.deepcopy(CONFIG)
        bad["arm"]["home"]["5"] = 900
        with self.assertRaises(ArmSafetyError):
            SafeArm(bad, SimulatedBackend())

    def test_gripper_outside_limits_rejected(self):
        bad = copy.deepcopy(CONFIG)
        bad["arm"]["gripper"]["close"] = 900
        with self.assertRaises(ArmSafetyError):
            SafeArm(bad, SimulatedBackend())


class DriverTests(unittest.TestCase):
    def test_no_motion_before_home(self):
        arm, _ = make_arm()
        with self.assertRaises(ArmSafetyError):
            arm.stream({6: 510.0}, {}, 120)
        with self.assertRaises(ArmSafetyError):
            arm.set_gripper(True)
        streamer = MotionStreamer(arm, CONFIG["arm"])
        run_stream(streamer, {6: 100}, 0.2)
        self.assertIsInstance(streamer.error, ArmSafetyError)
        self.assertFalse(streamer.is_moving())

    def test_unconfigured_servo_refused(self):
        arm, _ = make_arm()
        with self.assertRaises(ArmSafetyError):
            arm._send(3, 500, 1000)

    def test_positions_are_python_ints_and_clamped(self):
        arm, backend = make_arm()
        arm.go_home()
        streamer = MotionStreamer(arm, CONFIG["arm"])
        run_stream(streamer, {6: arm.hard_max_speed}, 10.0)
        lo, hi = arm.limits[6]
        self.assertEqual(arm.targets[6], hi)
        for sid, pos, ms in backend.commands:
            self.assertIs(type(pos), int)
            self.assertIs(type(ms), int)
            self.assertGreaterEqual(pos, arm.limits[sid][0])
            self.assertLessEqual(pos, arm.limits[sid][1])
        self.assertTrue(arm.at_limit(6, 1), "should report the limit")
        self.assertFalse(streamer.is_moving(), "must stop at the limit")

    def test_speed_never_exceeds_hard_cap(self):
        arm, backend = make_arm()
        arm.set_speed(10_000)  # request far above the cap
        self.assertEqual(arm.speed, arm.hard_max_speed)
        arm.go_home()
        streamer = MotionStreamer(arm, CONFIG["arm"])
        last = dict(arm.targets)
        backend.commands.clear()
        run_stream(streamer, {5: -10_000, 6: 10_000}, 1.5)
        prev = dict(last)
        for sid, pos, ms in backend.commands:
            speed = abs(pos - prev[sid]) / (ms / 1000.0)
            self.assertLessEqual(speed, arm.hard_max_speed + 1e-6)
            prev[sid] = pos

    def test_home_from_unknown_position_is_slow(self):
        backend = SimulatedBackend({})  # nothing readable
        arm = SafeArm(CONFIG, backend)
        arm.read_positions()
        seconds = arm.go_home()
        self.assertGreaterEqual(seconds, 1000 / arm.hard_max_speed)
        for _sid, _pos, ms in backend.commands:
            self.assertGreaterEqual(ms, 1000 / arm.hard_max_speed * 1000)

    def test_home_is_one_packet(self):
        # The real xArm drops all but the last of back-to-back commands.
        arm, backend = make_arm(start={1: 100, 2: 505, 4: 967, 5: 679, 6: 499})
        arm.go_home()
        self.assertEqual(len(backend.packets), 1)
        self.assertEqual(backend.packets[0][0], {2: 500, 4: 820, 5: 700, 6: 500})

    def test_gripper_initial_state_inferred(self):
        arm, _ = make_arm(start={1: 500, 2: 500, 4: 820, 5: 700, 6: 500})
        self.assertTrue(arm.gripper_closed)


def new_generator(**disabled):
    """CommandGenerator on the default mapping; keyword args name gestures to disable."""
    store = MappingStore(CONFIG)
    for gid in disabled:
        store.mappings[gid].enabled = False
    return CommandGenerator(store, CONFIG["filter"]["grip_cooldown_s"]), store


def make_filter(store=None):
    store = store or MappingStore(CONFIG)
    return GestureFilter(CONFIG["filter"], resolver=store.resolve, is_continuous=store.is_continuous,
                         conf_for=store.min_confidence)


def moves(gen, f, t):
    return [m.gesture_id for m in gen.generate(f, t).moves]


def shots(gen, f, t):
    return [o.action for o in gen.generate(f, t).oneshots]


class GestureTests(unittest.TestCase):
    def test_label_names_from_trained_models(self):
        cases = {"Left": "left", "Right": "right", "up": "up", "Move_Up": "up", "down": "down",
                 "Move_Down": "down", "dow": "down", "fist_close": "closed_fist", "Fist_Close": "closed_fist",
                 "Closed Fist": "closed_fist", "Double Closed Fist": "double_fist", "Victory Sign": "victory",
                 "Thumbs Up": "thumbs_up", "Pinch using thumb and index finger": "pinch", "Rest": "rest",
                 "rest": "rest", "wave_out": None}
        for label, expected in cases.items():
            self.assertEqual(canonical_gesture(label), expected, label)

    def test_needs_stable_majority(self):
        f = make_filter()
        gen, _ = new_generator()
        f.add_prediction("Left", 0.95, 0.0)
        self.assertEqual(moves(gen, f, 0.0), [], "one prediction must not move the arm")
        t = feed(f, "Left", 0.05, 4)
        self.assertEqual(moves(gen, f, t), ["left"])

    def test_low_confidence_ignored(self):
        f = make_filter()
        gen, _ = new_generator()
        t = feed(f, "Left", 0.0, 10, conf=0.4)
        self.assertEqual(moves(gen, f, t), [])

    def test_stale_predictions_stop_motion(self):
        f = make_filter()
        gen, _ = new_generator()
        t = feed(f, "Left", 0.0, 6)
        self.assertEqual(moves(gen, f, t), ["left"])
        self.assertEqual(moves(gen, f, t + 1.0), [])

    def test_stuck_gesture_locks_out_until_change(self):
        f = make_filter()
        gen, _ = new_generator()
        t = 0.0
        for _ in range(200):                           # 10 s of continuous Left
            t = feed(f, "Left", t, 1)
            gen.generate(f, t)
        self.assertEqual(moves(gen, f, t), [])
        t = feed(f, "Rest", t, 6)
        t = feed(f, "Left", t, 6)
        self.assertEqual(moves(gen, f, t), ["left"])

    def test_short_fist_does_not_fire(self):
        f = make_filter()
        gen, _ = new_generator()
        t = feed(f, "fist_close", 0.0, 6)  # ~0.1 s stable
        self.assertEqual(shots(gen, f, t), [])

    def test_per_gesture_confidence_threshold(self):
        store = MappingStore(CONFIG)
        store.mappings["left"].min_confidence = 0.9
        f = make_filter(store)
        t = feed(f, "Left", 0.0, 10, conf=0.8)         # above the global 0.65, below Left's own 0.9
        self.assertIsNone(f.stable)
        t = feed(f, "Left", t, 10, conf=0.95)
        self.assertEqual(f.stable, "left")


class GestureFilterSmoothingTests(unittest.TestCase):
    def test_switch_labels(self):
        self.assertEqual(canonical_gesture("Thumbs_Up"), "thumbs_up")
        self.assertEqual(canonical_gesture("Fist_Open_Fast"), "thumbs_up")

    def test_single_flicker_keeps_motion(self):
        f = make_filter()
        gen, _ = new_generator()
        t = feed(f, "Left", 0.0, 6)
        since = f.stable_since
        for label in ("Rest", "Right"):  # two bad votes inside a held Left
            f.add_prediction(label, 0.95, t)
            t += 0.05
        self.assertEqual(moves(gen, f, t), ["left"], "hysteresis should keep the motion")
        self.assertEqual(f.stable_since, since)

    def test_short_dropout_keeps_ramp(self):
        f = make_filter()
        gen, _ = new_generator()
        t = feed(f, "Left", 0.0, 10)
        since = f.stable_since
        t = feed(f, "Rest", t, 3, conf=0.3)  # low-confidence blip
        self.assertEqual(moves(gen, f, t), [])
        t = feed(f, "Left", t, 4)  # back to 4 of 5 votes within the grace time
        self.assertEqual(moves(gen, f, t), ["left"])
        self.assertEqual(f.stable_since, since, "ramp should continue after a short blip")

    def test_fist_release_toggles_gripper(self):
        f = make_filter()
        gen, _ = new_generator(thumbs_up=True)           # no dedicated switch gesture: legacy fist behaviour
        t = 0.0
        for _ in range(20):                              # ~0.85 s stable
            t = feed(f, "fist_close", t, 1)
            self.assertEqual(shots(gen, f, t), [], "fires on release, not while held")
        t = feed(f, "Rest", t, 5)
        fired = []
        for _ in range(3):
            fired += shots(gen, f, t)
            t += 0.05
        self.assertEqual(fired, ["gripper_toggle"])

    def test_long_fist_switches_joint_not_gripper(self):
        f = make_filter()
        gen, _ = new_generator(thumbs_up=True)
        fired, t = [], 0.0
        for _ in range(60):  # 3 s fist
            f.add_prediction("fist_close", 0.95, t)
            fired += shots(gen, f, t)
            t += 0.05
        self.assertEqual(fired, ["switch_joint"])
        t = feed(f, "Rest", t, 5)
        self.assertEqual(shots(gen, f, t), [], "no gripper toggle after a long hold")

    def test_long_fist_defers_to_a_dedicated_switch_gesture(self):
        f = make_filter()
        gen, _ = new_generator()                         # thumbs_up -> switch_joint is enabled
        gen.available = None
        fired, t = [], 0.0
        for _ in range(60):
            f.add_prediction("fist_close", 0.95, t)
            fired += shots(gen, f, t)
            t += 0.05
        self.assertEqual(fired, ["gripper_toggle"], "fires once at the hold time, joint switch comes from Thumbs Up")

    def test_gesture_stays_detected_fires_once(self):
        f = make_filter()
        gen, _ = new_generator()
        fired, t = [], 0.0
        for _ in range(200):                             # 10 s with the fist held
            f.add_prediction("fist_close", 0.95, t)
            fired += shots(gen, f, t)
            t += 0.05
        self.assertEqual(len(fired), 1)

    def test_oneshot_cooldown_blocks_rapid_retrigger(self):
        f = make_filter()
        gen, _ = new_generator()
        fired, t = [], 0.0
        for _ in range(3):                               # fist, rest, fist, rest, ... faster than the cooldown
            for _ in range(16):
                f.add_prediction("fist_close", 0.95, t)
                fired += shots(gen, f, t)
                t += 0.05
            for _ in range(8):
                f.add_prediction("rest", 0.95, t)
                fired += shots(gen, f, t)
                t += 0.05
        # cycles repeat every 1.2 s: the 2nd trigger lands inside the 1.5 s cooldown of the 1st and is
        # blocked; the 3rd is 2.4 s after the 1st and is allowed.
        self.assertEqual(len(fired), 2)

    def test_disabled_mapping_does_nothing(self):
        f = make_filter()
        gen, _ = new_generator(left=True)
        t = feed(f, "Left", 0.0, 8)
        self.assertEqual(moves(gen, f, t), [])


class StreamerTests(unittest.TestCase):
    def setUp(self):
        self.arm, self.backend = make_arm()
        self.arm.go_home()
        self.backend.packets.clear()
        self.streamer = MotionStreamer(self.arm, CONFIG["arm"])

    def test_constant_speed_and_one_packet_per_cycle(self):
        self.arm.set_speed(100)
        t = run_stream(self.streamer, {5: -100, 6: 100}, 1.0)
        self.assertEqual(len(self.backend.packets), 25, "one packet per 40 ms cycle")
        self.assertTrue(all(set(p) == {5, 6} for p, _ in self.backend.packets[3:]),
                        "both joints move together in one packet")
        before = self.arm.setpoint(6)
        run_stream(self.streamer, {5: -100, 6: 100}, 0.4, t0=t)
        self.assertAlmostEqual(self.arm.setpoint(6) - before, 40.0, delta=1.0)

    def test_soft_start_and_stop(self):
        run_stream(self.streamer, {6: 100}, 0.04)
        first_step = self.arm.setpoint(6) - 500
        self.assertLess(first_step, 100 * 0.04, "starts gently (acceleration limit)")
        t = run_stream(self.streamer, {6: 100}, 1.0)
        run_stream(self.streamer, {}, 0.5, t0=t)
        self.assertFalse(self.streamer.is_moving())
        self.assertEqual(self.arm.targets[6], round(self.arm.setpoint(6)),
                         "last packet parks the servo on its setpoint")

    def test_halt_parks_immediately(self):
        run_stream(self.streamer, {6: 100}, 1.0)
        self.assertNotEqual(self.arm.targets[6], round(self.arm.setpoint(6)), "aims ahead while moving")
        self.streamer.halt()
        self.assertFalse(self.streamer.is_moving())
        self.assertEqual(self.arm.targets[6], round(self.arm.setpoint(6)))
        n = len(self.backend.packets)
        self.streamer.step(0.04, 10.0)
        self.assertEqual(len(self.backend.packets), n, "nothing moves after halt")

    def test_stale_commands_stop_the_arm(self):
        self.streamer.command({6: 100}, 0.0)
        t = 0.0
        for _ in range(50):  # GUI frozen for 2 s: no new commands
            t += 0.04
            self.streamer.step(0.04, t)
        self.assertFalse(self.streamer.is_moving())
        self.assertLess(self.arm.setpoint(6) - 500, 100 * 0.5)

    def test_bad_stream_config_rejected(self):
        with self.assertRaises(ArmSafetyError):
            MotionStreamer(self.arm, dict(CONFIG["arm"], stream_period_ms=5))
        with self.assertRaises(ArmSafetyError):
            MotionStreamer(self.arm, dict(CONFIG["arm"], lookahead_ms=2000))


class TeleopTests(unittest.TestCase):
    def setUp(self):
        self.arm, self.backend = make_arm()
        self.arm.go_home()
        self.store = MappingStore(CONFIG)
        self.filter = make_filter(self.store)
        self.streamer = MotionStreamer(self.arm, CONFIG["arm"])
        self.teleop = Teleop(self.arm, self.filter, CONFIG, self.streamer, self.store)

    def tick(self, t, n=1, **kw):
        for _ in range(n):
            self.teleop.tick(t, **kw)
            t += 0.04
            self.streamer.step(0.04, t)
        return t

    def test_disarmed_never_moves(self):
        self.backend.commands.clear()
        t = feed(self.filter, "Left", 0.0, 10)
        self.tick(t, 10, key_motions={"right": 1.0})
        self.assertEqual(self.backend.commands, [])

    def test_left_rotates_base(self):
        self.teleop.armed = True
        start = self.arm.targets[6]
        t = feed(self.filter, "Left", 0.0, 6)
        for _ in range(10):
            t = self.tick(t)
            t = feed(self.filter, "Left", t, 1, step=0.0)
        self.assertGreater(self.arm.targets[6], start)

    def test_keys_combine_joints(self):
        self.teleop.armed = True
        start = dict(self.arm.targets)
        self.tick(0.0, 15, key_motions={"left": 0.5, "shoulder_down": 0.5, "elbow_up": 0.5})
        self.assertGreater(self.arm.targets[6], start[6])
        self.assertLess(self.arm.targets[5], start[5])
        self.assertLess(self.arm.targets[4], start[4])  # elbow up_direction is -1

    def test_vertical_joint_selection(self):
        self.teleop.armed = True
        self.teleop.vertical = "elbow"
        self.assertEqual(self.teleop.servo_for("up"), (4, -1))
        self.teleop.vertical = "shoulder"
        self.assertEqual(self.teleop.servo_for("up"), (5, 1))
        self.assertEqual(self.teleop.servo_for("elbow_down"), (4, 1))
        self.assertEqual(self.teleop.switch_vertical(), "elbow")

    def test_emg_long_fist_switches_joint(self):
        self.store.mappings["thumbs_up"].enabled = False     # no dedicated switch gesture
        self.teleop.armed = True
        t, switched = 0.0, None
        for _ in range(60):
            self.filter.add_prediction("fist_close", 0.95, t)
            status = self.teleop.tick(t)
            switched = switched or status["switched"]
            t += 0.05
        self.assertEqual(switched, "elbow")
        self.assertEqual(self.teleop.vertical, "elbow")

    def test_emg_loss_holds(self):
        self.teleop.armed = True
        t = feed(self.filter, "Left", 0.0, 6)
        for _ in range(10):
            t = self.tick(t)
            t = feed(self.filter, "Left", t, 1, step=0.0)
        self.assertTrue(self.streamer.is_moving())
        status = self.teleop.tick(t, emg_ok=False)
        self.assertIn("lost", status["note"])
        for _ in range(10):
            t += 0.04
            self.streamer.step(0.04, t)
        self.assertFalse(self.streamer.is_moving())

    def test_ui_stall_does_not_jump(self):
        self.teleop.armed = True
        start = self.arm.targets[6]
        self.teleop.tick(0.0, key_motions={"left": 1.0})
        self.streamer.step(5.0, 0.01)  # 5 s stall
        self.assertLessEqual(abs(self.arm.setpoint(6) - start), self.arm.speed * 0.1 + 1)


if __name__ == "__main__":
    unittest.main()
