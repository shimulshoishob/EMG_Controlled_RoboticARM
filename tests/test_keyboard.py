"""Keyboard control through the real window with Qt key events (offscreen platform).

These exercise the same event path as a real keyboard on macOS and Windows (application event filter,
press/release, auto-repeat, modifiers, focus). They cannot reproduce the operating system's own key
delivery (Cocoa / Win32); see the README hardware/OS checklist for the manual part.

    python -m unittest tests.test_keyboard -v
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PyQt5.QtCore import QEvent, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from emg_arm.arm.driver import load_config  # noqa: E402
from emg_arm.settings import Settings  # noqa: E402
from emg_arm.ui import theme  # noqa: E402
from emg_arm.ui.keyboard import KeyMap, build_actions, key_from_name, key_name  # noqa: E402
from emg_arm.ui.main_window import MainWindow  # noqa: E402

APP = QApplication.instance() or QApplication([])
CONFIG = load_config()


def pump(seconds=0.05):
    end = time.time() + seconds
    while time.time() < end:
        APP.processEvents()
        time.sleep(0.005)


class KeyMapTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "keymap.json"
        self.km = KeyMap(build_actions(CONFIG), self.path)

    def test_defaults_include_arrows_and_wasd(self):
        for key, action in ((Qt.Key_Left, "base_left"), (Qt.Key_Right, "base_right"), (Qt.Key_Up, "joint_up"),
                            (Qt.Key_Down, "joint_down"), (Qt.Key_A, "base_left"), (Qt.Key_D, "base_right"),
                            (Qt.Key_W, "shoulder_up"), (Qt.Key_S, "shoulder_down"), (Qt.Key_E, "elbow_up"),
                            (Qt.Key_Q, "elbow_down"), (Qt.Key_G, "gripper_toggle")):
            self.assertEqual(self.km.action_for(key).id, action, key_name(key))

    def test_space_and_escape_are_reserved(self):
        for key in (Qt.Key_Space, Qt.Key_Escape):
            with self.assertRaises(ValueError):
                self.km.assign("shoulder_up", [key])

    def test_conflicts_are_rejected_and_nothing_changes(self):
        before = self.km.keys_for("shoulder_up")
        with self.assertRaises(ValueError):
            self.km.assign("shoulder_up", [Qt.Key_Left])
        self.assertEqual(self.km.keys_for("shoulder_up"), before)

    def test_remap_persists(self):
        self.km.assign("shoulder_up", [Qt.Key_I])
        again = KeyMap(build_actions(CONFIG), self.path)
        self.assertEqual(again.keys_for("shoulder_up"), [Qt.Key_I])
        self.assertIsNone(again.action_for(Qt.Key_W))

    def test_corrupt_file_falls_back(self):
        self.path.write_text("{", encoding="utf-8")
        km = KeyMap(build_actions(CONFIG), self.path)
        self.assertEqual(km.action_for(Qt.Key_Left).id, "base_left")
        self.assertIn("ignored", km.load_warning)

    def test_key_names_round_trip(self):
        for key in (Qt.Key_Left, Qt.Key_A, Qt.Key_1, Qt.Key_F5):
            self.assertEqual(key_from_name(key_name(key)), key)
        with self.assertRaises(ValueError):
            key_from_name("Ctrl+A")


class KeyboardControlTests(unittest.TestCase):
    def setUp(self):
        theme.apply_theme(APP, "light")
        self.dir = Path(tempfile.mkdtemp())
        self.w = MainWindow(load_config(), Settings(self.dir / "s.json"))
        self.w.show()
        self.w.activateWindow()
        pump(0.1)
        self.addCleanup(self._close)
        self.kb = self.w.keyboard
        self.cp = self.w.pages["control"]
        self.cp.mode.setCurrentIndex(self.cp.mode.findData("keyboard"))

    def _close(self):
        self.w.close_ready = True
        self.w.close()

    def settle(self, timeout=5.0):
        end = time.time() + timeout
        while self.w.prep and time.time() < end:
            pump(0.05)
        self.assertIsNone(self.w.prep)

    def arm(self):
        self.w.toggle_arm()                                  # simulated arm (no xArm plugged in)
        self.settle()                                        # the rest routine runs by itself after connecting
        self.w.toggle_armed()
        self.settle()                                        # ... and again before motion is enabled
        self.assertTrue(self.w.is_armed())
        self.w.go_to("control")
        pump(0.05)

    def test_window_is_active_for_these_tests(self):
        self.assertTrue(self.w.isActiveWindow(), "offscreen test window must be active for key tests")

    def test_arrows_and_wasd_move_and_release_stops(self):
        self.arm()
        cases = {Qt.Key_Left: 6, Qt.Key_A: 6, Qt.Key_W: 5, Qt.Key_S: 5, Qt.Key_E: 4, Qt.Key_Q: 4,
                 Qt.Key_Up: 5, Qt.Key_Down: 5, Qt.Key_Right: 6, Qt.Key_D: 6}
        for key, servo in cases.items():
            before = self.w.arm.targets[servo]
            QTest.keyPress(self.w, key)
            self.assertTrue(self.kb.pressed, f"{key_name(key)} should be held")
            pump(0.35)
            QTest.keyRelease(self.w, key)
            self.assertFalse(self.kb.pressed, f"{key_name(key)} release must clear the held state")
            pump(0.6)                                     # the jerk-limited stop from 100 u/s takes up to ~0.3 s
            moved = self.w.arm.targets[servo]
            self.assertNotEqual(moved, before, f"{key_name(key)} should move servo {servo}")
            after_release = self.w.arm.targets[servo]
            pump(0.4)
            self.assertEqual(self.w.arm.targets[servo], after_release, f"{key_name(key)}: arm keeps moving after release")

    def test_left_and_right_go_opposite_ways(self):
        self.arm()
        start = self.w.arm.targets[6]
        QTest.keyPress(self.w, Qt.Key_A)
        pump(0.4)
        QTest.keyRelease(self.w, Qt.Key_A)
        pump(0.3)
        left_pos = self.w.arm.targets[6]
        QTest.keyPress(self.w, Qt.Key_D)
        pump(0.4)
        QTest.keyRelease(self.w, Qt.Key_D)
        pump(0.3)
        self.assertGreater(left_pos, start)               # Left = +direction for the base (config)
        self.assertLess(self.w.arm.targets[6], left_pos)

    def test_space_stops_immediately_and_disarms(self):
        self.arm()
        QTest.keyPress(self.w, Qt.Key_A)
        pump(0.3)
        QTest.keyPress(self.w, Qt.Key_Space)
        pump(0.1)
        self.assertFalse(self.w.is_armed())
        self.assertFalse(self.kb.pressed)

    def test_escape_stops(self):
        self.arm()
        QTest.keyClick(self.w, Qt.Key_Escape)
        self.assertFalse(self.w.is_armed())

    def test_autorepeat_never_changes_held_state(self):
        self.arm()
        ev_press = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a", False)
        ev_rep_rel = QKeyEvent(QEvent.KeyRelease, Qt.Key_A, Qt.NoModifier, "a", True)
        ev_rep_press = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a", True)
        self.assertTrue(self.kb.handle(ev_press, True))
        since = self.kb.pressed[Qt.Key_A]
        for _ in range(5):                                # Cocoa/Qt deliver repeat as release+press pairs
            self.assertTrue(self.kb.handle(ev_rep_rel, False))
            self.assertTrue(self.kb.handle(ev_rep_press, True))
        self.assertEqual(self.kb.pressed.get(Qt.Key_A), since)
        self.assertTrue(self.kb.handle(QKeyEvent(QEvent.KeyRelease, Qt.Key_A, Qt.NoModifier, "a", False), False))
        self.assertFalse(self.kb.pressed)

    def test_command_or_ctrl_combinations_are_ignored(self):
        self.arm()
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.ControlModifier, "w", False)   # Cmd+W on macOS
        self.assertFalse(self.kb.handle(ev, True))
        self.assertFalse(self.kb.pressed)

    def test_disarmed_keys_pass_through_to_widgets(self):
        self.w.go_to("control")
        pump(0.05)
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier, "", False)
        self.assertFalse(self.kb.handle(ev, True), "disarmed arrow keys must not be swallowed")
        self.assertFalse(self.kb.pressed)

    def test_text_field_focus_keeps_keys_for_typing(self):
        self.arm()
        self.w.go_to("data")
        pump(0.1)
        edit = self.w.pages["data"].label_edit
        edit.setFocus()
        pump(0.1)
        self.assertIs(QApplication.focusWidget(), edit)
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a", False)
        self.assertFalse(self.kb.handle(ev, True))
        QTest.keyClicks(edit, "wasd")
        self.assertEqual(edit.text(), "wasd")
        self.assertFalse(self.kb.pressed)
        QTest.keyPress(edit, Qt.Key_Space)               # STOP still wins while armed
        self.assertFalse(self.w.is_armed())

    def test_losing_focus_releases_held_keys(self):
        self.arm()
        QTest.keyPress(self.w, Qt.Key_A)
        self.assertTrue(self.kb.pressed)
        QApplication.sendEvent(self.w, QEvent(QEvent.WindowDeactivate))
        self.assertFalse(self.kb.pressed, "a key released while another window had focus must not stay held")

    def test_keys_combine(self):
        self.arm()
        start = dict(self.w.arm.targets)
        QTest.keyPress(self.w, Qt.Key_A)
        QTest.keyPress(self.w, Qt.Key_W)
        pump(0.5)
        QTest.keyRelease(self.w, Qt.Key_A)
        QTest.keyRelease(self.w, Qt.Key_W)
        pump(0.2)
        self.assertNotEqual(self.w.arm.targets[6], start[6])
        self.assertNotEqual(self.w.arm.targets[5], start[5])

    def test_gripper_and_joint_select_keys(self):
        self.arm()
        before = self.w.arm.gripper_closed
        QTest.keyClick(self.w, Qt.Key_G)
        pump(0.2)
        self.assertNotEqual(self.w.arm.gripper_closed, before)
        QTest.keyClick(self.w, Qt.Key_2)
        pump(0.2)
        self.assertEqual(self.cp.vertical.currentData(), "elbow")

    def test_remapped_key_works_and_old_key_does_not(self):
        self.w.keymap.assign("shoulder_up", [Qt.Key_I])
        self.arm()
        before = self.w.arm.targets[5]
        QTest.keyPress(self.w, Qt.Key_W)
        pump(0.3)
        QTest.keyRelease(self.w, Qt.Key_W)
        self.assertEqual(self.w.arm.targets[5], before)
        QTest.keyPress(self.w, Qt.Key_I)
        pump(0.4)
        QTest.keyRelease(self.w, Qt.Key_I)
        pump(0.2)
        self.assertNotEqual(self.w.arm.targets[5], before)

    def test_key_tester_reports_unassigned_and_stop_keys(self):
        seen = []
        self.kb.key_seen.connect(seen.append)
        QTest.keyClick(self.w, Qt.Key_Z)
        self.assertTrue(any("not assigned" in s for s in seen))
        QTest.keyClick(self.w, Qt.Key_Left)
        self.assertTrue(any("Base rotate left" in s for s in seen))


if __name__ == "__main__":
    unittest.main()
