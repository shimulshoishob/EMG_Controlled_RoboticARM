"""Window-level integration: mapping UI + persistence, safety on link/arm failure, diagnostics, config files.

    python -m unittest tests.test_integration -v
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from emg_arm.arm.driver import load_config  # noqa: E402
from emg_arm.armband.link import LinkState  # noqa: E402
from emg_arm.config_bundle import export_bundle, import_bundle  # noqa: E402
from emg_arm.control.gestures import GESTURES  # noqa: E402
from emg_arm.settings import Settings  # noqa: E402
from emg_arm.ui import theme  # noqa: E402
from emg_arm.ui.main_window import PAGES, MainWindow  # noqa: E402

APP = QApplication.instance() or QApplication([])


def pump(seconds=0.05):
    end = time.time() + seconds
    while time.time() < end:
        APP.processEvents()
        time.sleep(0.005)


class IntegrationBase(unittest.TestCase):
    def make_window(self, folder=None):
        theme.apply_theme(APP, "light")
        self.dir = folder or Path(tempfile.mkdtemp())
        w = MainWindow(load_config(), Settings(self.dir / "s.json"))
        w.show()
        w.activateWindow()
        pump(0.1)
        self.addCleanup(self._close, w)
        return w

    @staticmethod
    def _close(w):
        w.close_ready = True
        w.close()

    @staticmethod
    def arm_sim(w, mode="gestures"):
        cp = w.pages["control"]
        cp.mode.setCurrentIndex(cp.mode.findData(mode))
        if mode != "keyboard":                       # otherwise arming opens a modal "gestures not available" question
            w.link_state, w.model_loaded = LinkState.STREAMING, True
        w.toggle_arm()
        w.arm.homed = True
        w.toggle_armed()
        assert w.is_armed()


class NavigationTests(IntegrationBase):
    def test_all_sections_exist_and_open(self):
        w = self.make_window()
        wanted = {"dashboard", "model", "armband", "arm", "control", "monitor", "mapping", "keyboard", "diagnostics", "data"}
        self.assertEqual({p[0] for p in PAGES}, wanted)
        for key, *_ in PAGES:
            w.go_to(key)
            pump(0.05)
            self.assertTrue(w.stack.currentWidget().isVisible(), key)

    def test_dashboard_reports_every_required_status(self):
        w = self.make_window()
        w.refresh_dashboard()
        tiles = w.pages["dashboard"].tiles
        for key in ("armband", "data", "model", "gesture", "arm", "mode", "movement", "health"):
            self.assertIn(key, tiles)
        self.assertEqual(tiles["armband"].value.text(), "Idle")
        self.assertEqual(tiles["mode"].value.text(), "DISARMED")


class MappingUiTests(IntegrationBase):
    def test_all_ten_gestures_are_listed_and_editable_and_persist_across_restart(self):
        w = self.make_window()
        mp = w.pages["mapping"]
        self.assertEqual(mp.list.count(), 10)
        self.assertEqual([g.name for g in GESTURES][4:6], ["Closed Fist", "Double Closed Fist"])
        victory_row = [g.id for g in GESTURES].index("victory")
        mp.list.setCurrentRow(victory_row)
        mp._loading = False
        mp.action.setCurrentIndex(mp.action.findData("move_servo"))
        mp.servo.setCurrentIndex(mp.servo.findData(2))               # wrist rotate
        mp.direction.setCurrentIndex(1)
        mp.speed.setValue(40)
        pinch_row = [g.id for g in GESTURES].index("pinch")
        mp.list.setCurrentRow(pinch_row)
        mp.enabled.setChecked(False)
        self.assertTrue(mp.save_btn.isEnabled())
        mp._save()
        w.close_ready = True
        w.close()
        w2 = self.make_window(self.dir)                              # "restart" with the same settings folder
        m = w2.mapping.mappings
        self.assertEqual((m["victory"].action, m["victory"].servo, m["victory"].direction, m["victory"].speed),
                         ("move_servo", 2, -1, 40.0))
        self.assertFalse(m["pinch"].enabled)

    def test_conflicting_edit_cannot_be_saved(self):
        w = self.make_window()
        mp = w.pages["mapping"]
        mp.list.setCurrentRow([g.id for g in GESTURES].index("victory"))
        mp.action.setCurrentIndex(mp.action.findData("move_servo"))
        mp.servo.setCurrentIndex(mp.servo.findData(6))               # base...
        mp.direction.setCurrentIndex(0)                              # ...positive: same as Left
        self.assertFalse(mp.save_btn.isEnabled())
        self.assertIn("conflicts", mp.issues.text.text())

    def test_saving_is_blocked_while_armed(self):
        w = self.make_window()
        self.arm_sim(w, "keyboard")
        mp = w.pages["mapping"]
        mp.list.setCurrentRow(0)
        mp.hold.setValue(0.9) if mp.hold.isVisible() else mp.speed.setValue(30)
        self.assertFalse(mp.save_btn.isEnabled())

    def test_reset_restores_defaults(self):
        w = self.make_window()
        w.mapping.mappings["pinch"].enabled = False
        w.mapping.save()
        mp = w.pages["mapping"]
        from PyQt5.QtWidgets import QMessageBox
        orig = QMessageBox.question
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        self.addCleanup(setattr, QMessageBox, "question", orig)
        mp._reset()
        self.assertTrue(w.mapping.mappings["pinch"].enabled)


class SafetyTests(IntegrationBase):
    def test_link_loss_stops_gesture_control_and_clears_predictions(self):
        w = self.make_window()
        self.arm_sim(w, "gestures")
        now = time.monotonic()
        for i in range(8):
            w.gesture_filter.add_prediction("Left", 0.95, now + i * 0.05)
        w.on_link_state(LinkState.STALLED.value, "no data")
        self.assertIsNone(w.gesture_filter.stable, "stale gesture must be forgotten when the link drops")
        before = dict(w.arm.targets)
        pump(0.6)
        self.assertEqual(w.arm.targets, before, "arm must not move after the link drops")
        self.assertTrue(any("stalled" in e.message for e in w.diag.recent(20, "warning")))

    def test_arm_communication_failure_stops_and_is_logged(self):
        w = self.make_window()
        self.arm_sim(w, "keyboard")
        w.arm.backend.set_positions = lambda *a, **k: (_ for _ in ()).throw(OSError("USB device disconnected"))
        w.pad_held["left"] = time.monotonic()
        from PyQt5.QtWidgets import QMessageBox
        orig = QMessageBox.critical
        QMessageBox.critical = staticmethod(lambda *a, **k: None)
        self.addCleanup(setattr, QMessageBox, "critical", orig)
        for _ in range(40):
            pump(0.05)
            if not w.is_armed():
                break
        self.assertFalse(w.is_armed(), "a lost arm connection must disarm")
        self.assertTrue(w.diag.errors(), "and be recorded as an error")

    def test_gesture_mode_ignores_keyboard_but_stop_works(self):
        w = self.make_window()
        self.arm_sim(w, "gestures")
        before = dict(w.arm.targets)
        w.pad_held["left"] = time.monotonic()
        pump(0.5)
        self.assertEqual(w.arm.targets, before)
        w.emergency_stop("test")
        self.assertFalse(w.is_armed())


class ConfigFileTests(IntegrationBase):
    def test_bundle_roundtrip_between_installs(self):
        w = self.make_window()
        w.mapping.mappings["pinch"].enabled = False
        w.mapping.save()
        from PyQt5.QtCore import Qt
        w.keymap.assign("shoulder_up", [Qt.Key_I])
        w.settings.set("tuning", {"lookahead": 150.0})
        out = self.dir / "bundle.json"
        export_bundle(out, w.mapping, w.keymap, w.settings)
        w2 = self.make_window()
        import_bundle(out, w2.mapping, w2.keymap, w2.settings)
        self.assertFalse(w2.mapping.mappings["pinch"].enabled)
        self.assertEqual(w2.keymap.keys_for("shoulder_up"), [Qt.Key_I])
        self.assertEqual(w2.settings.get("tuning")["lookahead"], 150.0)

    def test_bad_bundle_changes_nothing(self):
        w = self.make_window()
        bad = self.dir / "bad.json"
        bad.write_text('{"bundle_version": 1, "mapping": {"mappings": {"left": {"action": "move_servo", "servo": 99}}}}')
        with self.assertRaises(ValueError):
            import_bundle(bad, w.mapping, w.keymap, w.settings)
        self.assertEqual(w.mapping.mappings["left"].servo, 6)


class DiagnosticsTests(IntegrationBase):
    def test_link_state_changes_and_errors_reach_the_log(self):
        w = self.make_window()
        w.on_link_state(LinkState.CONNECTING.value, "")
        w.on_link_state(LinkState.FAILED.value, "The access key does not match")
        pump(0.4)
        text = w.pages["diagnostics"].view.toPlainText()
        self.assertIn("connecting → failed", text)
        self.assertIn("access key", w.pages["diagnostics"].errors.toPlainText())


if __name__ == "__main__":
    unittest.main()
