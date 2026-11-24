"""Window-level integration: mapping UI + persistence, safety on link/arm failure, config files.

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
        pump(0.4)                                    # the arm homes by itself after connecting
        w.arm.homed = True
        w.toggle_armed()
        assert w.is_armed()


class NavigationTests(IntegrationBase):
    def test_all_sections_exist_and_open(self):
        w = self.make_window()
        wanted = {"control", "model", "armband", "monitor", "mapping", "data"}
        self.assertEqual({p[0] for p in PAGES}, wanted)
        for key, *_ in PAGES:
            w.go_to(key)
            pump(0.05)
            self.assertTrue(w.stack.currentWidget().isVisible(), key)


class ModelListTests(IntegrationBase):
    """Gesture model page: upload several files, clear them, no folder scanning."""

    def test_upload_several_files_then_clear_keeps_the_files(self):
        w = self.make_window()
        page = w.pages["model"]
        files = []
        for name in ("a", "b", "c"):
            f = self.dir / f"{name}.joblib"
            f.write_bytes(b"x")
            files.append(str(f))
        page.add_files(files + [files[0]])                      # a duplicate is ignored
        self.assertEqual(page.list.count(), 3)
        self.assertEqual(w.settings.get("model_files"), files)
        page._clear()
        self.assertEqual(page.list.count(), 0)
        self.assertEqual(w.settings.get("model_files"), [])
        self.assertEqual(w.settings.get("last_model"), "")
        self.assertTrue(all(Path(f).exists() for f in files), "clearing must never delete files")

    def test_no_rescan_button_and_missing_files_are_hidden(self):
        from PyQt5.QtWidgets import QPushButton
        w = self.make_window()
        page = w.pages["model"]
        texts = [b.text() for b in page.findChildren(QPushButton)]
        self.assertIn("Upload .joblib files…", texts)
        self.assertIn("Clear uploaded files", texts)
        self.assertFalse(any("Rescan" in t or "Browse" in t for t in texts))
        w.settings.set("model_files", [str(self.dir / "gone.joblib")])
        page.refresh()
        self.assertEqual(page.list.count(), 0)


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
        with self.assertLogs("emg_arm", "WARNING") as logged:
            w.on_link_state(LinkState.STALLED.value, "no data")
        self.assertTrue(any("stalled" in line for line in logged.output))
        self.assertIsNone(w.gesture_filter.stable, "stale gesture must be forgotten when the link drops")
        before = dict(w.arm.targets)
        pump(0.6)
        self.assertEqual(w.arm.targets, before, "arm must not move after the link drops")

    def test_arm_communication_failure_stops_and_is_logged(self):
        w = self.make_window()
        self.arm_sim(w, "keyboard")
        w.arm.backend.set_positions = lambda *a, **k: (_ for _ in ()).throw(OSError("USB device disconnected"))
        w.pad_held["left"] = time.monotonic()
        from PyQt5.QtWidgets import QMessageBox
        orig = QMessageBox.critical
        QMessageBox.critical = staticmethod(lambda *a, **k: None)
        self.addCleanup(setattr, QMessageBox, "critical", orig)
        with self.assertLogs("emg_arm", "ERROR"):          # recorded in the background log
            for _ in range(40):
                pump(0.05)
                if not w.is_armed():
                    break
        self.assertFalse(w.is_armed(), "a lost arm connection must disarm")

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


if __name__ == "__main__":
    unittest.main()
