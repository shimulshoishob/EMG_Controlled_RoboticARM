"""Boots the real window headless (offscreen Qt) and runs the virtual armband through it.

    python -m unittest tests.test_ui_smoke -v
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
from emg_arm.settings import Settings  # noqa: E402
from emg_arm.ui import theme  # noqa: E402
from emg_arm.ui.main_window import PAGES, MainWindow  # noqa: E402

APP = QApplication.instance() or QApplication([])


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        APP.processEvents()
        time.sleep(0.01)


class WindowSmokeTests(unittest.TestCase):
    def setUp(self):
        theme.apply_theme(APP, "light")
        self.window = MainWindow(load_config(), Settings(Path(tempfile.mkdtemp()) / "s.json"))
        self.window.show()
        self.addCleanup(self._close)

    def _close(self):
        self.window.close_ready = True
        self.window.close()

    def test_every_page_opens_in_both_themes(self):
        for mode in ("light", "dark", "pink"):
            theme.apply_theme(APP, mode)
            for key, *_ in PAGES:
                self.window.go_to(key)
                pump(0.05)
                self.assertTrue(self.window.stack.currentWidget().isVisible())

    def test_virtual_armband_streams_through_the_window(self):
        page = self.window.pages["armband"]
        page.mode.set_current("virtual")
        page._mode_changed("virtual")
        page.toggle_connection()
        end = time.time() + 8
        while time.time() < end and self.window.link_state != LinkState.STREAMING:
            pump(0.05)
        self.assertEqual(self.window.link_state, LinkState.STREAMING, page.banner.text.text())
        pump(2.0)
        stats = self.window.bridge.link.stats
        self.assertGreater(stats.frames_per_s, 400)
        self.assertEqual(stats.lost_packets, 0)
        page.toggle_connection()                          # disconnect
        end = time.time() + 8
        while time.time() < end and self.window.link_state != LinkState.IDLE:
            pump(0.05)
        self.assertEqual(self.window.link_state, LinkState.IDLE)
        self.assertFalse(self.window.bridge.active)

    def test_wrong_key_shows_actionable_message_and_cleans_up(self):
        from emg_arm.armband.fake import FakeArmband
        dev = FakeArmband("k" * 40).start()
        self.addCleanup(dev.stop)
        page = self.window.pages["armband"]
        page.ip_edit.setText("127.0.0.1")
        page.key_edit.setText("w" * 40)
        page.toggle_connection()
        end = time.time() + 8
        while time.time() < end and not page._failure:
            pump(0.05)
        self.assertIn("access key", page._failure.lower())
        pump(1.0)
        self.assertFalse(self.window.bridge.active)       # failed links are torn down, the user can retry
        self.assertIn("access key", page.banner.text.text().lower())

    def test_stop_button_disarms_simulated_arm(self):
        w = self.window
        w.pages["control"].emg_check.setChecked(False)     # otherwise arming asks a modal question
        w.toggle_arm()
        w.arm.homed = True
        w.toggle_armed()
        self.assertTrue(w.teleop.armed)
        w.emergency_stop("test")
        self.assertFalse(w.teleop.armed)

    def test_on_screen_pad_moves_the_simulated_arm(self):
        w = self.window
        cp = w.pages["control"]
        cp.emg_check.setChecked(False)
        w.toggle_arm()
        w.arm.homed = True
        w.toggle_armed()
        self.assertTrue(cp.pad_buttons["left"].isEnabled())
        before = dict(w.arm.targets)[6]
        cp.pad_buttons["left"].pressed.emit()
        for _ in range(30):
            pump(0.05)
        cp.pad_buttons["left"].released.emit()
        self.assertNotEqual(dict(w.arm.targets)[6], before)
        self.assertIn("LEFT", cp.action_label.text() + "LEFT")   # label updates while held
        w.emergency_stop("test")
        self.assertFalse(cp.pad_buttons["left"].isEnabled())

    def test_pad_is_disabled_until_armed(self):
        self.assertFalse(self.window.pages["control"].pad_buttons["up"].isEnabled())


if __name__ == "__main__":
    unittest.main()
