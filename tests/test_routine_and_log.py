"""The automatic rest / gripper routine, the joint-switch beep and the pipeline log (offscreen, simulated arm).

    python -m unittest tests.test_routine_and_log -v
"""

import csv
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from PyQt5.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QWheelEvent  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QAbstractSpinBox, QApplication, QComboBox  # noqa: E402

from emg_arm.arm.driver import SimulatedBackend, load_config  # noqa: E402
from emg_arm.armband.link import LinkState  # noqa: E402
from emg_arm.ml.inference import InferenceWorker  # noqa: E402
from emg_arm.settings import Settings  # noqa: E402
from emg_arm.telemetry import PipelineLog, Timing  # noqa: E402
from emg_arm.ui import main_window, sound, theme  # noqa: E402

APP = QApplication.instance() or QApplication([])


def pump(seconds=0.05):
    end = time.time() + seconds
    while time.time() < end:
        APP.processEvents()
        time.sleep(0.005)


class WindowBase(unittest.TestCase):
    def setUp(self):
        theme.apply_theme(APP, "light")
        folder = Path(tempfile.mkdtemp())
        self.beeps = []
        for target, name, value in ((sound, "beep", lambda: self.beeps.append(1)),
                                    (main_window, "xarm_plugged_in", lambda: False),     # never the real arm
                                    (main_window, "LOG_DIR", folder)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.w = main_window.MainWindow(load_config(), Settings(folder / "s.json"))
        self.w.show()
        self.addCleanup(self._close)
        pump(0.1)

    def _close(self):
        self.w.close_ready = True
        self.w.close()

    def settle(self, timeout=5.0):
        end = time.time() + timeout
        while self.w.prep and time.time() < end:
            pump(0.02)
        self.assertIsNone(self.w.prep)

    def gripper_moves(self, since=0):
        grip = self.w.arm.grip_servo
        return [p[grip] for p, _ms in self.w.arm.backend.packets[since:] if grip in p]


class BeepTests(WindowBase):
    def test_no_beep_at_start_up(self):
        self.assertEqual(self.beeps, [])

    def test_one_beep_per_switch_between_shoulder_and_elbow(self):
        box = self.w.pages["control"].vertical
        box.setCurrentIndex(box.findData("elbow"))
        self.assertEqual(len(self.beeps), 1)
        box.setCurrentIndex(box.findData("elbow"))                  # already selected: no switch, no beep
        self.assertEqual(len(self.beeps), 1)
        box.setCurrentIndex(box.findData("shoulder"))
        self.assertEqual(len(self.beeps), 2)


class InputFieldTests(WindowBase):
    def test_no_arrow_buttons_and_the_wheel_changes_nothing(self):
        boxes = self.w.findChildren(QAbstractSpinBox)
        self.assertGreater(len(boxes), 10)
        for box in boxes:
            self.assertEqual(box.buttonSymbols(), QAbstractSpinBox.NoButtons, box.parent())
        for widget in boxes + self.w.findChildren(QComboBox):
            before = widget.currentIndex() if isinstance(widget, QComboBox) else widget.value()
            for delta in (120, -120):
                wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, delta), Qt.NoButton,
                                    Qt.NoModifier, Qt.NoScrollPhase, False)
                QApplication.sendEvent(widget, wheel)
            self.assertEqual(widget.currentIndex() if isinstance(widget, QComboBox) else widget.value(), before)

    def test_numbers_can_be_typed_while_armed(self):
        w = self.w
        cp = w.pages["control"]
        cp.mode.setCurrentIndex(cp.mode.findData("keyboard"))
        w.activateWindow()
        w.toggle_arm()
        self.settle()
        w.toggle_armed()
        self.settle()
        w.go_to("control")
        pump(0.1)
        cp.speed.setFocus()
        cp.speed.selectAll()
        QTest.keyClicks(cp.speed, "120")
        self.assertEqual(cp.speed.value(), 120)
        QTest.keyClick(w, Qt.Key_E)
        pump(0.15)
        self.assertEqual(cp.vertical.currentData(), "elbow")
        QTest.keyClick(w, Qt.Key_S)
        pump(0.15)
        self.assertEqual(cp.vertical.currentData(), "shoulder")


class RestRoutineTests(WindowBase):
    def test_connecting_at_rest_cycles_the_gripper_once(self):
        w = self.w
        w.toggle_arm()                                              # the simulated arm starts at rest, gripper open
        self.assertIsNotNone(w.prep)
        self.assertFalse(w.can_arm())
        self.settle()
        o, c = w.arm.grip_open, w.arm.grip_close
        self.assertEqual(self.gripper_moves(), [c, o])
        moved = {sid for p, _ms in w.arm.backend.packets for sid in p}
        self.assertEqual(moved, {w.arm.grip_servo}, "only the gripper may move when already at rest")

    def test_arming_away_from_rest_goes_to_rest_without_using_the_gripper(self):
        w = self.w
        w.pages["control"].mode.setCurrentIndex(w.pages["control"].mode.findData("keyboard"))
        w.toggle_arm()
        self.settle()
        w.arm.backend.positions[6] = w.arm.targets[6] = 300         # the base was left elsewhere
        sent = len(w.arm.backend.packets)
        w.toggle_armed()
        self.assertFalse(w.is_armed(), "motion is enabled only after the routine")
        self.settle()
        self.assertTrue(w.is_armed())
        self.assertEqual(w.arm.backend.positions[6], w.arm.home_pose[6])
        self.assertEqual(self.gripper_moves(sent), [])

    def test_arming_at_rest_cycles_the_gripper_then_arms(self):
        w = self.w
        w.pages["control"].mode.setCurrentIndex(w.pages["control"].mode.findData("keyboard"))
        w.toggle_arm()
        self.settle()
        sent = len(w.arm.backend.packets)
        w.toggle_armed()
        self.settle()
        self.assertTrue(w.is_armed())
        self.assertEqual(len(self.gripper_moves(sent)), 2)

    def test_stop_cancels_the_routine_and_does_not_arm(self):
        w = self.w
        w.pages["control"].mode.setCurrentIndex(w.pages["control"].mode.findData("keyboard"))
        w.toggle_arm()
        self.settle()
        w.toggle_armed()
        w.emergency_stop("test")
        self.assertIsNone(w.prep)
        sent = len(w.arm.backend.packets)
        pump(0.8)
        self.assertEqual(len(w.arm.backend.packets), sent, "nothing may move after STOP")
        self.assertFalse(w.is_armed())

    def test_disconnect_during_the_routine_is_safe(self):
        w = self.w
        w.toggle_arm()
        w.disconnect_arm()
        pump(0.5)
        self.assertIsNone(w.prep)
        self.assertIsNone(w.arm)


class FakeBundle:
    """Looks like a Model_Training bundle: separate preprocessing and classifier."""
    classes = ["rest", "left"]

    class model:
        @staticmethod
        def predict_proba(x):
            return np.array([[0.1, 0.9]])

    @staticmethod
    def transform(window):
        return window


class InferenceTimingTests(unittest.TestCase):
    def test_worker_reports_preprocessing_and_classification_separately(self):
        worker = InferenceWorker(500)
        got = []
        worker.prediction_ready.connect(lambda *a: got.append(a))
        worker.load_model(FakeBundle(), FakeBundle.classes)
        worker.start()
        self.addCleanup(worker.stop)
        arrived = time.perf_counter()
        worker.submit_window(np.zeros((8, 100), np.float32), arrived, 0.05)
        end = time.time() + 3
        while not got and time.time() < end:
            pump(0.01)
        label, confidence, timing = got[0]
        self.assertEqual((label, round(confidence, 2)), ("left", 0.9))
        self.assertTrue(arrived <= timing.start <= timing.prepared <= timing.done)
        self.assertEqual(timing.since_previous, 0.05)


def timing(signal):
    t = Timing(signal, signal + 0.001, signal + 0.004, signal + 0.006, 0.05)
    t.delivered = signal + 0.007
    return t


class PipelineLogTests(unittest.TestCase):
    def test_stages_add_up_to_the_total(self):
        log = PipelineLog(5, {6: "base"})
        for i in range(4):
            log.add_prediction("left", 0.9, "left", timing(10.0 + i * 0.05))
        event = log.gesture_stable("left", 10.160)
        log.commanded("left", [6], 10.170)
        log.packet_sent({6: 520}, 120, 10.180, 10.185)
        self.assertEqual(event.status, "sent")
        self.assertIn("base", event.data)
        t = event.first.timing
        stages = [t.wait_ms, t.inference_ms, t.deliver_ms, event.vote_ms, event.dispatch_ms, event.send_ms]
        self.assertAlmostEqual(sum(stages), event.total_ms, places=6)
        self.assertAlmostEqual(event.total_ms, 185.0, places=6)

    def test_a_packet_written_before_the_command_was_registered_still_counts(self):
        log = PipelineLog()
        log.add_prediction("pinch", 0.9, "pinch", timing(1.0))
        event = log.gesture_stable("pinch", 1.01)
        log.packet_sent({1: 510}, 700, 1.03, 1.031)                 # gripper: written inside the control tick
        log.commanded("pinch", [1], 1.02)
        self.assertEqual(event.status, "sent")

    def test_other_servos_do_not_count_and_disarming_closes_the_row(self):
        log = PipelineLog()
        log.add_prediction("left", 0.9, "left", timing(1.0))
        event = log.gesture_stable("left", 1.01)
        log.packet_sent({4: 500}, 100, 1.03, 1.031)
        self.assertIsNone(event.sent)
        log.close_open("not sent (disarmed)")
        self.assertEqual(event.status, "not sent (disarmed)")

    def test_csv_export(self):
        log = PipelineLog()
        log.add_prediction("left", 0.9, "left", timing(1.0))
        log.gesture_stable("left", 1.01, "not sent (disarmed: preview)")
        log.packet_sent({6: 500}, 100, 2.0, 2.001)
        path = Path(tempfile.mkdtemp()) / "log.csv"
        self.assertEqual(log.export_csv(path), 3)
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual({r["kind"] for r in rows}, {"prediction", "gesture", "xarm_packet"})


class PipelineThroughTheWindowTests(WindowBase):
    def test_a_gesture_is_followed_from_signal_to_the_arm_and_shown(self):
        w = self.w
        cp = w.pages["control"]
        cp.mode.setCurrentIndex(cp.mode.findData("gestures"))
        w.link_state, w.model_loaded = LinkState.STREAMING, True
        w.toggle_arm()
        self.settle()
        w.toggle_armed()
        self.settle()
        self.assertIsInstance(w.arm.backend, SimulatedBackend)
        for _ in range(12):
            now = time.perf_counter()
            w.last_batch_t = time.monotonic()
            w.on_prediction("Left", 0.95, Timing(now - 0.004, now - 0.003, now - 0.002, now - 0.001, 0.05))
            pump(0.05)
        events = [e for e in w.pipeline.snapshot("gestures") if e.gesture == "left"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].status, "sent", events[0])
        self.assertGreater(events[0].total_ms, 0)
        w.go_to("log")
        page = w.pages["log"]
        pump(0.5)
        for key, table in page.tables.items():
            page.tabs.buttons[key].click()
            pump(0.05)
            self.assertGreater(table.rowCount(), 0, key)
        page.clear()
        self.assertEqual(page.tables[page.current].rowCount(), 0)


if __name__ == "__main__":
    unittest.main()
