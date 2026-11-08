"""Step 5 - Data & settings: record raw signals, find the logs, choose the appearance."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QVBoxLayout, QWidget

from ... import __version__
from ...paths import LOG_DIR, RECORDINGS_DIR
from ..widgets import Banner, Card, button, label


def open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class DataPage(QWidget):
    record_toggled = pyqtSignal(str)          # label ("" if none); emitted on start and stop
    theme_changed = pyqtSignal(str)

    def __init__(self, theme_mode: str, parent=None):
        super().__init__(parent)
        self.recording = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        c1 = Card("Record raw signals", "Saves every EMG sample (8 channels, 500 Hz) and the IMU angles to a CSV "
                  "file, ready for your analysis. The armband must be connected.")
        row = QHBoxLayout()
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("optional label, e.g. S02_fist_trial3")
        row.addWidget(self.label_edit, 1)
        self.record_btn = button("Start recording", "primary", slot=self._toggle)
        row.addWidget(self.record_btn)
        c1.add(row)
        self.record_status = Banner("Not recording.", "none")
        c1.add(self.record_status)
        c1.add(button("Open recordings folder", slot=lambda: open_folder(RECORDINGS_DIR)))
        root.addWidget(c1)

        c2 = Card("Session logs", "Each time the arm is connected, a log of predictions, gestures, and servo targets "
                  "is written (one row per event).")
        self.log_label = label("No session yet.", "mono", wrap=True)
        c2.add(self.log_label)
        c2.add(button("Open logs folder", slot=lambda: open_folder(LOG_DIR)))
        root.addWidget(c2)

        c3 = Card("Appearance")
        self.theme_combo = QComboBox()
        for key, text in (("auto", "Match my system"), ("light", "Light"), ("dark", "Dark"), ("pink", "Pink")):
            self.theme_combo.addItem(text, key)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(theme_mode)))
        self.theme_combo.currentIndexChanged.connect(lambda _i: self.theme_changed.emit(self.theme_combo.currentData()))
        c3.add(self.theme_combo)
        root.addWidget(c3)
        root.addWidget(label(f"EMG Robotic Arm {__version__}  ·  armband firmware emg-imu-2.x  ·  protocol BWIM v2", "caption"))
        root.addStretch(1)

    def _toggle(self):
        self.record_toggled.emit("" if self.recording else self.label_edit.text().strip().replace(" ", "_"))

    def set_recording(self, recording: bool, path: str = "", frames: int = 0):
        self.recording = recording
        self.record_btn.setText("Stop recording" if recording else "Start recording")
        from .. import theme
        theme.set_role(self.record_btn, "danger" if recording else "primary")
        self.label_edit.setEnabled(not recording)
        if recording:
            self.record_status.set(f"Recording → {path}   ({frames:,} samples)", "bad")
        elif path:
            self.record_status.set(f"Saved {frames:,} samples to {path}", "good")
        else:
            self.record_status.set("Not recording.", "none")

    def set_log(self, path: str):
        self.log_label.setText(path or "No session yet.")
