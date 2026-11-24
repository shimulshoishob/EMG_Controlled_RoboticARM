"""Arm control - the main screen: connect the arm, enable motion, watch what the gestures do, drive it with
buttons or keys."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QProgressBar, QSpinBox,
                             QVBoxLayout, QWidget)

from ..widgets import Banner, Card, button, label
from .arm_page import ArmPage
from .keyboard_page import DrivePanel


class ControlPage(QWidget):
    arm_toggled = pyqtSignal()
    pad_pressed = pyqtSignal(str)
    pad_released = pyqtSignal(str)
    grip_pressed = pyqtSignal()

    def _toggle_advanced_keys(self, shown: bool):
        self.drive.setVisible(shown)
        self.adv_keys_btn.setText("Hide advanced keyboard controls" if shown else "Show advanced keyboard controls")

    def set_pad_enabled(self, enabled: bool):
        self.drive.set_pad_enabled(enabled)

    def highlight_key(self, motion: str, down: bool):
        self.drive.highlight_key(motion, down)

    def __init__(self, config: dict, gesture_min_conf: float, keymap, simulate_default: bool, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        self.arm_page = ArmPage(simulate_default)
        root.addWidget(self.arm_page)

        c1 = Card("Enable motion", "The arm only moves while ARMED. The red STOP button (and Space or Esc) works at any time.", step=2)
        self.arm_btn = button("ARM - enable motion", "success", slot=self.arm_toggled.emit)
        self.arm_btn.setMinimumHeight(40)
        c1.header_right.addWidget(self.arm_btn)
        self.state_banner = Banner("DISARMED - the arm will not move.", "none")
        c1.add(self.state_banner)
        self.checklist = label("", "muted", wrap=True)
        c1.add(self.checklist)
        self.card_arm = c1
        root.addWidget(c1)

        row = QHBoxLayout()
        row.setSpacing(16)

        live = Card("Live", "What the model sees and what the arm does about it.")
        self.raw_label = label("Model output: -", "mono")
        live.add(self.raw_label)
        self.action_label = label("HOLD", "h1", "accent")
        live.add(self.action_label)
        self.note_label = label("", "muted", wrap=True)
        live.add(self.note_label)
        self.joint_label = label("", "h2")
        live.add(self.joint_label)
        self.grip_hint = label("", "muted", wrap=True)
        live.add(self.grip_hint)
        self.servo_bars = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        for r, (sid, servo) in enumerate(sorted(config["arm"]["servos"].items(), key=lambda kv: int(kv[0]))):
            grid.addWidget(label(f"{servo['name'].replace('_', ' ').title()}", "muted"), r, 0)
            bar = QProgressBar()
            bar.setRange(int(servo["min"]), int(servo["max"]))
            bar.setFormat("%v")
            bar.setValue(int(servo["min"]))
            grid.addWidget(bar, r, 1)
            self.servo_bars[int(sid)] = bar
        grid.setColumnStretch(1, 1)
        live.add(grid)
        self.gripper_label = label("Gripper: -", "muted")
        live.add(self.gripper_label)
        row.addWidget(live, 3, Qt.AlignTop)

        opts = Card("Control options")
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.addWidget(label("Control mode", "muted"), 3, 0)
        self.mode = QComboBox()
        self.mode.addItem("Keys + gestures", "hybrid")
        self.mode.addItem("Gestures only", "gestures")
        self.mode.addItem("Keyboard only", "keyboard")
        self.mode.setToolTip("Keys + gestures: the keyboard overrides gestures. Gestures only: keys are ignored (except STOP). "
                             "Keyboard only: EMG is ignored.")
        self.mode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mode.setMinimumContentsLength(10)
        form.addWidget(self.mode, 3, 1)
        form.addWidget(label("Up / Down moves", "muted"), 0, 0)
        self.vertical = QComboBox()
        for name in config["vertical_joints"]:
            self.vertical.addItem(name.capitalize(), name)
        form.addWidget(self.vertical, 0, 1)
        form.addWidget(label("Speed (units/s)", "muted"), 1, 0)
        self.speed = QSpinBox()
        self.speed.setRange(10, int(config["arm"]["hard_max_speed_units_per_s"]))
        self.speed.setSingleStep(10)
        self.speed.editingFinished.connect(self.speed.clearFocus)
        self.speed.setValue(int(config["arm"]["default_speed_units_per_s"]))
        form.addWidget(self.speed, 1, 1)
        form.addWidget(label("Min. confidence", "muted"), 2, 0)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(40, 99)
        self.conf.setSuffix(" %")
        self.conf.setValue(gesture_min_conf * 100)
        self.conf.editingFinished.connect(self.conf.clearFocus)
        form.addWidget(self.conf, 2, 1)
        opts.add(form)
        opts.add(label("Raise the confidence if the arm moves when you are relaxed; lower it if real gestures are ignored.", "caption", wrap=True))
        row.addWidget(opts, 2, Qt.AlignTop)
        root.addLayout(row)

        self.drive = DrivePanel(keymap)
        self.drive.pad_pressed.connect(self.pad_pressed)
        self.drive.pad_released.connect(self.pad_released)
        self.drive.grip_pressed.connect(self.grip_pressed)
        self.pad_buttons = self.drive.pad_buttons
        self.grip_btn = self.drive.grip_btn
        self.key_label = self.drive.key_label
        # Advanced: the button/key table is tucked away. The keys keep working while it is hidden.
        self.adv_keys_btn = button("Show advanced keyboard controls", "ghost")
        self.adv_keys_btn.setCheckable(True)
        self.adv_keys_btn.toggled.connect(self._toggle_advanced_keys)
        self.drive.setVisible(False)
        root.addWidget(self.adv_keys_btn)
        root.addWidget(self.drive)
        root.addStretch(1)
