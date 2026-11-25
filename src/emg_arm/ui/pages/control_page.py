"""Arm control - the main screen: connect the arm, enable motion, watch what the gestures do, drive it with
buttons or keys."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QProgressBar, QSpinBox, QVBoxLayout,
                             QWidget)

from .. import theme
from ..widgets import Banner, Card, button, label, section, two_columns
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
        root.setSpacing(14)
        columns, left, right = two_columns(1, 1)
        root.addLayout(columns, 1)

        # ---- left: connect, arm, options
        self.arm_page = ArmPage(simulate_default)
        left.addWidget(self.arm_page)

        c1 = Card("Enable motion", "ARM first runs the rest / gripper check, then the arm follows you.", step=2)
        self.arm_btn = button("ARM - enable motion", slot=self.arm_toggled.emit)
        self.arm_btn.setProperty("big", True)
        theme.set_role(self.arm_btn, "success")
        self.arm_btn.setMinimumHeight(42)
        c1.add(self.arm_btn)
        self.state_banner = Banner("DISARMED - the arm will not move.", "none")
        c1.add(self.state_banner)
        self.checklist = label("", "caption", wrap=True)
        c1.add(self.checklist)
        self.card_arm = c1
        left.addWidget(c1)

        opts = Card("Control options")
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(3)

        def field(text, widget, row, column):
            form.addWidget(label(text, "caption"), row * 2, column)
            form.addWidget(widget, row * 2 + 1, column)

        self.mode = QComboBox()
        self.mode.addItem("Keys + gestures", "hybrid")
        self.mode.addItem("Gestures only", "gestures")
        self.mode.addItem("Keyboard only", "keyboard")
        self.mode.setToolTip("Keys + gestures: the keyboard overrides gestures. Gestures only: keys are ignored (except STOP). "
                             "Keyboard only: EMG is ignored.")
        self.mode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mode.setMinimumContentsLength(10)
        field("Control mode", self.mode, 0, 0)
        self.vertical = QComboBox()
        for name in config["vertical_joints"]:
            self.vertical.addItem(name.capitalize(), name)
        field("Up / Down moves", self.vertical, 0, 1)
        self.speed = QSpinBox()
        self.speed.setRange(10, int(config["arm"]["hard_max_speed_units_per_s"]))
        self.speed.setSingleStep(10)
        self.speed.editingFinished.connect(self.speed.clearFocus)
        self.speed.setValue(int(config["arm"]["default_speed_units_per_s"]))
        field("Speed (units/s)", self.speed, 1, 0)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(40, 99)
        self.conf.setSuffix(" %")
        self.conf.setValue(gesture_min_conf * 100)
        self.conf.editingFinished.connect(self.conf.clearFocus)
        self.conf.setToolTip("Raise it if the arm moves when you are relaxed; lower it if real gestures are ignored.")
        field("Min. confidence", self.conf, 1, 1)
        form.setColumnStretch(0, 1)
        form.setColumnStretch(1, 1)
        opts.add(form)
        opts.setToolTip(self.conf.toolTip())

        # ---- right: what is happening now
        live = Card("Live", fill=True)
        live.setToolTip("What the model sees and what the arm does about it.")
        live.body.setSpacing(6)
        hero = QFrame()
        theme.set_role(hero, "hero")
        hl = QVBoxLayout(hero)
        hl.setContentsMargins(14, 10, 14, 10)
        hl.setSpacing(2)
        hl.addWidget(section("Arm action"))
        self.action_label = label("HOLD", "display", "accent", wrap=True)
        hl.addWidget(self.action_label)
        self.note_label = label("", "muted", wrap=True)
        hl.addWidget(self.note_label)
        live.add(hero)
        self.raw_label = label("Model output: -", "mono", wrap=True)
        live.add(self.raw_label)
        self.joint_label = label("", "h2", wrap=True)
        live.add(self.joint_label)
        self.grip_hint = label("", "muted", wrap=True)
        live.add(self.grip_hint)
        live.add(section("Servo positions"))
        self.servo_bars = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
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
        live.body.addStretch(1)
        right.addWidget(live, 1)
        right.addWidget(opts)

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
        left.addWidget(self.adv_keys_btn)
        left.addStretch(1)
        root.addWidget(self.drive)
