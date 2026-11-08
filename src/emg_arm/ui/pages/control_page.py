"""Step 4 - Control: arm the robot and watch what the gestures do."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QProgressBar, QSpinBox,
                             QVBoxLayout, QWidget)

from .. import theme
from ..theme import MOD
from ..widgets import Banner, Card, Chip, button, label


class ControlPage(QWidget):
    arm_toggled = pyqtSignal()
    pad_pressed = pyqtSignal(str)
    pad_released = pyqtSignal(str)
    grip_pressed = pyqtSignal()

    def set_pad_enabled(self, enabled: bool):
        for b in list(self.pad_buttons.values()) + [self.grip_btn]:
            b.setEnabled(enabled)

    def highlight_key(self, motion: str, down: bool):
        b = self.pad_buttons.get(motion)
        if b is not None:
            t = theme.tokens()
            b.setStyleSheet(f"QPushButton {{ background: {t.accent}; color: {t.accent_text}; }}" if down else "")

    def __init__(self, config: dict, gesture_min_conf: float, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        c1 = Card("Enable motion", "The arm only moves while ARMED. The red STOP button (and Space or Esc) works at any time.", step=1)
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

        opts = Card("Settings")
        self.emg_check = QCheckBox("Use EMG gestures (off = keyboard only)")
        self.emg_check.setChecked(True)
        opts.add(self.emg_check)
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.addWidget(label("Up / Down moves", "muted"), 0, 0)
        self.vertical = QComboBox()
        for name in config["vertical_joints"]:
            self.vertical.addItem(name.capitalize(), name)
        form.addWidget(self.vertical, 0, 1)
        form.addWidget(label("Speed (units/s)", "muted"), 1, 0)
        self.speed = QSpinBox()
        self.speed.setRange(10, int(config["arm"]["hard_max_speed_units_per_s"]))
        self.speed.setSingleStep(10)
        self.speed.setValue(int(config["arm"]["default_speed_units_per_s"]))
        form.addWidget(self.speed, 1, 1)
        form.addWidget(label("Min. confidence", "muted"), 2, 0)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(40, 99)
        self.conf.setSuffix(" %")
        self.conf.setValue(gesture_min_conf * 100)
        form.addWidget(self.conf, 2, 1)
        opts.add(form)
        opts.add(label("Raise the confidence if the arm moves when you are relaxed; lower it if real gestures are ignored.", "caption", wrap=True))
        row.addWidget(opts, 2, Qt.AlignTop)
        root.addLayout(row)

        pad = Card("Drive with the keyboard or this pad",
                   "Hold a key (or press and hold a pad button) to move; hold several to combine joints. Works while ARMED, "
                   "with or without the armband.")
        self.pad_buttons = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        rows = [("Base", [("left", "◀  Left  (←)"), ("right", "Right  (→)  ▶")]),
                ("Selected joint", [("up", "▲  Up  (↑)"), ("down", "▼  Down  (↓)")]),
                ("Shoulder", [("shoulder_up", "▲  W"), ("shoulder_down", "▼  S")]),
                ("Elbow", [("elbow_up", "▲  E"), ("elbow_down", "▼  D")])]
        for r, (name, items) in enumerate(rows):
            grid.addWidget(label(name, "muted"), r, 0)
            for c, (motion, text) in enumerate(items):
                b = button(text)
                b.setMinimumHeight(38)
                b.pressed.connect(lambda m=motion: self.pad_pressed.emit(m))
                b.released.connect(lambda m=motion: self.pad_released.emit(m))
                grid.addWidget(b, r, c + 1)
                self.pad_buttons[motion] = b
        self.grip_btn = button("Toggle gripper  (G)", slot=self.grip_pressed.emit)
        self.grip_btn.setMinimumHeight(38)
        grid.addWidget(self.grip_btn, len(rows), 1, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        pad.add(grid)
        root.addWidget(pad)

        keys = Card("Keyboard (works while ARMED, keys can be combined)")
        keys.add(label(
            "←  →   rotate base          ↑  ↓   selected joint (1 = shoulder, 2 = elbow)\n"
            "W  S   shoulder up / down   E  D   elbow up / down          G   toggle gripper\n"
            "Space or Esc   EMERGENCY STOP\n\n"
            "Gestures: fist ≈ 0.6 s then relax = gripper · keep the fist ≈ 1.8 s = switch joint · Rest = hold still", "mono", wrap=True))
        self.key_label = label("Keyboard: idle", "caption")
        keys.add(self.key_label)
        root.addWidget(keys)
        root.addStretch(1)
