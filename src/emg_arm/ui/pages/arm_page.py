"""Step 3 - Robotic arm: connect the xArm (or simulate it), then send it home."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from ..widgets import Banner, Card, button, label


class ArmPage(QWidget):
    connect_toggled = pyqtSignal()
    home_requested = pyqtSignal()
    gripper_requested = pyqtSignal(bool)

    def __init__(self, simulate_default: bool, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        c1 = Card("Connect the arm", "Power the xArm 1S from its own 7.5 V supply; USB is only for data. "
                  "Close any other program that controls the arm (for example the vegetable-sorting script).", step=1)
        self.connect_btn = button("Connect arm", "success", slot=self.connect_toggled.emit)
        c1.header_right.addWidget(self.connect_btn)
        self.simulate = QCheckBox("Simulate the arm (no hardware) - recommended for first tries")
        self.simulate.setChecked(simulate_default)
        c1.add(self.simulate)
        self.banner = Banner("Arm not connected.", "none")
        c1.add(self.banner)
        self.card_connect = c1
        root.addWidget(c1)

        c2 = Card("Send it home", "The arm moves slowly to its start pose. Nothing can be controlled until this is done.", step=2)
        self.home_btn = button("Go home (slow)", "primary", slot=self.home_requested.emit)
        c2.header_right.addWidget(self.home_btn)
        c2.add(label("Make sure nothing and nobody is within about 40 cm of the arm. The person wearing the armband "
                     "sits outside its reach.", "muted", wrap=True))
        self.card_home = c2
        root.addWidget(c2)

        c3 = Card("Test the gripper", "Optional, once the arm is home.")
        row = QHBoxLayout()
        self.open_btn = button("Open gripper", slot=lambda: self.gripper_requested.emit(False))
        self.close_btn = button("Close gripper", slot=lambda: self.gripper_requested.emit(True))
        row.addWidget(self.open_btn)
        row.addWidget(self.close_btn)
        row.addStretch(1)
        c3.add(row)
        root.addWidget(c3)
        root.addStretch(1)
