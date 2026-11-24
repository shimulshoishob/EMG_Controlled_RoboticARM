"""The robotic-arm connection card (part of the Arm control page): connect the xArm, send it home, test the gripper."""

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

        card = Card("Connect the arm", "Power the xArm 1S from its own 7.5 V supply; USB is only for data. Close any other "
                    "program that controls the arm. Keep everyone out of its reach before sending it home.", step=1)
        self.card = card
        self.connect_btn = button("Connect arm", "success", slot=self.connect_toggled.emit)
        card.header_right.addWidget(self.connect_btn)
        self.simulate = QCheckBox("Simulate the arm (no hardware)")
        self.simulate.setChecked(simulate_default)
        card.add(self.simulate)
        self.banner = Banner("Arm not connected.", "none")
        card.add(self.banner)
        row = QHBoxLayout()
        self.home_btn = button("Go home (slow)", "primary", slot=self.home_requested.emit)
        self.open_btn = button("Open gripper", slot=lambda: self.gripper_requested.emit(False))
        self.close_btn = button("Close gripper", slot=lambda: self.gripper_requested.emit(True))
        for b in (self.home_btn, self.open_btn, self.close_btn):
            row.addWidget(b)
        row.addStretch(1)
        card.add(row)
        card.add(label("Nothing can be controlled until the arm is home.", "caption"))
        root.addWidget(card)
