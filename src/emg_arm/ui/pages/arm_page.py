"""The robotic-arm card (part of the Arm control page): connect the xArm (it goes home by itself) and open/close the gripper."""

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

        card = Card("Connect the arm", "Use its 7.5 V supply and keep clear: it then goes to rest, or if already there "
                    "works the gripper once.", step=1)
        card.setToolTip("Power the xArm 1S from its own 7.5 V supply; USB is only for data. Close any other program that "
                        "controls the arm. After connecting and when you press ARM, the arm moves slowly to its rest pose. "
                        "If it is already there, the gripper closes and opens once instead, so remove anything it holds.")
        self.card = card
        top = QHBoxLayout()
        top.setSpacing(10)
        self.connect_btn = button("Connect arm", "success", slot=self.connect_toggled.emit)
        self.connect_btn.setMinimumWidth(130)
        top.addWidget(self.connect_btn)
        self.simulate = QCheckBox("Simulate the arm (no hardware)")
        self.simulate.setChecked(simulate_default)
        top.addWidget(self.simulate)
        top.addStretch(1)
        card.add(top)
        self.banner = Banner("Arm not connected.", "none")
        card.add(self.banner)
        row = QHBoxLayout()
        self.home_btn = button("Go home again (slow)", slot=self.home_requested.emit)
        self.grip_btn = button("Open gripper", slot=self._grip_clicked)
        self._gripper_closed = None
        for b in (self.home_btn, self.grip_btn):
            row.addWidget(b, 1)
        card.add(row)
        root.addWidget(card)

    def set_gripper_state(self, closed):
        """The one gripper button always offers the opposite of what the gripper is doing now:
        closed (or unknown) -> "Open gripper", open -> "Close gripper"."""
        self._gripper_closed = closed
        self.grip_btn.setText("Close gripper" if closed is False else "Open gripper")

    def _grip_clicked(self):
        self.gripper_requested.emit(self._gripper_closed is False)       # open -> close; closed/unknown -> open
