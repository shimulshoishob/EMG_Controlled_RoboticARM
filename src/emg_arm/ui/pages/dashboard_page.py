"""Dashboard: the whole system state at a glance (read-only; fixes happen on the pages it links to)."""

from __future__ import annotations

from typing import Dict

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from .. import theme
from ..widgets import Card, button, label


class Tile(QFrame):
    """One status: caption, big value, one line of detail. The border colour carries the severity."""

    def __init__(self, caption: str):
        super().__init__()
        theme.set_role(self, "card", "none")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(2)
        lay.addWidget(label(caption.upper(), "caption", wrap=True))
        self.value = label("—", "h2", wrap=True)
        self.detail = label("", "caption", wrap=True)
        lay.addWidget(self.value)
        lay.addWidget(self.detail)
        lay.addStretch(1)
        self.setMinimumHeight(100)

    def set(self, value: str, detail: str = "", tone: str = "none"):
        self.value.setText(value)
        self.detail.setText(detail)
        theme.set_role(self, tone=tone)
        theme.set_role(self.value, tone="none" if tone == "none" else tone)


TILES = [("armband", "Armband / ESP32-S3"), ("data", "Incoming data"), ("model", "Gesture model"),
         ("gesture", "Detected gesture"), ("arm", "Robotic arm"), ("mode", "Control mode"),
         ("movement", "Current movement"), ("health", "Errors & warnings")]


class DashboardPage(QWidget):
    navigate = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        grid = QGridLayout()
        grid.setSpacing(14)
        self.tiles: Dict[str, Tile] = {}
        for i, (key, caption) in enumerate(TILES):
            t = Tile(caption)
            self.tiles[key] = t
            grid.addWidget(t, i // 4, i % 4)
        for c in range(4):
            grid.setColumnStretch(c, 1)
        root.addLayout(grid)

        quick = Card("Quick actions")
        row = QHBoxLayout()
        for text, page in (("Connection", "armband"), ("Robotic arm", "arm"), ("Arm control", "control"),
                           ("Gesture mapping", "mapping"), ("Diagnostics", "diagnostics")):
            row.addWidget(button(text, slot=lambda _=False, p=page: self.navigate.emit(p)))
        row.addStretch(1)
        quick.add(row)
        root.addWidget(quick)

        self.events_card = Card("Recent events", "Warnings and errors from every part of the system. Full history: Diagnostics.")
        self.events = label("No events yet.", "mono", wrap=True)
        self.events_card.add(self.events)
        root.addWidget(self.events_card)
        root.addStretch(1)

    def set_events(self, lines):
        self.events.setText("\n".join(lines) if lines else "No warnings or errors. ✓")
