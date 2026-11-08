"""Small reusable widgets. Colours come from the stylesheet (role/tone), never from the widget."""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

from . import theme


def label(text: str = "", role: str = None, tone: str = None, wrap: bool = False, selectable: bool = False) -> QLabel:
    w = QLabel(text)
    theme.set_role(w, role, tone)
    w.setWordWrap(wrap)
    if selectable:
        w.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return w


def button(text: str, role: str = None, tooltip: str = "", slot=None) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    theme.set_role(b, role)
    if tooltip:
        b.setToolTip(tooltip)
    if slot:
        b.clicked.connect(slot)
    return b


class Dot(QLabel):
    def __init__(self, tone: str = None):
        super().__init__("")
        self.setFixedSize(10, 10)
        theme.set_role(self, "dot", tone or "none")

    def set_tone(self, tone: str):
        theme.set_role(self, tone=tone)


class Chip(QLabel):
    def __init__(self, text: str = "", tone: str = "none"):
        super().__init__(text)
        theme.set_role(self, "chip", tone)
        self.setAlignment(Qt.AlignCenter)

    def set(self, text: str, tone: str):
        self.setText(text)
        theme.set_role(self, tone=tone)


class Card(QFrame):
    """A rounded panel with an optional numbered badge, title and one-line explanation."""

    def __init__(self, title: str = "", subtitle: str = "", step: Optional[int] = None, parent=None):
        super().__init__(parent)
        theme.set_role(self, "card", "none")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(12)
        self.badge: Optional[QLabel] = None
        self._num = str(step) if step is not None else ""
        if title:
            head = QHBoxLayout()
            head.setSpacing(12)
            if step is not None:
                self.badge = QLabel(str(step))
                self.badge.setAlignment(Qt.AlignCenter)
                theme.set_role(self.badge, "badge", "none")
                head.addWidget(self.badge, 0, Qt.AlignTop)
            texts = QVBoxLayout()
            texts.setSpacing(2)
            self.title_label = label(title, "h2")
            texts.addWidget(self.title_label)
            if subtitle:
                self.subtitle_label = label(subtitle, "muted", wrap=True)
                texts.addWidget(self.subtitle_label)
            head.addLayout(texts, 1)
            self.header_right = QHBoxLayout()
            head.addLayout(self.header_right)
            outer.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        outer.addLayout(self.body)

    def add(self, widget_or_layout):
        if isinstance(widget_or_layout, QWidget):
            self.body.addWidget(widget_or_layout)
        else:
            self.body.addLayout(widget_or_layout)

    def set_done(self, done: bool):
        """Turns the step badge into a green tick (and back to its number)."""
        if self.badge is not None:
            self.badge.setText("✓" if done else self._num)
            theme.set_role(self.badge, tone="good" if done else "none")

    def set_tone(self, tone: str):
        theme.set_role(self, tone=tone)


class Banner(QFrame):
    """A message strip: tone = good | warn | bad | accent | none."""

    def __init__(self, text: str = "", tone: str = "none", parent=None):
        super().__init__(parent)
        theme.set_role(self, "banner", tone)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)
        self.dot = Dot(tone)
        lay.addWidget(self.dot, 0, Qt.AlignTop)
        self.text = label(text, wrap=True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.text, 1)
        self.actions = QHBoxLayout()
        lay.addLayout(self.actions)

    def set(self, text: str, tone: str):
        self.text.setText(text)
        theme.set_role(self, tone=tone)
        self.dot.set_tone(tone)


class Segmented(QWidget):
    """Joined buttons where exactly one is selected."""
    changed = pyqtSignal(str)

    def __init__(self, options: List[tuple], parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for i, (key, text) in enumerate(options):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            theme.set_role(b, "seg")
            b.setProperty("pos", "first" if i == 0 else "last" if i == len(options) - 1 else "mid")
            self.group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
        lay.addStretch(1)

    def set_current(self, key: str):
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def current(self) -> str:
        return next((k for k, b in self.buttons.items() if b.isChecked()), "")


class Stat(QWidget):
    """Big number with a caption."""

    def __init__(self, caption: str, value: str = "-"):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.value = label(value, "stat")
        lay.addWidget(self.value)
        lay.addWidget(label(caption, "caption"))

    def set(self, text: str, tone: str = "none"):
        self.value.setText(text)
        theme.set_role(self.value, tone=tone)


class SignalScope(QWidget):
    """Rolling 8-channel EMG plot (each channel centred and offset) so the signal can be checked by eye."""

    def __init__(self, seconds: float = 4.0, rate: int = 500, channels: int = 8, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg
        self._pg = pg
        self.channels = channels
        self.length = int(seconds * rate)
        self.buffer = np.zeros((channels, self.length), dtype=np.float32)
        self.filled = 0
        self.span = 400.0
        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()
        self.plot.getAxis("left").setStyle(showValues=False, tickLength=0)
        self.plot.getAxis("bottom").setStyle(showValues=False, tickLength=0)
        self.plot.setXRange(0, self.length, padding=0)
        self.plot.setYRange(-self.span, self.span * (2 * channels - 1), padding=0)
        self.curves = []
        self.x = np.arange(self.length)
        for ch in range(channels):
            pen = pg.mkPen(theme.CHANNEL_COLORS[ch % 8], width=1.2)
            self.curves.append(self.plot.plot(self.x, np.zeros(self.length), pen=pen))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        self.setMinimumHeight(220)
        self.restyle()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.redraw)
        self.dirty = False
        self.ema = np.zeros(channels, dtype=np.float32)

    def restyle(self):
        t = theme.tokens()
        self.plot.setBackground(t.plot_bg)
        for axis in ("left", "bottom"):
            self.plot.getAxis(axis).setPen(self._pg.mkPen(t.plot_grid))

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()

    def clear(self):
        self.buffer[:] = 0
        self.filled = 0
        self.dirty = True

    def push(self, frames: np.ndarray):
        """frames: (n, >=channels) raw counts. The slow baseline is removed so channels line up."""
        n = len(frames)
        if n == 0:
            return
        data = frames[:, :self.channels].T
        if self.filled == 0:
            self.ema = data.mean(axis=1)
        self.ema = 0.995 * self.ema + 0.005 * data.mean(axis=1)
        data = data - self.ema[:, None]
        n = min(n, self.length)
        self.buffer[:, :-n] = self.buffer[:, n:]
        self.buffer[:, -n:] = data[:, -n:]
        self.filled = min(self.length, self.filled + n)
        self.dirty = True

    def redraw(self):
        if not self.dirty or not self.isVisible():
            return
        self.dirty = False
        for ch, curve in enumerate(self.curves):
            curve.setData(self.x, np.clip(self.buffer[ch], -self.span, self.span) + ch * 2 * self.span)


class Divider(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {theme.tokens().border};")
