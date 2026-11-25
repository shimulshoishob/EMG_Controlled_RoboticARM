"""Small reusable widgets. Colours come from the stylesheet (role/tone), never from the widget."""

from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QAbstractSpinBox, QButtonGroup, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from . import theme


def label(text: str = "", role: str = None, tone: str = None, wrap: bool = False, selectable: bool = False) -> QLabel:
    w = QLabel(text)
    theme.set_role(w, role, tone)
    w.setWordWrap(wrap)
    if selectable:
        w.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return w


def section(text: str) -> QLabel:
    """Small spaced-out capital heading that groups related controls."""
    w = label(text.upper(), "section")
    font = w.font()
    font.setLetterSpacing(QFont.AbsoluteSpacing, 1.4)
    w.setFont(font)
    return w


def two_columns(left: int = 1, right: int = 1):
    """A row with two top-to-bottom columns: returns (row, left column, right column)."""
    row = QHBoxLayout()
    row.setSpacing(14)
    columns = []
    for stretch in (left, right):
        col = QVBoxLayout()
        col.setSpacing(14)
        row.addLayout(col, stretch)
        columns.append(col)
    return row, columns[0], columns[1]


def button(text: str, role: str = None, tooltip: str = "", slot=None) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    theme.set_role(b, role)
    if tooltip:
        b.setToolTip(tooltip)
    if slot:
        b.clicked.connect(slot)
    return b


# Values are typed (or picked from the list), never changed by the mouse wheel: scrolling a page must not
# silently change the arm's speed or a setting that happens to be under the pointer.
class _NoWheel:
    def wheelEvent(self, event):
        event.ignore()                              # the page behind still scrolls


class _NumberBox(_NoWheel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setFocusPolicy(Qt.StrongFocus)         # the wheel does not give focus either


class SpinBox(_NumberBox, QSpinBox):
    """A whole-number field without up/down arrows."""


class DoubleSpinBox(_NumberBox, QDoubleSpinBox):
    """A decimal-number field without up/down arrows."""


class ComboBox(_NoWheel, QComboBox):
    """A drop-down that only changes when you pick from it."""

    def __init__(self, *args):
        super().__init__(*args)
        self.setFocusPolicy(Qt.StrongFocus)


class Dot(QLabel):
    def __init__(self, tone: str = None):
        super().__init__("")
        self.setFixedSize(8, 8)
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
    """A rounded panel with an optional numbered badge, title and one-line explanation.

    ``fill=True`` lets the card grow to use the free height of the page (for lists, tables and plots)."""

    def __init__(self, title: str = "", subtitle: str = "", step: Optional[int] = None, parent=None,
                 fill: bool = False):
        super().__init__(parent)
        theme.set_role(self, "card", "none")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding if fill else QSizePolicy.Maximum)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)
        self.badge: Optional[QLabel] = None
        self._num = str(step) if step is not None else ""
        if title:
            head = QHBoxLayout()
            head.setSpacing(10)
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
        self.body.setSpacing(8)
        outer.addLayout(self.body, 1 if fill else 0)

    def add(self, widget_or_layout, stretch: int = 0):
        if isinstance(widget_or_layout, QWidget):
            self.body.addWidget(widget_or_layout, stretch)
        else:
            self.body.addLayout(widget_or_layout, stretch)

    def set_done(self, done: bool):
        """Turns the step badge into a green tick (and back to its number)."""
        if self.badge is not None:
            self.badge.setText("✓" if done else self._num)
            theme.set_role(self.badge, tone="good" if done else "none")

    def set_step(self, step: int):
        """Change the number shown on the badge (a finished step keeps its tick)."""
        self._num = str(step)
        if self.badge is not None and self.badge.text() != "✓":
            self.badge.setText(self._num)

    def set_tone(self, tone: str):
        theme.set_role(self, tone=tone)


class Banner(QFrame):
    """A message strip: tone = good | warn | bad | accent | none."""

    def __init__(self, text: str = "", tone: str = "none", parent=None, hide_empty: bool = False):
        super().__init__(parent)
        self.hide_empty = hide_empty                      # no empty box while there is nothing to say
        theme.set_role(self, "banner", tone)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(9)
        self.dot = Dot(tone)
        mark = QVBoxLayout()
        mark.setContentsMargins(0, 5, 0, 0)               # level with the first line of text
        mark.addWidget(self.dot)
        mark.addStretch(1)
        lay.addLayout(mark)
        self.text = label(text, wrap=True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.text, 1)
        self.actions = QHBoxLayout()
        lay.addLayout(self.actions)
        if hide_empty and not text:
            self.hide()

    def set(self, text: str, tone: str):
        self.text.setText(text)
        theme.set_role(self, tone=tone)
        self.dot.set_tone(tone)
        if self.hide_empty:
            self.setVisible(bool(text))


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


class Stat(QFrame):
    """Big number with a caption, in its own tile."""

    def __init__(self, caption: str, value: str = "-"):
        super().__init__()
        theme.set_role(self, "tile")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        lay.addWidget(label(caption, "caption"))
        self.value = label(value, "stat")
        self.value.setMinimumWidth(10)
        lay.addWidget(self.value)

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
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)   # takes the free height, asks for none
        self.setMinimumHeight(200)
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
