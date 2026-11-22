"""Live monitor: EMG channels, the recognised gesture, its confidence, and the transition history."""

from __future__ import annotations

import time
from collections import deque

from PyQt5.QtWidgets import QHBoxLayout, QListWidget, QProgressBar, QVBoxLayout, QWidget

from ..widgets import Banner, Card, SignalScope, Stat, label


class MonitorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        g = Card("Gesture recognition", "What the model says right now, and what the filtered (stable) decision is.")
        row = QHBoxLayout()
        row.setSpacing(28)
        self.raw = Stat("model output")
        self.conf = Stat("confidence")
        self.stable = Stat("stable gesture")
        self.latency = Stat("signal → prediction")
        for s in (self.raw, self.conf, self.stable, self.latency):
            row.addWidget(s)
        row.addStretch(1)
        g.add(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFormat("confidence  %v %%")
        g.add(self.bar)
        self.threshold = label("", "caption")
        g.add(self.threshold)
        self.movement = Banner("Arm: no movement", "none")
        g.add(self.movement)
        root.addWidget(g)

        s = Card("Live EMG", "Eight channels. At rest each line should be calm; squeeze and the matching channels swell.")
        self.scope = SignalScope()
        s.add(self.scope)
        root.addWidget(s)

        h = Card("Gesture transitions", "Newest first. One line each time the stable gesture changes.")
        self.history = QListWidget()
        self.history.setMinimumHeight(200)
        self.history.setMaximumHeight(260)
        h.add(self.history)
        root.addWidget(h)
        root.addStretch(1)
        self._last_stable = None
        self._latencies = deque(maxlen=50)

    def showEvent(self, e):
        super().showEvent(e)
        self.scope.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self.scope.stop()

    def restyle(self):
        self.scope.restyle()

    def set_threshold(self, text: str):
        self.threshold.setText(text)

    def on_prediction(self, label_text: str, confidence: float, accepted_gesture, latency_s: float):
        self.raw.set(label_text, "good" if accepted_gesture else "warn")
        self.conf.set(f"{confidence * 100:.0f} %", "good" if accepted_gesture else "warn")
        self.bar.setValue(int(round(confidence * 100)))
        if latency_s >= 0:
            self._latencies.append(latency_s)
            avg = sum(self._latencies) / len(self._latencies)
            self.latency.set(f"{avg * 1000:.0f} ms", "good" if avg < 0.1 else "warn")

    def on_stable(self, gesture_name, action_text: str, confidence: float):
        if gesture_name == self._last_stable:
            return
        self._last_stable = gesture_name
        self.stable.set(gesture_name or "—")
        stamp = time.strftime("%H:%M:%S")
        self.history.insertItem(0, f"{stamp}   {gesture_name or '(none)'}   {action_text}")
        while self.history.count() > 60:
            self.history.takeItem(self.history.count() - 1)

    def set_movement(self, text: str, active: bool):
        self.movement.set(text, "accent" if active else "none")
