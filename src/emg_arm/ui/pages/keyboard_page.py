"""Keyboard control: the key reference, a key tester, and re-assigning keys."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QPushButton, QVBoxLayout, QWidget

from ..keyboard import MODIFIER_KEYS, KeyMap, pretty_key
from ..widgets import Banner, Card, button, label


class KeyCapture(QPushButton):
    """Click, then press the new key. Esc cancels."""
    captured = pyqtSignal(int)
    started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("—", parent)
        self.setMinimumWidth(84)
        self.setCursor(Qt.PointingHandCursor)
        self.key = 0
        self._waiting = False
        self.clicked.connect(self._begin)

    def show_key(self, key: int):
        self.key = key
        self.setText(pretty_key(key) if key else "—")

    def _begin(self):
        self._waiting = True
        self.setText("Press a key…")
        self.started.emit()
        self.grabKeyboard()

    def _end(self):
        self._waiting = False
        self.releaseKeyboard()
        self.setText(pretty_key(self.key) if self.key else "—")
        self.finished.emit()

    def keyPressEvent(self, event):
        if not self._waiting:
            return super().keyPressEvent(event)
        key = event.key()
        if key in MODIFIER_KEYS:
            return
        if key == Qt.Key_Escape:
            self._end()
            return
        if key == Qt.Key_Backspace or key == Qt.Key_Delete:
            self.key = 0
            self._end()
            self.captured.emit(0)
            return
        self._end()
        self.captured.emit(key)

    def focusOutEvent(self, event):
        if self._waiting:
            self._end()
        super().focusOutEvent(event)


class KeyboardPage(QWidget):
    def __init__(self, keymap: KeyMap, parent=None):
        super().__init__(parent)
        self.keymap = keymap
        self.armed = False
        self.capture_started = lambda: None
        self.capture_finished = lambda: None
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        tester = Card("Key tester", "Press any key while the arm is disarmed to see how the application understands it. "
                      "Use it to check your keyboard (Mac built-in, external, other layouts) before arming.")
        self.last_key = Banner("Press a key…", "none")
        tester.add(self.last_key)
        self.held = label("", "caption")
        tester.add(self.held)
        root.addWidget(tester)

        self.card = Card("Key assignments", "Click a key button, then press the new key. Backspace clears it. "
                         "Space and Esc are always STOP and cannot be changed.")
        self.reset_btn = button("Reset to defaults", "ghost", slot=self._reset)
        self.card.header_right.addWidget(self.reset_btn)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(6)
        self.buttons = {}
        group = None
        row = 0
        for action in keymap.actions:
            if action.group != group:
                group = action.group
                self.grid.addWidget(label(group.upper(), "caption"), row, 0, 1, 3)
                row += 1
            self.grid.addWidget(label(action.label), row, 0)
            pair = []
            for slot in range(KeyMap.MAX_KEYS_PER_ACTION):
                b = KeyCapture()
                b.started.connect(self._capture_started)
                b.finished.connect(self._capture_finished)
                b.captured.connect(lambda key, a=action.id, s=slot: self._captured(a, s, key))
                self.grid.addWidget(b, row, 1 + slot)
                pair.append(b)
            self.buttons[action.id] = pair
            row += 1
        self.grid.setColumnStretch(0, 1)
        self.card.add(self.grid)
        self.status = Banner("", "none")
        self.card.add(self.status)
        root.addWidget(self.card)

        stop = Card("Always available")
        stop.add(label("Space  or  Esc      EMERGENCY STOP (works on every page)\n"
                       "Combinations with Ctrl / Cmd / Alt are ignored so system shortcuts keep working.\n"
                       "Motion keys only move the arm while it is ARMED and no text box has focus; "
                       "they continue as long as you hold them and stop when you release.", "mono", wrap=True))
        root.addWidget(stop)
        root.addStretch(1)
        self.refresh()
        if keymap.load_warning:
            self.status.set(keymap.load_warning, "warn")

    # ---- outside
    def attach(self, controller):
        """Connect the live key tester to the controller."""
        controller.key_seen.connect(lambda text: self.last_key.set(text, "none" if "not assigned" in text else "good"))
        controller.held_changed.connect(lambda keys: self.held.setText("Held: " + "  ".join(keys) if keys else ""))
        self.capture_started = lambda: setattr(controller, "capturing", True)
        self.capture_finished = lambda: setattr(controller, "capturing", False)

    def set_armed(self, armed: bool):
        self.armed = armed
        for pair in self.buttons.values():
            for b in pair:
                b.setEnabled(not armed)
        self.reset_btn.setEnabled(not armed)
        if armed:
            self.status.set("Disarm to change key assignments.", "warn")
        elif self.status.text.text().startswith("Disarm"):
            self.status.set("", "none")

    def refresh(self):
        for aid, pair in self.buttons.items():
            keys = self.keymap.keys_for(aid)
            for i, b in enumerate(pair):
                b.show_key(keys[i] if i < len(keys) else 0)

    def reference_text(self) -> str:
        km = self.keymap
        return "\n".join(f"{a.label:<34} {km.label_for(a.id)}" for a in km.actions) + "\nSTOP                               Space / Esc"

    # ---- internals
    def _capture_started(self):
        self.capture_started()

    def _capture_finished(self):
        self.capture_finished()

    def _captured(self, action_id: str, slot: int, key: int):
        keys = self.keymap.keys_for(action_id)
        while len(keys) < KeyMap.MAX_KEYS_PER_ACTION:
            keys.append(0)
        keys[slot] = key
        keys = [k for k in keys if k]
        try:
            self.keymap.assign(action_id, keys)
            self.status.set("Saved.", "good")
        except ValueError as exc:
            self.status.set(f"Not changed: {exc}", "bad")
        self.refresh()

    def _reset(self):
        self.keymap.reset_defaults()
        self.refresh()
        self.status.set("Defaults restored.", "good")
