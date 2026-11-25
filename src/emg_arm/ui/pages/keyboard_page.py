"""Drive panel: every arm movement in one table - its on-screen button and its keyboard shortcuts side by side."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QPushButton

from .. import theme
from ..keyboard import MODIFIER_KEYS, KeyMap, pretty_key
from ..widgets import Banner, Card, button, label, section, two_columns

# on-screen button (motion name used by the control loop) -> keyboard action id
PAD_ACTIONS = {"left": "base_left", "right": "base_right", "up": "joint_up", "down": "joint_down",
               "shoulder_up": "shoulder_up", "shoulder_down": "shoulder_down",
               "elbow_up": "elbow_up", "elbow_down": "elbow_down"}
_PAD_TEXT = {"left": "◀  Hold", "right": "Hold  ▶", "up": "▲  Hold", "down": "▼  Hold"}


def _pad_text(motion: str) -> str:
    return _PAD_TEXT.get(motion) or ("▲  Hold" if motion.endswith("_up") else "▼  Hold")


class KeyCapture(QPushButton):
    """Click, then press the new key. Esc cancels."""
    captured = pyqtSignal(int)
    started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("—", parent)
        self.setMinimumWidth(62)
        self.setProperty("compact", True)
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


class DrivePanel(Card):
    """One row per movement: [on-screen button] [key] [alternative key]. Works while the arm is ARMED."""
    pad_pressed = pyqtSignal(str)
    pad_released = pyqtSignal(str)
    grip_pressed = pyqtSignal()

    def __init__(self, keymap: KeyMap, parent=None):
        super().__init__("Move the arm: buttons and keys",
                         "Hold a button or its key to move; release to stop; hold several to combine joints. "
                         "To change a key, click it and press the new one (Backspace clears it). "
                         "Space and Esc are always STOP.", parent=parent)
        self.keymap = keymap
        self.capture_started = lambda: None
        self.capture_finished = lambda: None
        self.reset_btn = button("Reset keys", "ghost", slot=self._reset)
        self.header_right.addWidget(self.reset_btn)

        self.pad_buttons = {}
        self.grip_btn = None
        motion_for = {action: motion for motion, action in PAD_ACTIONS.items()}
        groups = {}
        for action in keymap.actions:                   # keep first-seen group order, actions together
            groups.setdefault(action.group, []).append(action)
        columns, left, right = two_columns(1, 1)
        columns.setSpacing(18)
        grids, rows = [], []
        for column in (left, right):
            grid = QGridLayout()
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(6)
            for c, text in enumerate(("Movement", "Button", "Key", "Alt. key")):
                grid.addWidget(label(text, "caption"), 0, c)
            grid.setColumnStretch(0, 1)
            column.addLayout(grid)
            column.addStretch(1)
            grids.append(grid)
            rows.append(1)
        total = len(keymap.actions) + len(groups)
        self.buttons = {}
        side = 0
        for group, actions in groups.items():
            if side == 0 and rows[0] > 1 and rows[0] + len(actions) > (total + 1) // 2 + 1:
                side = 1                                # the first column is full: continue in the second
            grid, row = grids[side], rows[side]
            grid.addWidget(section(group), row, 0, 1, 4)
            row += 1
            for action in actions:
                grid.addWidget(label(action.label, wrap=True), row, 0)
                if action.id in motion_for:
                    motion = motion_for[action.id]
                    b = button(_pad_text(motion))
                    b.setProperty("compact", True)
                    b.setMinimumSize(70, 34)
                    b.pressed.connect(lambda m=motion: self.pad_pressed.emit(m))
                    b.released.connect(lambda m=motion: self.pad_released.emit(m))
                    grid.addWidget(b, row, 1)
                    self.pad_buttons[motion] = b
                elif action.id == "gripper_toggle":
                    self.grip_btn = button("Toggle")
                    self.grip_btn.setProperty("compact", True)
                    self.grip_btn.setMinimumSize(70, 34)
                    self.grip_btn.clicked.connect(self.grip_pressed.emit)
                    grid.addWidget(self.grip_btn, row, 1)
                pair = []
                for slot in range(KeyMap.MAX_KEYS_PER_ACTION):
                    k = KeyCapture()
                    k.started.connect(lambda: self.capture_started())
                    k.finished.connect(lambda: self.capture_finished())
                    k.captured.connect(lambda key, a=action.id, s=slot: self._captured(a, s, key))
                    grid.addWidget(k, row, 2 + slot)
                    pair.append(k)
                self.buttons[action.id] = pair
                row += 1
            rows[side] = row
        self.add(columns)
        self.key_label = label("Keyboard: idle", "caption")
        self.add(self.key_label)
        self.status = Banner("", "none", hide_empty=True)
        self.add(self.status)
        self.set_pad_enabled(False)
        self.refresh()
        if keymap.load_warning:
            self.status.set(keymap.load_warning, "warn")

    # ---- outside
    def attach(self, controller):
        """Let the key controller pause itself while a key is being re-assigned."""
        self.capture_started = lambda: setattr(controller, "capturing", True)
        self.capture_finished = lambda: setattr(controller, "capturing", False)

    def set_pad_enabled(self, enabled: bool):
        for b in list(self.pad_buttons.values()) + ([self.grip_btn] if self.grip_btn else []):
            b.setEnabled(enabled)

    def highlight_key(self, motion: str, down: bool):
        b = self.pad_buttons.get(motion)
        if b is not None:
            t = theme.tokens()
            b.setStyleSheet(f"QPushButton {{ background: {t.accent}; color: {t.accent_text}; }}" if down else "")

    def set_armed(self, armed: bool):
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

    # ---- internals
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
