"""Keyboard control: configurable key map and the application-wide key handler.

Why this replaced the old handler (found by reading the code, then reproduced with Qt test events):

* Arrow keys / W / S / E / D were *swallowed* whenever the window was active and disarmed, so they
  never reached lists, combo boxes or spin boxes (focus conflict), and on the Control page the
  only feedback was a short grey label, which looked like "the keys do nothing".
* Letter keys and Cmd/Ctrl shortcuts were not told apart: Cmd+W (close window on macOS) counted as
  "W held". macOS also never delivers the key-up of a letter pressed while Cmd is held, so the key
  would stay "held" (and the arm moving) until the 6 s safety limit.
* Releases lost to a modal dialog, a focus change or the window going to the background were only
  partly handled.
* A, D (and Q) had no function and nothing could be re-assigned.

Rules now:
* STOP keys (Space, Esc) always work and are never remappable.
* Motion keys are only *consumed* while ARMED and no text field has focus. Disarmed, they pass
  through to the focused widget; the key tester on the Keyboard page still shows what was pressed.
* Key combinations with Ctrl/Cmd/Alt are ignored (they belong to the OS and the menu).
* Auto-repeat events never change the held state (Qt/Cocoa deliver auto-repeat as release+press pairs).
* Every held key is dropped when the window deactivates, a modal dialog opens, a text field gets
  focus, or the arm is disarmed/stopped.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QKeySequenceEdit, QLineEdit, QPlainTextEdit,
                             QTextEdit)

STOP_KEYS = (Qt.Key_Space, Qt.Key_Escape)
MODIFIER_KEYS = (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Meta, Qt.Key_Alt, Qt.Key_AltGr, Qt.Key_CapsLock,
                 Qt.Key_NumLock, Qt.Key_ScrollLock, Qt.Key_unknown)
COMBO_MODIFIERS = Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier   # Ctrl on Windows, Cmd on macOS
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class KeyAction:
    id: str
    label: str
    motion: str = ""        # name understood by Teleop.servo_for (held key)
    special: str = ""       # "grip" | "select:<joint>"
    group: str = "Arm"


def build_actions(config: dict) -> List[KeyAction]:
    actions = [
        KeyAction("base_left", "Base rotate left", "left", group="Base"),
        KeyAction("base_right", "Base rotate right", "right", group="Base"),
        KeyAction("joint_up", "Selected joint up", "up", group="Selected joint"),
        KeyAction("joint_down", "Selected joint down", "down", group="Selected joint"),
    ]
    for name in config["vertical_joints"]:
        pretty = name.replace("_", " ").title()
        actions += [KeyAction(f"{name}_up", f"{pretty} up", f"{name}_up", group=pretty),
                    KeyAction(f"{name}_down", f"{pretty} down", f"{name}_down", group=pretty)]
    actions.append(KeyAction("gripper_toggle", "Toggle gripper", special="grip", group="Gripper"))
    for name in config["vertical_joints"]:
        actions.append(KeyAction(f"select_{name}", f"Up/Down controls the {name}", special=f"select:{name}",
                                 group="Selected joint"))
    return actions


DEFAULT_KEYS: Dict[str, List[str]] = {
    "base_left": ["Left", "A"], "base_right": ["Right", "D"],
    "joint_up": ["Up"], "joint_down": ["Down"],
    "shoulder_up": ["I"], "shoulder_down": ["K"], "elbow_up": ["O"], "elbow_down": ["L"],
    "gripper_toggle": ["G"], "select_shoulder": ["S"], "select_elbow": ["E"],
}

# Defaults of version 1 (number keys selected the joint, which stopped people typing 1 and 2 into number fields)
V1_DEFAULTS: Dict[str, List[str]] = {
    "shoulder_up": ["W"], "shoulder_down": ["S"], "elbow_up": ["E"], "elbow_down": ["Q"],
    "select_shoulder": ["1"], "select_elbow": ["2"],
}


def upgrade_bindings(raw: dict, version: int) -> dict:
    """Saved maps hold every key, so move those still at an old default to the new one. Keys the user
    chose stay as they are."""
    if version >= SCHEMA_VERSION:
        return raw
    return {aid: DEFAULT_KEYS[aid] if V1_DEFAULTS.get(aid) == names else names for aid, names in raw.items()}


def key_name(key: int) -> str:
    return QKeySequence(key).toString(QKeySequence.PortableText)


def key_from_name(name: str) -> int:
    seq = QKeySequence(name, QKeySequence.PortableText)
    if seq.count() != 1:
        raise ValueError(f"'{name}' is not a single key")
    value = seq[0]
    if value & int(Qt.KeyboardModifierMask) or value in MODIFIER_KEYS:
        raise ValueError(f"'{name}': use a plain key without Ctrl/Alt/Shift")
    return value


def pretty_key(key: int) -> str:
    """Short on-screen label: arrows as symbols."""
    return {Qt.Key_Left: "←", Qt.Key_Right: "→", Qt.Key_Up: "↑", Qt.Key_Down: "↓",
            Qt.Key_Space: "Space", Qt.Key_Escape: "Esc"}.get(key, key_name(key))


class KeyMap:
    """Action id -> list of Qt key codes, with conflict checking and JSON persistence."""

    MAX_KEYS_PER_ACTION = 2

    def __init__(self, actions: List[KeyAction], path: Optional[Path] = None):
        self.actions = actions
        self.by_id = {a.id: a for a in actions}
        self.path = Path(path) if path else None
        self.bindings: Dict[str, List[int]] = {}
        self.load_warning = ""
        self.reset_defaults(save=False)
        if self.path and self.path.exists():
            self.load()

    # ---- queries
    def action_for(self, key: int) -> Optional[KeyAction]:
        for aid, keys in self.bindings.items():
            if key in keys:
                return self.by_id[aid]
        return None

    def keys_for(self, action_id: str) -> List[int]:
        return list(self.bindings.get(action_id, []))

    def label_for(self, action_id: str) -> str:
        keys = self.bindings.get(action_id, [])
        return " / ".join(pretty_key(k) for k in keys) if keys else "—"

    # ---- editing
    @staticmethod
    def check(bindings: Dict[str, List[int]], actions: Dict[str, KeyAction]) -> List[str]:
        problems, seen = [], {}
        for aid, keys in bindings.items():
            if len(keys) > KeyMap.MAX_KEYS_PER_ACTION:
                problems.append(f"{actions[aid].label}: at most {KeyMap.MAX_KEYS_PER_ACTION} keys")
            for k in keys:
                if k in STOP_KEYS:
                    problems.append(f"{pretty_key(k)} is reserved for STOP")
                elif k in MODIFIER_KEYS:
                    problems.append("modifier keys cannot be used")
                elif k in seen and seen[k] != aid:
                    problems.append(f"{pretty_key(k)} is used by both “{actions[seen[k]].label}” and “{actions[aid].label}”")
                seen[k] = aid
        return problems

    def assign(self, action_id: str, keys: List[int]):
        """Set the keys of one action. Raises ValueError (and changes nothing) on a conflict."""
        trial = {**self.bindings, action_id: list(dict.fromkeys(keys))}
        problems = self.check(trial, self.by_id)
        if problems:
            raise ValueError("; ".join(problems))
        self.bindings = trial
        self.save()

    def parse(self, raw: Dict[str, List[str]], version: int = SCHEMA_VERSION) -> Dict[str, List[int]]:
        """Key names -> key codes for every known action, validated. Raises ValueError."""
        if not isinstance(raw, dict):
            raise ValueError("keyboard map must be an object")
        raw = upgrade_bindings(raw, version)
        parsed = {a.id: [key_from_name(n) for n in raw.get(a.id, DEFAULT_KEYS.get(a.id, []))] for a in self.actions}
        problems = self.check(parsed, self.by_id)
        if problems:
            raise ValueError("; ".join(problems))
        return parsed

    def set_all(self, bindings: Dict[str, List[int]]):
        self.bindings = {a.id: list(bindings.get(a.id, [])) for a in self.actions}
        self.save()

    def reset_defaults(self, save: bool = True):
        self.bindings = {a.id: [key_from_name(n) for n in DEFAULT_KEYS.get(a.id, [])] for a in self.actions}
        if save:
            self.save()

    # ---- persistence
    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION, "bindings": {aid: [key_name(k) for k in keys]
                                                        for aid, keys in self.bindings.items()}}

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = upgrade_bindings(dict(data["bindings"]), int(data.get("version", 1)))
            loaded = {a.id: [key_from_name(n) for n in raw.get(a.id, DEFAULT_KEYS.get(a.id, []))]
                      for a in self.actions}
            problems = self.check(loaded, self.by_id)
            if problems:
                raise ValueError("; ".join(problems))
            self.bindings = loaded
            self.load_warning = ""
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.load_warning = f"Saved keyboard map ignored ({exc}); defaults are active."

    def save(self):
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(self.path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            pass


def is_text_entry(widget) -> bool:
    """True for widgets where the user is typing text (keys must go to them, not to the arm).
    Spin boxes and read-only combo boxes are *not* text entry: you adjust them with the mouse."""
    if widget is None:
        return False
    if isinstance(widget, QLineEdit):
        parent = widget.parent()
        if isinstance(parent, (QAbstractSpinBox, QComboBox)):
            return False
        return not widget.isReadOnly()
    return isinstance(widget, (QTextEdit, QPlainTextEdit, QKeySequenceEdit))


class KeyboardController(QObject):
    """Application-wide key handler for one MainWindow (installed as an event filter)."""

    held_changed = pyqtSignal(list)      # list of pretty key names currently held
    key_seen = pyqtSignal(str)           # key tester: every non-repeat key event, armed or not
    ignored = pyqtSignal(str)            # a control key was pressed but the arm is not armed

    def __init__(self, window, keymap: KeyMap, is_armed: Callable[[], bool], on_stop: Callable[[str], None],
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(window)
        self.window, self.keymap, self.is_armed, self.on_stop, self.clock = window, keymap, is_armed, on_stop, clock
        self.pressed: Dict[int, float] = {}
        self.grip_pending = False
        self.select_pending: Optional[str] = None
        self.capturing = False               # the Keyboard page is recording a new key: hands off
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            app.focusChanged.connect(self._focus_changed)

    def detach(self):
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
            try:
                app.focusChanged.disconnect(self._focus_changed)
            except TypeError:
                pass

    # ---- state used by the control loop
    def held_motions(self, now: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, since in self.pressed.items():
            action = self.keymap.action_for(key)
            if action and action.motion:
                out[action.motion] = max(out.get(action.motion, 0.0), now - since)
        return out

    def take_grip(self) -> bool:
        pending, self.grip_pending = self.grip_pending, False
        return pending

    def take_select(self) -> Optional[str]:
        pending, self.select_pending = self.select_pending, None
        return pending

    def release_all(self):
        if self.pressed:
            self.pressed.clear()
            self.held_changed.emit([])

    # ---- Qt plumbing
    def _focus_changed(self, _old, new):
        if is_text_entry(new):
            self.release_all()

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype in (QEvent.KeyPress, QEvent.KeyRelease):
            return self.handle(event, etype == QEvent.KeyPress)
        if etype in (QEvent.WindowDeactivate, QEvent.WindowBlocked) and obj is self.window:
            self.release_all()
        elif etype == QEvent.ApplicationDeactivate:
            self.release_all()
        return False

    def handle(self, event, pressed: bool) -> bool:
        """Returns True when the event was consumed."""
        if self.capturing or not self.window.isActiveWindow():
            return False
        key = event.key()
        repeat = event.isAutoRepeat()
        if key in MODIFIER_KEYS:
            return False
        if pressed and not repeat:
            self.key_seen.emit(self._describe(key))
        if key in STOP_KEYS:
            if pressed and not repeat:
                self.release_all()
                self.on_stop("Space/Esc pressed")
            return True
        if event.modifiers() & COMBO_MODIFIERS:
            return False                       # OS / menu shortcut, e.g. Cmd+W
        action = self.keymap.action_for(key)
        if action is None:
            return False
        armed = self.is_armed()
        if is_text_entry(QApplication.focusWidget()):
            self.release_all()
            return False
        if not armed:
            if pressed and not repeat:
                self.ignored.emit(f"{pretty_key(key)} → {action.label}: press ARM first (nothing moves while disarmed)")
            return False
        if repeat:
            return True                        # Cocoa/Qt auto-repeat: state is unchanged
        if action.motion:
            if pressed:
                self.pressed[key] = self.clock()
            else:
                self.pressed.pop(key, None)
            self.held_changed.emit([pretty_key(k) for k in self.pressed])
        elif pressed and action.special == "grip":
            self.grip_pending = True
        elif pressed and action.special.startswith("select:"):
            self.select_pending = action.special.split(":", 1)[1]
        return True

    def _describe(self, key: int) -> str:
        action = self.keymap.action_for(key)
        if key in STOP_KEYS:
            return f"{pretty_key(key)} → STOP (always active)"
        return f"{pretty_key(key)} → {action.label}" if action else f"{pretty_key(key)} → not assigned"
