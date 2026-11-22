"""Gesture recognition result -> movement commands (no robot access, no Qt).

``CommandGenerator`` reads the stable gesture from the ``GestureFilter``, looks it up in the
``MappingStore`` and returns *what should happen*: continuous moves (while the gesture is held)
and one-shot actions (gripper, joint switch, go-to-position, home).

Repeat safety: a one-shot action fires at most once per activation of a gesture. A gesture that
stays detected does not fire again; it must be left (a different or no stable gesture) first. A
per-action cooldown additionally stops rapid re-triggering by flickering predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .gestures import BY_ID
from .mapping import ACTIONS, GestureMapping, MappingStore, VERTICAL


@dataclass
class Move:
    gesture_id: str
    servo: Union[int, str]      # servo id or "vertical"
    direction: int
    speed: float                # units/s; 0 = use the global speed
    held_s: float


@dataclass
class OneShot:
    gesture_id: str
    action: str
    mapping: GestureMapping
    reason: str = "hold"        # "hold" | "long-hold" | "release"


@dataclass
class GestureCommands:
    gesture_id: Optional[str] = None
    moves: List[Move] = field(default_factory=list)
    oneshots: List[OneShot] = field(default_factory=list)
    note: str = ""
    hint: str = ""


class CommandGenerator:
    def __init__(self, store: MappingStore, cooldown_s: float = 1.5):
        self.store = store
        self.cooldown_s = float(cooldown_s)
        self.available = None              # gestures the loaded model can produce (None = assume all)
        self.reset()

    def reset(self):
        self._seen = None                  # (gesture id, stable_since) of the activation being tracked
        self._since = 0.0
        self._fired = False
        self._long_fired = False
        self._last_fire: Dict[str, float] = {}

    # -------------------------------------------------------------- helpers
    def _long_action(self, m: GestureMapping) -> str:
        """The long-hold action in effect. A long-hold joint switch is the fallback for models that
        have no dedicated switch gesture; it is dropped once another gesture does that job."""
        if m.long_action == "switch_joint" and self.store.uses_action("switch_joint", skip=m.gesture_id,
                                                                      among=self.available):
            return "none"
        return m.long_action

    def _cooldown_ok(self, action: str, now: float) -> bool:
        return now - self._last_fire.get(action, float("-inf")) >= self.cooldown_s

    def _fire(self, out: GestureCommands, m: GestureMapping, action: str, reason: str, now: float):
        if action in ("none", "hold") or not self._cooldown_ok(action, now):
            return
        self._last_fire[action] = now
        out.oneshots.append(OneShot(m.gesture_id, action, m, reason))

    # ------------------------------------------------------------- generate
    def generate(self, gfilter, now: float) -> GestureCommands:
        out = GestureCommands()
        fresh = gfilter.is_fresh(now)
        stable = gfilter.stable if fresh else None
        key = (stable, gfilter.stable_since if stable else 0.0)

        if key != self._seen:                                   # the gesture changed: close the old activation
            prev = self.store.get(self._seen[0]) if self._seen and self._seen[0] else None
            if (prev and prev.enabled and prev.kind == "oneshot" and fresh
                    and self._long_action(prev) != "none"
                    and not self._fired and not self._long_fired and now - self._since >= prev.hold_s):
                self._fire(out, prev, prev.action, "release", now)    # tap = primary action
            self._seen, self._since = key, gfilter.stable_since if stable else now
            self._fired = self._long_fired = False

        out.gesture_id = stable
        m = self.store.get(stable)
        if not m or not m.enabled:
            return out
        held = max(0.0, now - self._since)
        if m.kind == "continuous":
            if gfilter.locked_out:
                out.note = "Same gesture too long - relax to continue"
            elif gfilter.held_for(now) > gfilter.max_move_s:
                gfilter.locked_out = True                       # a stuck classification must not keep driving
                out.note = "Same gesture too long - relax to continue"
            else:
                out.moves.append(Move(m.gesture_id, m.servo, m.direction, m.speed, held))
        elif m.kind == "oneshot":
            long_action = self._long_action(m)
            if long_action != "none":
                if held >= m.long_hold_s and not self._long_fired:
                    self._long_fired = True
                    self._fire(out, m, long_action, "long-hold", now)
            elif held >= m.hold_s and not self._fired:
                self._fired = True
                self._fire(out, m, m.action, "hold", now)
            out.hint = self._hint(m, long_action, held)
        return out

    def _hint(self, m: GestureMapping, long_action: str, held: float) -> str:
        name = BY_ID[m.gesture_id].name
        label = ACTIONS[m.action][0]
        if m.action == "go_home":
            label = "go home"
        if self._long_fired:
            return ""
        if long_action == "none":
            return "" if self._fired else f"{name} {held:.1f}/{m.hold_s:.1f} s → {label}"
        long_label = ACTIONS[long_action][0]
        if held < m.hold_s:
            return f"{name} {held:.1f} s…"
        return f"{name} {held:.1f} s: RELAX now = {label}, keep holding to {m.long_hold_s:.1f} s = {long_label}"

    @staticmethod
    def describe(m: GestureMapping, servo_names: Dict[int, str], vertical: str = "") -> str:
        """Plain-language result of a mapping, for the preview in the UI."""
        if not m.enabled:
            return "Disabled"
        if m.action == "move_servo":
            if m.servo == VERTICAL:
                joint = (vertical or "selected joint").lower()
            else:
                joint = servo_names.get(int(m.servo), f"servo {m.servo}")
            word = ("up" if m.direction > 0 else "down") if m.servo == VERTICAL else ("+" if m.direction > 0 else "−")
            speed = f" at {m.speed:g} u/s" if m.speed else ""
            return f"Move {joint.replace('_', ' ')} {word}{speed} while held"
        if m.action == "move_to":
            name = servo_names.get(int(m.servo), f"servo {m.servo}") if str(m.servo).lstrip("-").isdigit() else str(m.servo)
            return f"Move {name.replace('_', ' ')} to {m.target_position} over {m.duration_ms / 1000:.1f} s"
        text = ACTIONS.get(m.action, (m.action,))[0]
        if m.long_action != "none":
            text += f"; long hold: {ACTIONS[m.long_action][0]}"
        return text
