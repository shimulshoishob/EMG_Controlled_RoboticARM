"""Centralised gesture -> robot action configuration.

Pure Python (no Qt). Nothing else in the application hard-codes what a gesture does: the
``GestureFilter`` produces a gesture ID, ``MappingStore`` says what that ID means, and the
``Teleop`` executes it through the safety layer.

    model label -> resolve_gesture -> GestureFilter -> MappingStore -> Teleop -> SafeArm

The store persists to one JSON file per user, can be exported/imported, and validates every
mapping against the robot's configuration (servo limits, speed cap) before it is accepted.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from .gestures import GESTURE_IDS, normalize_label, resolve_gesture

SCHEMA_VERSION = 1
VERTICAL = "vertical"        # "whichever joint is currently selected for Up/Down"

# action id -> (label, kind). kind: continuous (acts while held), oneshot (fires once), stop, none
ACTIONS: Dict[str, tuple] = {
    "none": ("Do nothing", "none"),
    "hold": ("Hold still (stop motion)", "stop"),
    "move_servo": ("Move a joint while held", "continuous"),
    "move_to": ("Go to a fixed position", "oneshot"),
    "gripper_toggle": ("Toggle gripper", "oneshot"),
    "gripper_open": ("Open gripper", "oneshot"),
    "gripper_close": ("Close gripper", "oneshot"),
    "switch_joint": ("Switch Up/Down joint (shoulder ↔ elbow)", "oneshot"),
    "go_home": ("Go to home pose (slow)", "oneshot"),
}
LONG_ACTIONS = ("none", "switch_joint", "gripper_toggle", "gripper_open", "gripper_close")


def action_kind(action: str) -> str:
    return ACTIONS.get(action, ("", "none"))[1]


@dataclass
class GestureMapping:
    gesture_id: str
    enabled: bool = True
    action: str = "none"
    servo: Union[int, str] = VERTICAL     # servo id, or "vertical" (the selected Up/Down joint)
    direction: int = 1                    # +1 / -1 (move_servo)
    speed: float = 0.0                    # units/s; 0 = use the global control speed (move_servo)
    target_position: int = 0              # xArm units (move_to)
    duration_ms: int = 1500               # move time (move_to)
    min_confidence: float = 0.0           # 0 = use the global threshold
    hold_s: float = 0.6                   # one-shot: gesture must be held this long before it fires
    long_action: str = "none"             # optional second action for a longer hold
    long_hold_s: float = 1.8
    model_label: str = ""                 # optional: the model class name to treat as this gesture

    @property
    def kind(self) -> str:
        return action_kind(self.action)

    @property
    def is_continuous(self) -> bool:
        return self.enabled and self.kind == "continuous"

    @classmethod
    def from_dict(cls, data: dict) -> "GestureMapping":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        return cls(**clean)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Issue:
    level: str            # "error" blocks saving; "warning" is shown but allowed
    gesture_id: str
    message: str


class MappingError(ValueError):
    pass


# --------------------------------------------------------------------------- defaults
def default_mappings(config: dict) -> Dict[str, GestureMapping]:
    """Defaults derived from the robot's own config, so a different arm gets sensible defaults.
    The first four reproduce the behaviour the application had before mapping existed."""
    m: Dict[str, GestureMapping] = {}
    for gid in ("left", "right", "up", "down"):
        entry = config.get("gestures", {}).get(gid, {})
        servo = entry.get("servo", VERTICAL if gid in ("up", "down") else 6)
        m[gid] = GestureMapping(gid, True, "move_servo", servo if servo == VERTICAL else int(servo),
                                int(entry.get("direction", 1)))
    m["closed_fist"] = GestureMapping("closed_fist", True, "gripper_toggle", hold_s=0.6,
                                      long_action="switch_joint", long_hold_s=1.8)
    m["double_fist"] = GestureMapping("double_fist", False, "go_home", hold_s=1.0)
    m["victory"] = GestureMapping("victory", True, "gripper_open", hold_s=0.6)
    m["thumbs_up"] = GestureMapping("thumbs_up", True, "switch_joint", hold_s=0.6)
    m["pinch"] = GestureMapping("pinch", True, "gripper_close", hold_s=0.6)
    m["rest"] = GestureMapping("rest", True, "hold")
    return m


# --------------------------------------------------------------------------- validation
def validate(mappings: Dict[str, GestureMapping], config: dict) -> List[Issue]:
    issues: List[Issue] = []
    arm = config["arm"]
    servos = {int(k): v for k, v in arm["servos"].items()}
    cap = float(arm["hard_max_speed_units_per_s"])
    gripper = int(arm["gripper"]["servo"])

    def err(gid, msg):
        issues.append(Issue("error", gid, msg))

    def warn(gid, msg):
        issues.append(Issue("warning", gid, msg))

    for gid in GESTURE_IDS:
        if gid not in mappings:
            err(gid, "missing mapping")
    claimed: Dict[tuple, str] = {}
    labels: Dict[str, str] = {}
    for gid, m in mappings.items():
        if gid not in GESTURE_IDS:
            err(gid, "unknown gesture")
            continue
        if m.action not in ACTIONS:
            err(gid, f"unknown action '{m.action}'")
            continue
        if m.long_action not in LONG_ACTIONS:
            err(gid, f"unknown long-hold action '{m.long_action}'")
        if not 0.0 <= m.min_confidence <= 0.99:
            err(gid, "confidence threshold must be between 0 and 0.99 (0 = use the global one)")
        if m.model_label:
            norm = normalize_label(m.model_label)
            if norm in labels and labels[norm] != gid:
                err(gid, f"model class '{m.model_label}' is already used by {labels[norm]}")
            labels[norm] = gid
        if m.kind == "continuous":
            if m.direction not in (-1, 1):
                err(gid, "direction must be +1 or -1")
            if m.servo != VERTICAL:
                try:
                    sid = int(m.servo)
                except (TypeError, ValueError):
                    err(gid, f"invalid servo '{m.servo}'")
                    continue
                if sid not in servos:
                    err(gid, f"servo {sid} does not exist in the arm configuration")
                elif sid == gripper:
                    err(gid, "the gripper is opened/closed with gripper actions, not moved continuously")
            if m.speed < 0 or m.speed > cap:
                err(gid, f"speed must be 0 (global) to {cap:g} units/s")
            if m.enabled:
                key = (VERTICAL if m.servo == VERTICAL else int(m.servo), m.direction)
                if key in claimed:
                    err(gid, f"conflicts with {claimed[key]}: both move the same joint in the same direction")
                claimed[key] = gid
        if m.kind == "oneshot":
            if not 0.0 <= m.hold_s <= 10.0:
                err(gid, "hold time must be between 0 and 10 s")
            if m.long_action != "none" and m.long_hold_s <= m.hold_s:
                err(gid, "the long-hold time must be longer than the hold time")
            if m.action == "move_to":
                try:
                    sid = int(m.servo)
                except (TypeError, ValueError):
                    err(gid, "move_to needs a specific servo")
                    continue
                if sid not in servos:
                    err(gid, f"servo {sid} does not exist in the arm configuration")
                else:
                    lo, hi = int(servos[sid]["min"]), int(servos[sid]["max"])
                    if not lo <= m.target_position <= hi:
                        err(gid, f"target {m.target_position} is outside the safe range {lo}-{hi} of servo {sid}")
                if not 300 <= m.duration_ms <= 30000:
                    err(gid, "move duration must be between 300 and 30000 ms")
            if m.action == "go_home" and m.enabled:
                warn(gid, "Go-to-home moves the whole arm; make sure the workspace is clear when you perform this gesture")
    if sum(1 for m in mappings.values() if m.enabled and m.action == "switch_joint") > 1:
        warn("-", "more than one gesture switches the Up/Down joint")
    return issues


def errors(issues: Iterable[Issue]) -> List[Issue]:
    return [i for i in issues if i.level == "error"]


# --------------------------------------------------------------------------- store
class MappingStore:
    """Holds the active mappings, persists them, and answers the runtime questions."""

    def __init__(self, config: dict, path: Optional[Path] = None):
        self.config = config
        self.path = Path(path) if path else None
        self.mappings: Dict[str, GestureMapping] = default_mappings(config)
        self.load_warning = ""
        if self.path and self.path.exists():
            self.load()

    # ---- runtime queries (hot path: keep cheap)
    def get(self, gesture_id: Optional[str]) -> Optional[GestureMapping]:
        return self.mappings.get(gesture_id) if gesture_id else None

    def label_overrides(self) -> Dict[str, str]:
        return {normalize_label(m.model_label): gid for gid, m in self.mappings.items() if m.model_label}

    def resolve(self, label: str) -> Optional[str]:
        return resolve_gesture(label, self.label_overrides())

    def is_continuous(self, gesture_id: Optional[str]) -> bool:
        m = self.get(gesture_id)
        return bool(m and m.is_continuous)

    def min_confidence(self, gesture_id: Optional[str], default: float) -> float:
        m = self.get(gesture_id)
        return m.min_confidence if m and m.min_confidence > 0 else default

    def uses_action(self, action: str, skip: Optional[str] = None, among=None) -> bool:
        """Is any enabled gesture (other than ``skip``, limited to ``among`` if given) mapped to ``action``?"""
        return any(m.enabled and m.action == action for gid, m in self.mappings.items()
                   if gid != skip and (among is None or gid in among))

    # ---- editing
    def set_all(self, new: Dict[str, GestureMapping]):
        """Replace everything, or raise MappingError listing every problem. Nothing changes on error."""
        problems = errors(validate(new, self.config))
        if problems:
            raise MappingError("\n".join(f"{i.gesture_id}: {i.message}" for i in problems))
        self.mappings = copy.deepcopy(new)
        self.save()

    def reset_defaults(self):
        self.mappings = default_mappings(self.config)
        self.save()

    def issues(self) -> List[Issue]:
        return validate(self.mappings, self.config)

    # ---- persistence
    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION,
                "mappings": {gid: m.to_dict() for gid, m in self.mappings.items()}}

    def _parse(self, data: dict) -> Dict[str, GestureMapping]:
        if not isinstance(data, dict) or not isinstance(data.get("mappings"), dict):
            raise MappingError("not a gesture-mapping file")
        if int(data.get("version", 0)) > SCHEMA_VERSION:
            raise MappingError("this mapping file was made by a newer version of the application")
        merged = default_mappings(self.config)             # gestures missing from the file keep their default
        for gid, raw in data["mappings"].items():
            if gid in GESTURE_IDS and isinstance(raw, dict):
                try:
                    merged[gid] = GestureMapping.from_dict({**raw, "gesture_id": gid})
                except TypeError as exc:
                    raise MappingError(f"{gid}: {exc}") from exc
        return merged

    def load(self):
        """Load the saved file. A damaged or invalid file never crashes the app: defaults stay active
        and the reason is kept in ``load_warning`` for the UI."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                parsed = self._parse(json.load(f))
            problems = errors(validate(parsed, self.config))
            if problems:
                raise MappingError("; ".join(f"{i.gesture_id}: {i.message}" for i in problems))
            self.mappings = parsed
            self.load_warning = ""
        except (OSError, ValueError, MappingError) as exc:
            self.load_warning = f"Saved gesture mapping ignored ({exc}); defaults are active."

    def save(self, path: Optional[Path] = None):
        target = Path(path) if path else self.path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(target) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
        os.replace(tmp, target)

    def export_to(self, path: Path):
        self.save(path)

    def import_from(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except ValueError as exc:
                raise MappingError(f"not valid JSON: {exc}") from exc
        self.set_all(self._parse(data))
