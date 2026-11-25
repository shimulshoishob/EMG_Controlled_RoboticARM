"""Export / import of the whole user configuration as one JSON file (mapping, keys, tuning, calibration).

Pure Python. Importing validates everything first and applies nothing when anything is invalid.
"""

from __future__ import annotations

import json
from pathlib import Path

from .control.mapping import MappingError, MappingStore

BUNDLE_VERSION = 1
SETTING_KEYS = ("control_mode", "min_confidence", "tuning", "servo_ranges", "home_overrides")


def effective_config(base: dict, settings) -> dict:
    """The robot config with the user's calibration applied. Ranges can only be *narrowed* and the
    home pose must stay inside them; anything else in the settings is ignored, so a bad or
    hand-edited settings file can never widen the verified safety limits."""
    import copy
    cfg = copy.deepcopy(base)
    servos = cfg["arm"]["servos"]
    for sid, pair in (settings.get("servo_ranges") or {}).items():
        try:
            lo, hi = int(pair[0]), int(pair[1])
            blo, bhi = int(servos[str(sid)]["min"]), int(servos[str(sid)]["max"])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if blo <= lo < hi <= bhi:
            servos[str(sid)]["min"], servos[str(sid)]["max"] = lo, hi
    for sid, pos in (settings.get("home_overrides") or {}).items():
        try:
            pos = int(pos)
            lo, hi = int(servos[str(sid)]["min"]), int(servos[str(sid)]["max"])
        except (KeyError, TypeError, ValueError):
            continue
        if str(sid) in cfg["arm"]["home"] and lo <= pos <= hi:
            cfg["arm"]["home"][str(sid)] = pos
    # Keep the home pose valid if a narrowed range excludes the shipped one.
    for sid, pos in list(cfg["arm"]["home"].items()):
        lo, hi = int(servos[sid]["min"]), int(servos[sid]["max"])
        cfg["arm"]["home"][sid] = min(hi, max(lo, int(pos)))
    return cfg


def export_bundle(path: Path, store: MappingStore, keymap, settings) -> None:
    data = {"bundle_version": BUNDLE_VERSION, "mapping": store.to_json(), "keymap": keymap.to_json(),
            "settings": {k: settings.get(k) for k in SETTING_KEYS}}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def import_bundle(path: Path, store: MappingStore, keymap, settings) -> None:
    """Apply a bundle. Raises ValueError (MappingError is one) with *nothing* changed if any part is invalid."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the file: {exc}") from exc
    if not isinstance(data, dict) or "bundle_version" not in data:
        raise ValueError("not an EMG Robotic Arm configuration file")
    if int(data["bundle_version"]) > BUNDLE_VERSION:
        raise ValueError("this file was made by a newer version of the application")
    mappings = store._parse(data["mapping"]) if "mapping" in data else None
    if mappings is not None:
        from .control.mapping import errors, validate
        problems = errors(validate(mappings, store.config))
        if problems:
            raise MappingError("; ".join(f"{i.gesture_id}: {i.message}" for i in problems))
    saved_keys = data.get("keymap")
    bindings = keymap.parse(saved_keys["bindings"], int(saved_keys.get("version", 1))) if "keymap" in data else None
    new_settings = {k: v for k, v in (data.get("settings") or {}).items() if k in SETTING_KEYS}
    if new_settings.get("control_mode") not in (None, "hybrid", "gestures", "keyboard"):
        raise ValueError("unknown control mode")
    # everything parsed: apply
    if mappings is not None:
        store.set_all(mappings)
    if bindings is not None:
        keymap.set_all(bindings)
    for key, value in new_settings.items():
        settings.set(key, value)
