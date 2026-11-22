"""The ten gestures of the study and how model class names are matched to them.

Pure Python (no Qt). A trained model only knows its own class names ("Left", "fist_close",
"Thumbs_Up", ...). ``resolve_gesture`` turns such a name into one of the stable gesture IDs below,
which is what the mapping, the filter and the safety layer work with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class GestureDef:
    id: str
    name: str            # shown in the UI
    hint: str            # how to perform it
    number: int          # position in the study's gesture list


GESTURES: Tuple[GestureDef, ...] = (
    GestureDef("left", "Left", "Wrist/hand movement to the left", 1),
    GestureDef("right", "Right", "Wrist/hand movement to the right", 2),
    GestureDef("up", "Up", "Wrist/hand movement up", 3),
    GestureDef("down", "Down", "Wrist/hand movement down", 4),
    GestureDef("closed_fist", "Closed Fist", "Make a fist and hold it", 5),
    GestureDef("double_fist", "Double Closed Fist", "Clench the fist twice in quick succession", 6),
    GestureDef("victory", "Victory Sign", "Index and middle finger spread in a V", 7),
    GestureDef("thumbs_up", "Thumbs Up", "Fist with the thumb pointing up", 8),
    GestureDef("pinch", "Pinch", "Thumb and index finger pinched together", 9),
    GestureDef("rest", "Rest", "Relaxed hand, no deliberate gesture", 10),
)
GESTURE_IDS: Tuple[str, ...] = tuple(g.id for g in GESTURES)
BY_ID: Dict[str, GestureDef] = {g.id: g for g in GESTURES}

# Gestures that mean "keep moving while I hold this" in the default configuration. The real
# answer for a given setup comes from the mapping (see mapping.GestureMapping.is_continuous).
DIRECTION_GESTURES = ("left", "right", "up", "down")


def normalize_label(label: str) -> str:
    """'Fist_Close', 'fist-close' and 'fist close' all become 'fist close'."""
    return " ".join(re.sub(r"[^a-z]+", " ", str(label).lower()).split())


def resolve_gesture(label, overrides: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Map a model class name to a gesture ID, or None when it is not recognised.

    ``overrides`` ({normalised label: gesture id}) wins over the built-in word rules; it is how a
    user attaches an oddly named class (e.g. "grab2") to a gesture.
    """
    norm = normalize_label(label)
    if overrides and norm in overrides:
        return overrides[norm]
    words = set(norm.split())
    squashed = norm.replace(" ", "")
    if words & {"rest", "relax", "idle", "neutral"}:
        return "rest"
    if "double" in words or "doublefist" in squashed or "twice" in words:
        return "double_fist"
    if words & {"victory", "peace", "vsign"} or "victory" in squashed:
        return "victory"
    if "pinch" in words or "pinch" in squashed:
        return "pinch"
    # "open"/"spread"/"switch"/"mode" are how older models named the joint-switch gesture; they
    # are checked before fist so "Fist_Open_Fast" is not read as a fist.
    if words & {"thumb", "thumbs", "open", "spread", "switch", "mode"}:
        return "thumbs_up"
    if words & {"fist", "grip", "grab", "close", "closed"}:
        return "closed_fist"
    for gid in DIRECTION_GESTURES:
        if gid in words:
            return gid
    if "dow" in words:             # a typo present in one trained model
        return "down"
    return None
