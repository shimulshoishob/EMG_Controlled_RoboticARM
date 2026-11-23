"""The model list shown in the GUI: every model discovered in ``models/registry.py`` plus availability.

Nothing is hard-coded here - classification models come straight from ``registry.SPECS`` so a model
added to the registry later shows up in the application automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..models.registry import SPECS
from ..utils import have, import_error
from .regression import NOT_FOR_REGRESSION, REGRESSION_LABELS, REGRESSION_REQUIRES

FAMILY_NAMES = {"ml": "Classical ML", "dl": "Deep learning", "hybrid": "Hybrid"}


@dataclass
class ModelEntry:
    key: str
    label: str
    family: str
    available: bool
    reason: str = ""

    @property
    def file_stem(self) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "", self.label) or self.key


def list_models(task: str) -> List[ModelEntry]:
    out: List[ModelEntry] = []
    for key, spec in SPECS.items():
        if task == "regression":
            if spec.family != "ml":
                out.append(ModelEntry(key, spec.label, spec.family, False, "classification-only network"))
                continue
            if key in NOT_FOR_REGRESSION:
                out.append(ModelEntry(key, spec.label, spec.family, False, NOT_FOR_REGRESSION[key]))
                continue
            label = REGRESSION_LABELS.get(key, spec.label)
            mod = REGRESSION_REQUIRES.get(key)
            ok = mod is None or have(mod)
            out.append(ModelEntry(key, label, spec.family, ok,
                                  "" if ok else f"{mod}: {import_error(mod)} (pip install {spec.pip or mod})"))
            continue
        missing = [m for m in spec.requires if not have(m)]
        why = "; ".join(f"{m}: {import_error(m)}" for m in missing)
        if missing and spec.pip:
            why += f" (pip install {spec.pip})"
        out.append(ModelEntry(key, spec.label, spec.family, not missing, why))
    return out


def entry_for(key: str, task: str) -> ModelEntry:
    for e in list_models(task):
        if e.key == key:
            return e
    raise KeyError(key)
