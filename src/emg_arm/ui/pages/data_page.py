"""Settings & data: appearance, configuration files, raw-signal recording, arm calibration, advanced motion tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from PyQt5.QtCore import QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ...paths import RECORDINGS_DIR
from ..widgets import Banner, Card, ComboBox, DoubleSpinBox, SpinBox, button, label, two_columns

# key -> (label, min, max, step, suffix, default-from-config)
TUNING_FIELDS = {
    "lookahead": ("Lookahead", 40, 500, 10, " ms", None),
    "accel": ("Acceleration limit", 50, 2000, 50, " u/s²", None),
    "decel": ("Deceleration limit", 50, 4000, 50, " u/s²", None),
    "jerk": ("Jerk limit", 500, 50000, 500, " u/s³", None),
    "min_speed": ("Minimum speed", 0, 60, 1, " u/s", None),
    "min_lead": ("Minimum lead", 0, 20, 1, " units", None),
    "reversal_guard": ("Direction-change guard", 0, 1.5, 0.05, " s", None),
}


def open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class DataPage(QWidget):
    theme_changed = pyqtSignal(str)
    tuning_changed = pyqtSignal(dict)         # {key: value} in the units of TUNING_FIELDS (lookahead in ms)
    calibration_saved = pyqtSignal()
    export_requested = pyqtSignal()
    import_requested = pyqtSignal()

    def __init__(self, theme_mode: str, config: dict, settings, parent=None):
        super().__init__(parent)
        self.recording = False
        self.config, self.settings = config, settings
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        columns, left, right = two_columns(2, 3)
        root.addLayout(columns)

        c3 = Card("Appearance", "Colour theme of the whole app.")
        self.theme_combo = ComboBox()
        for key, text in (("auto", "Match my system"), ("light", "Light"), ("dark", "Dark"), ("pink", "Pink")):
            self.theme_combo.addItem(text, key)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(theme_mode)))
        self.theme_combo.currentIndexChanged.connect(lambda _i: self.theme_changed.emit(self.theme_combo.currentData()))
        c3.add(self.theme_combo)
        left.addWidget(c3)
        # --------------------------------------------------------------- config files
        f = Card("Configuration file", "Gesture mapping, keyboard shortcuts, smoothing and calibration in one file, "
                 "for backup or to move to another computer (macOS ↔ Windows).")
        frow = QVBoxLayout()
        frow.addWidget(button("Export configuration…", slot=self.export_requested.emit))
        frow.addWidget(button("Import configuration…", slot=self.import_requested.emit))
        f.add(frow)
        self.config_note = label("", "caption", wrap=True)
        f.add(self.config_note)
        left.addWidget(f)

        # -------------------------------------------------------------- calibration
        cal = Card("Calibration: servo range and rest position",
                   "Narrow a servo's travel or move the home (rest) pose. Values can never go outside the safe limits "
                   "verified in config/arm_config.json. Applies the next time the arm is connected.")
        g2 = QGridLayout()
        g2.setHorizontalSpacing(8)
        g2.setColumnStretch(0, 1)
        for c_i, text in enumerate(("Servo", "Min", "Max", "Home / rest")):
            g2.addWidget(label(text, "caption"), 0, c_i)
        self.cal: Dict[int, tuple] = {}
        arm = config["arm"]
        gripper = int(arm["gripper"]["servo"])
        for r, (sid, servo) in enumerate(sorted(arm["servos"].items(), key=lambda kv: int(kv[0])), start=1):
            sid = int(sid)
            if sid == gripper:
                continue
            lo, hi = int(servo["min"]), int(servo["max"])
            g2.addWidget(label(f"{servo['name'].replace('_', ' ').title()}  (servo {sid})"), r, 0)
            mn, mx = SpinBox(), SpinBox()
            for w, v in ((mn, lo), (mx, hi)):
                w.setRange(lo, hi)
                w.setValue(v)
                w.editingFinished.connect(w.clearFocus)
            home = None
            if str(sid) in arm["home"]:
                home = SpinBox()
                home.setRange(lo, hi)
                home.setValue(int(arm["home"][str(sid)]))
                home.editingFinished.connect(home.clearFocus)
                g2.addWidget(home, r, 3)
            g2.addWidget(mn, r, 1)
            g2.addWidget(mx, r, 2)
            self.cal[sid] = (mn, mx, home, lo, hi)
        cal.add(g2)
        crow = QHBoxLayout()
        crow.addWidget(button("Save calibration", "primary", slot=self._save_calibration))
        crow.addWidget(button("Restore shipped limits", "ghost", slot=self._reset_calibration))
        crow.addStretch(1)
        cal.add(crow)
        self.cal_note = Banner("", "none", hide_empty=True)
        cal.add(self.cal_note)
        self.load_calibration()
        right.addWidget(cal)
        right.addStretch(1)

        # ---------------------------------------------------------------- smoothing
        c = Card("Advanced: motion smoothing", "Reduces vibration of the base servo. Changes apply immediately, even while connected. "
                 "If the base still buzzes at low speed, raise Minimum speed and Minimum lead; if it feels sluggish, lower Lookahead.")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.tuning: Dict[str, QDoubleSpinBox] = {}
        for r, (key, (text, lo, hi, step, suffix, _)) in enumerate(TUNING_FIELDS.items()):
            grid.addWidget(label(text, "muted"), r // 2, (r % 2) * 2)
            w = DoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(2 if step < 1 else 0)
            w.setSuffix(suffix)
            w.editingFinished.connect(w.clearFocus)
            self.tuning[key] = w
            grid.addWidget(w, r // 2, (r % 2) * 2 + 1)
        c.add(grid)
        row = QHBoxLayout()
        row.addWidget(button("Apply smoothing", "primary", slot=self._apply_tuning))
        row.addWidget(button("Restore defaults", "ghost", slot=self._reset_tuning))
        row.addStretch(1)
        c.add(row)
        self.tuning_note = label("", "caption", wrap=True)
        c.add(self.tuning_note)
        self.load_tuning()
        adv = button("Show advanced motion smoothing", "ghost")
        adv.setCheckable(True)
        adv.toggled.connect(lambda on: (c.setVisible(on), adv.setText("Hide advanced motion smoothing" if on
                                                                     else "Show advanced motion smoothing")))
        c.setVisible(False)
        left.addStretch(1)
        root.addWidget(adv)                                  # full width, under both columns
        root.addWidget(c)

        root.addStretch(1)

    # ------------------------------------------------------------------ smoothing
    def default_tuning(self) -> Dict[str, float]:
        a, f = self.config["arm"], self.config["filter"]
        return {"lookahead": float(a.get("lookahead_ms", 120)), "accel": float(a.get("accel_units_per_s2", 500)),
                "decel": float(a.get("decel_units_per_s2", 1000)), "jerk": float(a.get("jerk_units_per_s3", 6000)),
                "min_speed": float(a.get("min_speed_units_per_s", 8)), "min_lead": float(a.get("min_lead_units", 3)),
                "reversal_guard": float(f.get("reversal_guard_s", 0.2))}

    def current_tuning(self) -> Dict[str, float]:
        """Defaults overlaid with the saved user values."""
        values = self.default_tuning()
        saved = self.settings.get("tuning") or {}
        values.update({k: float(v) for k, v in saved.items() if k in values})
        return values

    def load_tuning(self):
        for key, value in self.current_tuning().items():
            self.tuning[key].setValue(value)

    def _apply_tuning(self):
        values = {k: w.value() for k, w in self.tuning.items()}
        self.settings.set("tuning", values)
        self.tuning_changed.emit(values)
        self.tuning_note.setText("Applied and saved.")

    def _reset_tuning(self):
        self.settings.set("tuning", {})
        for key, value in self.default_tuning().items():
            self.tuning[key].setValue(value)
        self.tuning_changed.emit(self.default_tuning())
        self.tuning_note.setText("Defaults restored.")

    # ---------------------------------------------------------------- calibration
    def load_calibration(self):
        ranges = self.settings.get("servo_ranges") or {}
        homes = self.settings.get("home_overrides") or {}
        for sid, (mn, mx, home, lo, hi) in self.cal.items():
            pair = ranges.get(str(sid))
            mn.setValue(int(pair[0]) if pair else lo)
            mx.setValue(int(pair[1]) if pair else hi)
            if home is not None:
                home.setValue(int(homes.get(str(sid), self.config["arm"]["home"][str(sid)])))

    def _save_calibration(self):
        ranges, homes = {}, {}
        for sid, (mn, mx, home, lo, hi) in self.cal.items():
            if mn.value() >= mx.value():
                self.cal_note.set(f"Servo {sid}: Min must be below Max.", "bad")
                return
            if home is not None and not mn.value() <= home.value() <= mx.value():
                self.cal_note.set(f"Servo {sid}: the home position must lie between Min and Max.", "bad")
                return
            if (mn.value(), mx.value()) != (lo, hi):
                ranges[str(sid)] = [mn.value(), mx.value()]
            if home is not None and home.value() != int(self.config["arm"]["home"][str(sid)]):
                homes[str(sid)] = home.value()
        self.settings.set("servo_ranges", ranges)
        self.settings.set("home_overrides", homes)
        self.cal_note.set("Saved. It takes effect the next time the arm is connected.", "good")
        self.calibration_saved.emit()

    def _reset_calibration(self):
        self.settings.set("servo_ranges", {})
        self.settings.set("home_overrides", {})
        self.load_calibration()
        self.cal_note.set("Shipped limits restored. Reconnect the arm to apply.", "good")
        self.calibration_saved.emit()
