"""Gesture Mapping: assign a robot action to each of the ten gestures, with live preview."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QListWidget,
                             QListWidgetItem, QMessageBox, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from ...control.commands import CommandGenerator
from ...control.gestures import BY_ID, GESTURES
from ...control.mapping import ACTIONS, LONG_ACTIONS, VERTICAL, GestureMapping, MappingError, MappingStore, errors, validate
from .. import theme
from ..widgets import Banner, Card, button, label, two_columns


class MappingPage(QWidget):
    mapping_saved = pyqtSignal()
    disarm_requested = pyqtSignal()

    def __init__(self, store: MappingStore, config: dict, parent=None):
        super().__init__(parent)
        self.store, self.config = store, config
        arm = config["arm"]
        self.servos = {int(k): v for k, v in arm["servos"].items()}
        self.gripper = int(arm["gripper"]["servo"])
        self.names = {sid: v.get("name", f"servo {sid}") for sid, v in self.servos.items()}
        self.cap = float(arm["hard_max_speed_units_per_s"])
        self.work: Dict[str, GestureMapping] = copy.deepcopy(store.mappings)
        self.available: set = set(BY_ID)           # gestures the loaded model can produce (all until a model is loaded)
        self.model_loaded = False
        self.classes: List[str] = []
        self.armed = False
        self.vertical = "shoulder"
        self.live_id: Optional[str] = None
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        columns, left, right = two_columns(2, 3)
        root.addLayout(columns, 1)

        # ---- live preview + list
        head = Card("Gestures", "Select one to edit it. The highlighted row is detected right now.", fill=True)
        self.live = Banner("No gesture detected yet.", "none")
        head.add(self.live)
        self.list = QListWidget()
        self.list.setMinimumHeight(240)
        self.list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.list.setWordWrap(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.currentRowChanged.connect(self._select)
        head.add(self.list, 1)
        row = QHBoxLayout()
        self.reset_btn = button("Reset to defaults", "ghost", slot=self._reset)
        self.export_btn = button("Export…", slot=self._export)
        self.import_btn = button("Import…", slot=self._import)
        for b in (self.reset_btn, self.export_btn, self.import_btn):
            b.setProperty("compact", True)
        row.addWidget(self.reset_btn)
        row.addStretch(1)
        row.addWidget(self.export_btn)
        row.addWidget(self.import_btn)
        head.add(row)
        left.addWidget(head, 1)

        # ---- editor
        self.editor = Card("Edit gesture", " ", fill=True)
        self.preview = label("", "mono", wrap=True)
        self.enabled = QCheckBox("Enabled")
        top = QHBoxLayout()
        top.setSpacing(14)
        top.addWidget(self.preview, 1)
        top.addWidget(self.enabled, 0, Qt.AlignTop)
        self.editor.add(top)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(5)
        self.rows: Dict[str, tuple] = {}

        def add_row(key, text, widget, hint=""):
            r = 2 * len(self.rows)                  # the hint sits under its field
            lab = label(text, "muted")
            grid.addWidget(lab, r, 0)
            grid.addWidget(widget, r, 1)
            if isinstance(widget, QComboBox):
                widget.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                widget.setMinimumContentsLength(12)
            h = label(hint, "caption", wrap=True) if hint else None
            if h:
                grid.addWidget(h, r + 1, 1)
            self.rows[key] = (lab, widget, h)

        self.action = QComboBox()
        for aid, (text, _k) in ACTIONS.items():
            self.action.addItem(text, aid)
        add_row("action", "Robot action", self.action)
        self.servo = QComboBox()
        add_row("servo", "Joint / servo", self.servo, "“Selected Up/Down joint” follows the shoulder/elbow switch.")
        self.direction = QComboBox()
        self.direction.addItem("Positive (+)  increases the servo value / Up", 1)
        self.direction.addItem("Negative (−)  decreases the servo value / Down", -1)
        add_row("direction", "Direction", self.direction)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0, self.cap)
        self.speed.setDecimals(0)
        self.speed.setSingleStep(10)
        self.speed.setSuffix(" units/s")
        self.speed.setSpecialValueText("Use the global speed")
        add_row("speed", "Movement speed", self.speed, f"Capped at {self.cap:g} units/s by the safety layer.")
        self.target = QSpinBox()
        self.target.setRange(0, 1000)
        add_row("target", "Target position", self.target, "xArm units. Must stay inside the servo's safe range.")
        self.duration = QSpinBox()
        self.duration.setRange(300, 30000)
        self.duration.setSingleStep(100)
        self.duration.setSuffix(" ms")
        add_row("duration", "Move duration", self.duration)
        self.hold = QDoubleSpinBox()
        self.hold.setRange(0, 10)
        self.hold.setSingleStep(0.1)
        self.hold.setDecimals(1)
        self.hold.setSuffix(" s")
        add_row("hold", "Hold before it fires", self.hold,
                "Fires once per gesture. Leave the gesture and repeat it to trigger again.")
        self.long_action = QComboBox()
        for aid in LONG_ACTIONS:
            self.long_action.addItem("None" if aid == "none" else ACTIONS[aid][0], aid)
        add_row("long_action", "Longer hold does", self.long_action,
                "Optional second action; the short action then fires when you relax.")
        self.long_hold = QDoubleSpinBox()
        self.long_hold.setRange(0.2, 12)
        self.long_hold.setSingleStep(0.1)
        self.long_hold.setDecimals(1)
        self.long_hold.setSuffix(" s")
        add_row("long_hold", "Long-hold time", self.long_hold)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0, 99)
        self.conf.setDecimals(0)
        self.conf.setSuffix(" %")
        self.conf.setSpecialValueText("Use the global threshold")
        add_row("conf", "Confidence threshold", self.conf, "Higher = fewer false triggers, slower response.")
        self.model_label = QComboBox()
        add_row("model_label", "Model class", self.model_label, "Automatic matches by name. Pick a class if your model names it differently.")
        for spin in (self.speed, self.conf):
            spin.setMinimumWidth(170)               # the "use the global ..." text would otherwise widen small windows
        grid.setColumnStretch(1, 1)
        self.editor.add(grid)
        self.editor.body.addStretch(1)
        self.issues = Banner("", "none")
        self.editor.add(self.issues)
        actions = QHBoxLayout()
        self.save_btn = button("Save mapping", "primary", slot=self._save)
        self.revert_btn = button("Revert changes", slot=self._revert)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.revert_btn)
        self.save_note = label("", "caption", wrap=True)
        actions.addStretch(1)
        self.editor.add(actions)
        self.editor.add(self.save_note)
        right.addWidget(self.editor, 1)

        for w in (self.enabled,):
            w.toggled.connect(self._edited)
        for w in (self.action, self.servo, self.direction, self.long_action, self.model_label):
            w.currentIndexChanged.connect(self._edited)
        for w in (self.speed, self.target, self.duration, self.hold, self.long_hold, self.conf):
            w.valueChanged.connect(self._edited)
        self.action.currentIndexChanged.connect(self._action_changed)
        self.servo.currentIndexChanged.connect(self._servo_changed)

        self._rebuild_list()
        self.list.setCurrentRow(0)
        if store.load_warning:
            self.save_note.setText(store.load_warning)

    # ================================================================ outside world
    def set_model_classes(self, classes: List[str]):
        self.classes = list(classes)
        self.model_loaded = True
        self.available = {self.store.resolve(c) for c in classes} - {None}
        self.available |= {gid for gid, m in self.work.items() if m.model_label and
                           any(self._same(m.model_label, c) for c in classes)}
        self._rebuild_list()
        self._load_editor()

    @staticmethod
    def _same(a: str, b: str) -> bool:
        from ...control.gestures import normalize_label
        return normalize_label(a) == normalize_label(b)

    def set_armed(self, armed: bool):
        self.armed = armed
        self._update_buttons()

    def set_vertical(self, name: str):
        self.vertical = name
        self._refresh_texts()

    def set_live(self, gesture_id: Optional[str], confidence: float = 0.0, accepted: bool = True):
        self.live_id = gesture_id
        if gesture_id is None:
            self.live.set("No stable gesture (rest / unrecognised).", "none")
        else:
            m = self.store.get(gesture_id)
            what = CommandGenerator.describe(m, self.names, self.vertical) if m else "unmapped"
            self.live.set(f"Detected: {BY_ID[gesture_id].name}  ({confidence * 100:.0f}%)   →   {what}",
                          "accent" if m and m.enabled else "warn")
        self._paint_live()

    def restyle(self):
        self._paint_live()

    # ================================================================ list
    def _rebuild_list(self):
        current = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for g in GESTURES:
            self.list.addItem(QListWidgetItem())
        self.list.blockSignals(False)
        self._refresh_texts()
        self.list.setCurrentRow(max(0, current))

    def _refresh_texts(self):
        for i, g in enumerate(GESTURES):
            m = self.work[g.id]
            tags = []
            if not m.enabled:
                tags.append("OFF")
            if self.model_loaded and g.id not in self.available:
                tags.append("not in model")
            tag = f"   [{', '.join(tags)}]" if tags else ""
            item = self.list.item(i)
            item.setText(f"{g.number:>2}.  {g.name}{tag}\n      {CommandGenerator.describe(m, self.names, self.vertical)}")
            item.setToolTip(g.hint)
        self._paint_live()

    def _paint_live(self):
        t = theme.tokens()
        for i, g in enumerate(GESTURES):
            item = self.list.item(i)
            if item is None:
                continue
            live = g.id == self.live_id
            f = item.font()
            f.setBold(live)
            item.setFont(f)
            item.setBackground(QBrush(QColor(t.accent_soft)) if live else QBrush())
            dim = (self.model_loaded and g.id not in self.available) or not self.work[g.id].enabled
            item.setForeground(QBrush(QColor(t.muted if dim else t.text)))

    # ================================================================ editor
    def _current(self) -> GestureMapping:
        return self.work[GESTURES[max(0, self.list.currentRow())].id]

    def _select(self, _row):
        self._load_editor()

    def _fill_servos(self, include_vertical: bool, select):
        self.servo.blockSignals(True)
        self.servo.clear()
        if include_vertical:
            self.servo.addItem("Selected Up/Down joint (shoulder / elbow)", VERTICAL)
        for sid in sorted(self.servos):
            if sid != self.gripper or not include_vertical:
                self.servo.addItem(f"{self.names[sid].replace('_', ' ').title()}  (servo {sid})", sid)
        idx = self.servo.findData(select)
        self.servo.setCurrentIndex(max(0, idx))
        self.servo.blockSignals(False)

    def _load_editor(self):
        m = self._current()
        g = BY_ID[m.gesture_id]
        self._loading = True
        self.editor.title_label.setText(f"{g.number}. {g.name}")
        self.editor.subtitle_label.setText(g.hint)
        self.enabled.setChecked(m.enabled)
        self.action.setCurrentIndex(max(0, self.action.findData(m.action)))
        self._fill_servos(m.action != "move_to", m.servo)
        self.direction.setCurrentIndex(0 if m.direction >= 0 else 1)
        self.speed.setValue(m.speed)
        self.target.setValue(m.target_position)
        self.duration.setValue(m.duration_ms)
        self.hold.setValue(m.hold_s)
        self.long_action.setCurrentIndex(max(0, self.long_action.findData(m.long_action)))
        self.long_hold.setValue(m.long_hold_s)
        self.conf.setValue(m.min_confidence * 100)
        self.model_label.blockSignals(True)
        self.model_label.clear()
        self.model_label.addItem("Automatic (match by name)", "")
        for c in self.classes:
            self.model_label.addItem(c, c)
        if m.model_label and self.model_label.findData(m.model_label) < 0:
            self.model_label.addItem(f"{m.model_label} (not in loaded model)", m.model_label)
        self.model_label.setCurrentIndex(max(0, self.model_label.findData(m.model_label)))
        self.model_label.blockSignals(False)
        self._loading = False
        self._apply_visibility()
        self._update_target_range()
        self._show_issues()

    def _apply_visibility(self):
        kind = ACTIONS.get(self.action.currentData(), ("", "none"))[1]
        action = self.action.currentData()
        show = {"servo": kind == "continuous" or action == "move_to",
                "direction": kind == "continuous", "speed": kind == "continuous",
                "target": action == "move_to", "duration": action == "move_to",
                "hold": kind == "oneshot", "long_action": kind == "oneshot" and action != "move_to",
                "long_hold": kind == "oneshot" and self.long_action.currentData() != "none" and action != "move_to",
                "conf": kind != "none", "model_label": True}
        for key, (lab, widget, hint) in self.rows.items():
            for w in (lab, widget, hint):
                if w is not None:
                    w.setVisible(show.get(key, True))

    def _update_target_range(self):
        sid = self.servo.currentData()
        if isinstance(sid, int) and sid in self.servos:
            lo, hi = int(self.servos[sid]["min"]), int(self.servos[sid]["max"])
            self.target.blockSignals(True)
            self.target.setRange(lo, hi)
            self.target.blockSignals(False)

    def _action_changed(self, _i):
        if self._loading:
            return
        keep = self.servo.currentData()
        self._fill_servos(self.action.currentData() != "move_to", keep)
        self._update_target_range()
        self._apply_visibility()

    def _servo_changed(self, _i):
        if not self._loading:
            self._update_target_range()

    def _edited(self, *_):
        if self._loading:
            return
        m = self._current()
        m.enabled = self.enabled.isChecked()
        m.action = self.action.currentData()
        servo = self.servo.currentData()
        m.servo = servo if servo is not None else m.servo
        m.direction = int(self.direction.currentData())
        m.speed = float(self.speed.value())
        m.target_position = int(self.target.value())
        m.duration_ms = int(self.duration.value())
        m.hold_s = float(self.hold.value())
        m.long_action = self.long_action.currentData()
        m.long_hold_s = float(self.long_hold.value())
        m.min_confidence = self.conf.value() / 100.0
        m.model_label = self.model_label.currentData() or ""
        self._apply_visibility()
        self._refresh_texts()
        self._show_issues()

    def _show_issues(self):
        m = self._current()
        self.preview.setText("Result:  " + CommandGenerator.describe(m, self.names, self.vertical))
        found = validate(self.work, self.config)
        errs = [i for i in found if i.level == "error"]
        warns = [i for i in found if i.level == "warning"]
        if errs:
            self.issues.set("Cannot save yet:\n" + "\n".join(f"• {BY_ID[i.gesture_id].name if i.gesture_id in BY_ID else i.gesture_id}: {i.message}"
                                                             for i in errs), "bad")
        elif warns:
            self.issues.set("\n".join(f"• {i.message}" for i in warns), "warn")
        else:
            self.issues.set("Mapping is valid.", "good")
        self._has_errors = bool(errs)
        self._update_buttons()

    def _dirty(self) -> bool:
        return self.work != self.store.mappings

    def _update_buttons(self):
        dirty = self._dirty()
        can_edit = not self.armed
        self.save_btn.setEnabled(dirty and not getattr(self, "_has_errors", False))
        self.revert_btn.setEnabled(dirty)
        for b in (self.reset_btn, self.import_btn):
            b.setEnabled(True)
        if self.armed:
            self.save_note.setText("Arm is armed: saving, resetting or importing disarms it first.")
        elif dirty:
            self.save_note.setText("Unsaved changes.")
        elif not self.save_note.text().startswith("Saved") and not self.store.load_warning:
            self.save_note.setText("")

    # ================================================================ actions
    def _ensure_disarmed(self) -> bool:
        """Mappings must not change while the arm can move: stop motion first."""
        if not self.armed:
            return True
        self.disarm_requested.emit()
        if self.armed:      # main window clears this synchronously via set_armed(False)
            QMessageBox.warning(self, "Arm still armed", "Disarm the arm first, then try again.")
            return False
        return True

    def _save(self):
        if not self._ensure_disarmed():
            return
        try:
            self.store.set_all(self.work)
        except MappingError as exc:
            QMessageBox.warning(self, "Mapping not saved", str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(self, "Mapping not saved", f"Could not write the file: {exc}")
            return
        self.work = copy.deepcopy(self.store.mappings)
        self.save_note.setText("Saved. The new mapping is active now.")
        self._rebuild_list()
        self._update_buttons()
        self.mapping_saved.emit()

    def _revert(self):
        self.work = copy.deepcopy(self.store.mappings)
        self._rebuild_list()
        self._load_editor()

    def _reset(self):
        if not self._ensure_disarmed():
            return
        if QMessageBox.question(self, "Reset to defaults", "Replace all ten gesture mappings with the defaults?") != QMessageBox.Yes:
            return
        self.store.reset_defaults()
        self.work = copy.deepcopy(self.store.mappings)
        self.save_note.setText("Saved. Defaults restored.")
        self._rebuild_list()
        self._load_editor()
        self.mapping_saved.emit()

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export gesture mapping", str(Path.home() / "gesture_mapping.json"),
                                              "JSON (*.json)")
        if path:
            try:
                self.store.export_to(Path(path))
                self.save_note.setText(f"Saved. Exported the saved mapping to {path}")
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import gesture mapping", str(Path.home()), "JSON (*.json)")
        if not path or not self._ensure_disarmed():
            return
        try:
            self.store.import_from(Path(path))
        except (MappingError, OSError) as exc:
            QMessageBox.warning(self, "Import failed", f"The file was not imported and nothing changed:\n{exc}")
            return
        self.work = copy.deepcopy(self.store.mappings)
        self.save_note.setText("Saved. Imported and activated.")
        self._rebuild_list()
        self._load_editor()
        self.mapping_saved.emit()
