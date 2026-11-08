"""Step 1 - Model: pick the trained Random Forest and see what each gesture will do."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView, QListWidget, QListWidgetItem, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView)

from ...ml.inference import LoadedModel, find_models
from ...paths import MODELS_DIR, PROJECT_ROOT
from ..widgets import Banner, Card, button, label

ACTION_TEXT = {
    "left": "Rotate base left", "right": "Rotate base right", "up": "Selected joint up",
    "down": "Selected joint down", "grip": "Hold ≈0.6 s then relax: toggle gripper  ·  hold ≈1.8 s: switch joint",
    "switch": "Switch shoulder / elbow", "rest": "Hold still", None: "Ignored (not mapped)"}


class ModelPage(QWidget):
    model_requested = pyqtSignal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        c1 = Card("Choose your trained model",
                  "The model turns the EMG signal into gestures. Use one trained on recordings of the person who "
                  "will wear the armband, with the armband in the same position.", step=1)
        self.use_btn = button("Use selected model", "primary", slot=self._use_selected)
        c1.header_right.addWidget(self.use_btn)
        self.list = QListWidget()
        self.list.setMinimumHeight(170)
        self.list.setMaximumHeight(240)
        self.list.itemDoubleClicked.connect(lambda _i: self._use_selected())
        c1.add(self.list)
        row = QHBoxLayout()
        row.addWidget(button("Browse for a .joblib file…", slot=self._browse))
        row.addWidget(button("Rescan folders", "ghost", slot=self.refresh))
        row.addStretch(1)
        c1.add(row)
        self.status = Banner("No model loaded yet.", "none")
        c1.add(self.status)
        self.card_pick = c1
        root.addWidget(c1)

        c2 = Card("What each gesture does", "Names come from the model; the app matches them to arm actions.")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Gesture in the model", "Arm action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(190)
        c2.add(self.table)
        root.addWidget(c2)
        root.addStretch(1)
        self.refresh()

    def refresh(self):
        self.list.clear()
        roots = [MODELS_DIR, PROJECT_ROOT.parent / "BioWaveEMG_ArmBand" / "trained_model"]
        last = self.settings.get("last_model")
        paths = find_models(*roots)
        if last and Path(last).exists() and Path(last) not in paths:
            paths.insert(0, Path(last))
        for p in paths[:200]:
            item = QListWidgetItem(f"{p.parent.name}\n{p.parent.parent.name}/{p.name}")
            item.setData(Qt.UserRole, str(p))
            self.list.addItem(item)
            if last and str(p) == last:
                self.list.setCurrentItem(item)
        if not paths:
            self.status.set("No models found in the “models” folder. Use Browse to pick a rf_realtime_model.joblib.", "warn")

    def _use_selected(self):
        item = self.list.currentItem()
        if item is None:
            self.status.set("Select a model in the list first.", "warn")
            return
        self.model_requested.emit(item.data(Qt.UserRole))

    def _browse(self):
        start = str(MODELS_DIR) if MODELS_DIR.exists() else str(PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(self, "Load Random Forest model", start, "Joblib model (*.joblib)")
        if path:
            self.model_requested.emit(path)

    def show_loaded(self, loaded: LoadedModel, mapping: dict, problems: list):
        name = Path(loaded.path).parent.name
        tone = "warn" if problems else "good"
        text = (f"Loaded “{name}”  ·  {len(loaded.class_names)} gestures  ·  {loaded.input_channels} input channels  ·  "
                f"{loaded.window_samples}-sample window")
        if problems:
            text += "\n" + "\n".join(problems)
        self.status.set(text, tone)
        self.card_pick.set_done(True)
        self.table.setRowCount(len(loaded.class_names))
        for r, cls in enumerate(loaded.class_names):
            self.table.setItem(r, 0, QTableWidgetItem(cls))
            self.table.setItem(r, 1, QTableWidgetItem(ACTION_TEXT.get(mapping.get(cls), "")))
        self.table.resizeRowsToContents()
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == loaded.path:
                self.list.setCurrentRow(i)

    def show_error(self, message: str):
        self.status.set(message, "bad")
