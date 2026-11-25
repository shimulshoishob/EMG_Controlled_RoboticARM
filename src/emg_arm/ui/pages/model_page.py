"""Gesture model: pick the trained Random Forest and see what each gesture will do."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView, QListWidget, QListWidgetItem, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView)

from ...ml.inference import LoadedModel
from ...paths import MODELS_DIR, PROJECT_ROOT
from ...control.commands import CommandGenerator
from ...control.gestures import BY_ID
from .. import theme
from ..widgets import Banner, Card, button, label, two_columns


class ModelPage(QWidget):
    model_requested = pyqtSignal(str)

    def __init__(self, settings, store, servo_names, parent=None):
        super().__init__(parent)
        self.settings, self.store, self.servo_names = settings, store, servo_names
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        columns, left, right = two_columns(3, 5)
        root.addLayout(columns, 1)

        c1 = Card("Choose your trained model",
                  "Use one trained on the person who will wear the armband, with the armband in the same position.",
                  step=1, fill=True)
        self.list = QListWidget()
        self.list.setMinimumHeight(120)
        self.list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.ElideMiddle)             # keeps both ends of a long folder path readable
        self.list.itemDoubleClicked.connect(lambda _i: self._use_selected())
        c1.add(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(button("Upload .joblib files…", slot=self._upload))
        row.addWidget(button("Clear uploaded files", "ghost",
                             "Empties this list. The files themselves stay on your computer.", slot=self._clear))
        row.addStretch(1)
        c1.add(row)
        self.use_btn = button("Use selected model", slot=self._use_selected)
        self.use_btn.setProperty("big", True)
        theme.set_role(self.use_btn, "primary")
        self.use_btn.setMinimumHeight(40)
        c1.add(self.use_btn)
        self.status = Banner("No model loaded yet.", "none")
        c1.add(self.status)
        self.card_pick = c1
        left.addWidget(c1, 1)

        c2 = Card("What each gesture does", "Model classes matched to the ten study gestures. Change the actions under "
                  "Gesture mapping.", fill=True)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model class", "Recognised as", "Training\nwindows", "Robot action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)   # long actions wrap in full
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(190)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        c2.add(self.table, 1)
        right.addWidget(c2, 1)
        self.refresh()

    def refresh(self, select: str = ""):
        """Show the uploaded models."""
        self.list.clear()
        paths = [p for p in (self.settings.get("model_files") or []) if Path(p).is_file()]
        wanted = select or self.settings.get("last_model")
        for path in paths:
            p = Path(path)
            item = QListWidgetItem(f"{p.stem}\n{p.parent}")
            item.setData(Qt.UserRole, str(p))
            item.setToolTip(str(p))
            self.list.addItem(item)
            if wanted and str(p) == wanted:
                self.list.setCurrentItem(item)
        if not paths:
            self.status.set("No models yet. Click “Upload .joblib files…” and pick one or more trained models.", "none")

    def _upload(self):
        trained = PROJECT_ROOT / "Model_Training" / "outputs" / "trained_models"
        start = str(next((d for d in (trained, MODELS_DIR, PROJECT_ROOT) if d.exists()), PROJECT_ROOT))
        paths, _ = QFileDialog.getOpenFileNames(self, "Upload trained models", start, "Joblib model (*.joblib)")
        if not paths:
            return
        self.add_files(paths)

    def add_files(self, paths):
        """Add .joblib files to the list (duplicates ignored) and select the first one added."""
        files = list(self.settings.get("model_files") or [])
        added = []
        for p in map(str, paths):
            if p.lower().endswith(".joblib") and p not in files and p not in added:
                added.append(p)
        self.settings.set("model_files", files + added)
        self.refresh(select=(added or [str(Path(p)) for p in paths])[0])
        count = len(added)
        self.status.set(f"Added {count} model{'s' if count != 1 else ''}. Select one and press “Use selected model”."
                        if count else "Those files are already in the list.", "none")

    def _clear(self):
        """Empty the list (never deletes files) and forget the last model so the next start is clean."""
        self.settings.set("model_files", [])
        self.settings.set("last_model", "")
        self.refresh()
        if self.table.rowCount():
            self.status.set("Uploaded files cleared from the list. The loaded model stays active until you load another.", "none")

    def _use_selected(self):
        item = self.list.currentItem()
        if item is None:
            self.status.set("Select a model in the list first.", "warn")
            return
        self.model_requested.emit(item.data(Qt.UserRole))

    def show_loaded(self, loaded: LoadedModel, mapping: dict, problems: list):
        name = Path(loaded.path).stem if Path(loaded.path).parent.name == "trained_models" else Path(loaded.path).parent.name
        tone = "warn" if problems else "good"
        text = (f"Loaded “{name}”  ·  {len(loaded.class_names)} gestures  ·  {loaded.input_channels} input channels  ·  "
                f"{loaded.window_samples}-sample window")
        if loaded.test_accuracy is not None:
            text += f"  ·  test accuracy {loaded.test_accuracy * 100:.0f} %"
        if problems:
            text += "\n" + "\n".join(problems)
        self.status.set(text, tone)
        self.card_pick.set_done(True)
        files = list(self.settings.get("model_files") or [])
        if loaded.path not in files:                       # a model loaded any other way still appears in the list
            self.settings.set("model_files", files + [loaded.path])
            self.refresh(select=loaded.path)
        self.table.setRowCount(len(loaded.class_names))
        for r, cls in enumerate(loaded.class_names):
            self.table.setItem(r, 0, QTableWidgetItem(cls))
            gid = mapping.get(cls)
            m = self.store.get(gid)
            self.table.setItem(r, 1, QTableWidgetItem(BY_ID[gid].name if gid else "not recognised"))
            n = (loaded.train_windows or {}).get(cls)
            item = QTableWidgetItem("-" if n is None else str(n))
            if n is not None and n < 50:                       # very few examples: this gesture will be unreliable
                item.setToolTip("Few training examples - record more of this gesture and retrain.")
            self.table.setItem(r, 2, item)
            self.table.setItem(r, 3, QTableWidgetItem(
                CommandGenerator.describe(m, self.servo_names) if m else "Ignored (not mapped)"))
        self.table.resizeRowsToContents()
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == loaded.path:
                self.list.setCurrentRow(i)

    def show_error(self, message: str):
        self.status.set(message, "bad")
