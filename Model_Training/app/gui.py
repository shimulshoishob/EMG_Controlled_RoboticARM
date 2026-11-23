"""PySide6 GUI: Data -> Configure -> Train -> Results.

All heavy work happens outside the GUI thread: the CSV is read in a ``QThread`` and every model trains in
its own process (``engine.TrainingEngine``); the window only polls the engine from a 200 ms timer.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from string import Template
from typing import Dict, List, Optional

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings, QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QFont, QPalette
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import engine as eng
from .catalog import FAMILY_NAMES, list_models
from .paths import Dirs
from .preprocess import CATEGORICAL, IGNORE, NUMERIC, PreprocessOptions, analyze, detect_task, infer_types
from .reports import (COLUMNS, PRIMARY, best_model, comparison_table, fig_confusion, fig_cost, fig_pred_vs_actual,
                      fig_scores, set_chart_theme, write_reports)

# ------------------------------------------------------------------------------------------ themes
# Every colour in the UI comes from one of these token sets: a QPalette (so native widgets, dialogs, menus and
# the combo-box popups are readable on any macOS appearance) plus a stylesheet built from the same tokens.
THEMES: Dict[str, Dict[str, str]] = {
    "Light": dict(bg="#F5F7FB", panel="#FFFFFF", text="#1F2A44", muted="#5F6B83", border="#D9DFEC", header="#EDF1FA",
                  alt="#F8F9FD", accent="#3B5BDB", accent_hover="#364FC7", on_accent="#FFFFFF", select="#DBE4FF",
                  select_text="#1F2A44", input="#FFFFFF", ok="#2B8A3E", ok_bg="#E6F9EA", danger="#C92A2A",
                  warn="#D9480F", pending="#6B7489", track="#E3E8F3", disabled="#A5ACBC", chart_cmap="Blues",
                  chart_accent="#4C6EF5"),
    "Dark": dict(bg="#14171F", panel="#1D212B", text="#E8EBF4", muted="#A3ACC2", border="#313748", header="#262B39",
                 alt="#222733", accent="#4C6EF5", accent_hover="#5C7CFA", on_accent="#FFFFFF", select="#34406F",
                 select_text="#FFFFFF", input="#262B39", ok="#69DB7C", ok_bg="#16341F", danger="#FF8787",
                 warn="#FFA94D", pending="#A3ACC2", track="#343A4C", disabled="#6C7488", chart_cmap="magma",
                 chart_accent="#748FFC"),
    "Pink": dict(bg="#FFF0F6", panel="#FFFFFF", text="#4A1D33", muted="#8A5A72", border="#F0C4D8", header="#FDE3EE",
                 alt="#FFF7FA", accent="#D6336C", accent_hover="#C2255C", on_accent="#FFFFFF", select="#FCC2D7",
                 select_text="#4A1D33", input="#FFFFFF", ok="#2B8A3E", ok_bg="#E6F9EA", danger="#C92A2A",
                 warn="#D9480F", pending="#8A5A72", track="#F9D7E5", disabled="#C9A3B6", chart_cmap="RdPu",
                 chart_accent="#D6336C"),
}
CUR: Dict[str, str] = dict(THEMES["Light"])


def T(name: str) -> str:
    """Current theme colour token."""
    return CUR[name]


def state_color(state: str) -> str:
    return {eng.PENDING: T("pending"), eng.TRAINING: T("accent"), eng.COMPLETED: T("ok"),
            eng.FAILED: T("danger"), eng.CANCELLED: T("warn")}[state]


QSS_TEMPLATE = Template("""
* { font-size: 13px; color: $text; }
QMainWindow, QWidget#root, QDialog, QMessageBox { background: $bg; }
QToolTip { background: $panel; color: $text; border: 1px solid $border; padding: 4px; }
QTabWidget::pane { border: 0; }
QTabBar::tab { background: transparent; padding: 10px 22px; color: $muted; font-weight: 600; border-bottom: 3px solid transparent; }
QTabBar::tab:selected { color: $accent; border-bottom: 3px solid $accent; }
QTabBar::tab:disabled { color: $disabled; }
QGroupBox { background: $panel; border: 1px solid $border; border-radius: 10px; margin-top: 14px; padding: 14px 12px 10px 12px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; background: $bg; color: $text; }
QFrame#card { background: $panel; border: 1px solid $border; border-radius: 10px; }
QFrame#card QLabel { background: transparent; }
QLabel { background: transparent; }
QLabel#cardValue { font-size: 22px; font-weight: 700; }
QLabel#cardTitle { color: $muted; font-size: 12px; }
QLabel#title { font-size: 20px; font-weight: 700; }
QLabel#muted { color: $muted; }
QLabel#banner { background: $ok_bg; border: 1px solid $ok; border-radius: 10px; padding: 12px 16px; font-size: 15px; color: $ok; font-weight: 600; }
QLabel#bannerIdle { background: $panel; border: 1px solid $border; border-radius: 10px; padding: 12px 16px; color: $muted; }
QPushButton { background: $panel; border: 1px solid $border; border-radius: 8px; padding: 7px 16px; color: $text; }
QPushButton:hover { background: $select; }
QPushButton:disabled { color: $disabled; background: $header; }
QPushButton#primary { background: $accent; color: $on_accent; border: 0; font-weight: 600; padding: 9px 22px; }
QPushButton#primary:hover { background: $accent_hover; }
QPushButton#primary:disabled { background: $track; color: $disabled; }
QPushButton#danger { background: $panel; color: $danger; border: 1px solid $danger; }
QPushButton#danger:hover { background: $header; }
QPushButton#danger:disabled { color: $disabled; border: 1px solid $border; }
QTableWidget, QTreeWidget, QPlainTextEdit { background: $panel; color: $text; border: 1px solid $border; border-radius: 8px; gridline-color: $border; alternate-background-color: $alt; selection-background-color: $select; selection-color: $select_text; }
QTableWidget::item:selected, QTreeWidget::item:selected { background: $select; }
QTableCornerButton::section { background: $header; border: 0; }
QHeaderView::section { background: $header; color: $text; border: 0; border-bottom: 1px solid $border; padding: 6px 8px; font-weight: 600; }
QProgressBar { border: 0; background: $track; border-radius: 6px; text-align: center; height: 14px; color: $text; font-size: 11px; }
QProgressBar::chunk { background: $accent; border-radius: 6px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: $input; color: $text; border: 1px solid $border; border-radius: 6px; padding: 4px 8px; min-height: 22px; selection-background-color: $select; selection-color: $select_text; }
QComboBox QAbstractItemView { background: $panel; color: $text; border: 1px solid $border; selection-background-color: $select; selection-color: $select_text; outline: 0; }
QCheckBox { color: $text; spacing: 8px; background: transparent; }
QMenu { background: $panel; color: $text; border: 1px solid $border; padding: 4px; }
QMenu::item { padding: 6px 18px; color: $text; }
QMenu::item:selected { background: $select; color: $select_text; }
QStatusBar { background: $bg; color: $muted; }
QStatusBar QLabel { color: $muted; }
QScrollBar:vertical { background: $bg; width: 12px; margin: 0; }
QScrollBar::handle:vertical { background: $border; border-radius: 5px; min-height: 28px; }
QScrollBar:horizontal { background: $bg; height: 12px; margin: 0; }
QScrollBar::handle:horizontal { background: $border; border-radius: 5px; min-width: 28px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QSplitter::handle { background: transparent; }
""")


def build_palette(t: Dict[str, str]) -> QPalette:
    p = QPalette()
    c = QColor
    for role, key in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"), (QPalette.Base, "panel"),
                      (QPalette.AlternateBase, "alt"), (QPalette.Text, "text"), (QPalette.Button, "panel"),
                      (QPalette.ButtonText, "text"), (QPalette.ToolTipBase, "panel"), (QPalette.ToolTipText, "text"),
                      (QPalette.Highlight, "select"), (QPalette.HighlightedText, "select_text"),
                      (QPalette.PlaceholderText, "muted"), (QPalette.Link, "accent"), (QPalette.BrightText, "danger")):
        p.setColor(role, c(t[key]))
    for role in (QPalette.Mid, QPalette.Dark, QPalette.Shadow):      # frames / check-box outlines drawn by Fusion
        p.setColor(role, c(t["muted"]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c(t["disabled"]))
    return p


def apply_theme(app: QApplication, name: str) -> None:
    CUR.clear()
    CUR.update(THEMES[name])
    app.setStyle("Fusion")                       # Fusion honours the palette on every macOS appearance
    app.setPalette(build_palette(CUR))
    app.setStyleSheet(QSS_TEMPLATE.substitute(CUR))
    set_chart_theme(bg=CUR["panel"], fg=CUR["text"], muted=CUR["muted"], grid=CUR["border"],
                    accent=CUR["chart_accent"], best=CUR["ok"], cmap=CUR["chart_cmap"])


# ------------------------------------------------------------------------------------------ helpers
class CSVLoader(QThread):
    loaded = Signal(object, str)
    failed = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            try:
                df = pd.read_csv(self.path, low_memory=False)
                if df.shape[1] == 1:                            # maybe ';' or tab separated
                    df = pd.read_csv(self.path, sep=None, engine="python")
            except UnicodeDecodeError:
                df = pd.read_csv(self.path, encoding="latin-1", low_memory=False)
            df.columns = [str(c) for c in df.columns]
            self.loaded.emit(df, self.path)
        except Exception as exc:                                 # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def make_table(headers: List[str], stretch_last: bool = False) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(stretch_last)
    t.setShowGrid(False)
    return t


def cell(text, align=Qt.AlignLeft | Qt.AlignVCenter, color: Optional[str] = None, bold=False) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setTextAlignment(align)
    if color:
        it.setForeground(QBrush(QColor(color)))
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    return it


def fmt(v, nd=4) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def fmt_time(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}" if m else f"{s:.1f}s"


def open_folder(path: str) -> None:
    p = Path(path)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p if p.is_dir() else p.parent)))


class StatCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        self.value = QLabel("-")
        self.value.setObjectName("cardValue")
        lay.addWidget(t)
        lay.addWidget(self.value)

    def set(self, v) -> None:
        self.value.setText(str(v))


class ChartPane(QWidget):
    """A matplotlib Figure inside Qt."""

    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(8, 4.5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def clear(self, msg: str = "") -> None:
        self.fig.clear()
        if msg:
            self.fig.set_facecolor(T("panel"))
            self.fig.text(0.5, 0.5, msg, ha="center", va="center", color=T("muted"))
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------------------------- tabs
class DataTab(QWidget):
    target_changed = Signal()
    dataset_loaded = Signal()

    def __init__(self, dirs: Dirs):
        super().__init__()
        self.dirs = dirs
        self.df: Optional[pd.DataFrame] = None
        self.path = ""
        self.info: dict = {}
        self.types: Dict[str, str] = {}
        self._loader: Optional[CSVLoader] = None
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        top = QHBoxLayout()
        self.btn_open = QPushButton("Open CSV…")
        self.btn_open.setObjectName("primary")
        self.btn_open.clicked.connect(self.choose_file)
        self.path_lbl = QLabel("No dataset loaded - open a CSV file to begin.")
        self.path_lbl.setObjectName("muted")
        top.addWidget(self.btn_open)
        top.addWidget(self.path_lbl, 1)
        top.addWidget(QLabel("Target column:"))
        self.target_box = QComboBox()
        self.target_box.setMinimumWidth(220)
        self.target_box.currentTextChanged.connect(self._on_target)
        top.addWidget(self.target_box)
        self.task_lbl = QLabel("")
        self.task_lbl.setObjectName("muted")
        top.addWidget(self.task_lbl)
        root.addLayout(top)

        cards = QHBoxLayout()
        self.cards = {k: StatCard(k) for k in ("Rows", "Columns", "Missing cells", "Rows with missing",
                                               "Duplicate rows", "Numeric", "Categorical")}
        for c in self.cards.values():
            cards.addWidget(c)
        root.addLayout(cards)

        split = QSplitter(Qt.Vertical)
        g1 = QGroupBox("Columns - role of every column (change a role to include / exclude it)")
        l1 = QVBoxLayout(g1)
        self.col_table = make_table(["Column", "Data type", "Role", "Missing", "Missing %", "Unique", "Example"])
        l1.addWidget(self.col_table)
        g2 = QGroupBox("Statistical summary")
        l2 = QVBoxLayout(g2)
        self.stat_table = make_table([])
        l2.addWidget(self.stat_table)
        split.addWidget(g1)
        split.addWidget(g2)
        split.setSizes([360, 260])
        root.addWidget(split, 1)

    # ---- loading
    def choose_file(self):
        start = str(self.dirs.datasets if self.dirs.datasets.exists() else Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open dataset", start, "CSV files (*.csv *.tsv *.txt);;All files (*)")
        if path:
            self.load(path)

    def load(self, path: str):
        self.btn_open.setEnabled(False)
        self.path_lbl.setText(f"Loading {Path(path).name} …")
        self._loader = CSVLoader(path)
        self._loader.loaded.connect(self._loaded)
        self._loader.failed.connect(self._failed)
        self._loader.start()

    def _failed(self, msg: str):
        self.btn_open.setEnabled(True)
        self.path_lbl.setText("Could not read the file.")
        QMessageBox.critical(self, "Could not read CSV", msg)

    def _loaded(self, df: pd.DataFrame, path: str):
        self.btn_open.setEnabled(True)
        if df.empty or df.shape[1] < 2:
            QMessageBox.warning(self, "Dataset too small", "The file needs at least two columns and one row.")
            self.path_lbl.setText("No dataset loaded.")
            return
        self.df, self.path = df, path
        self.path_lbl.setText(f"{Path(path).name}   ({Path(path).stat().st_size / 1e6:.2f} MB)")
        self.info = analyze(df)
        self.cards["Rows"].set(f"{self.info['rows']:,}")
        self.cards["Columns"].set(self.info["cols"])
        self.cards["Missing cells"].set(f"{self.info['missing_cells']:,}")
        self.cards["Rows with missing"].set(f"{self.info['rows_with_missing']:,}")
        self.cards["Duplicate rows"].set(f"{self.info['duplicates']:,}")
        self.target_box.blockSignals(True)
        self.target_box.clear()
        self.target_box.addItems(list(df.columns))
        self.target_box.setCurrentIndex(len(df.columns) - 1)         # label is usually the last column
        self.target_box.blockSignals(False)
        self._fill_stats()
        self._on_target(self.target_box.currentText(), reinfer=True)
        self.dataset_loaded.emit()

    # ---- tables
    def _on_target(self, target: str, reinfer: bool = True):
        if self.df is None or not target:
            return
        self.types = infer_types(self.df, target)
        self._fill_columns(target)
        self.task_lbl.setText(f"Detected task: {self.detected_task()}")
        self.target_changed.emit()

    def _fill_columns(self, target: str):
        t = self.col_table
        t.setRowCount(0)
        meta = {c["name"]: c for c in self.info["columns"]}
        for name in self.df.columns:
            m = meta[name]
            r = t.rowCount()
            t.insertRow(r)
            t.setItem(r, 0, cell(name, bold=(name == target)))
            t.setItem(r, 1, cell(m["dtype"]))
            if name == target:
                t.setItem(r, 2, cell("target", color=T("accent"), bold=True))
            else:
                box = QComboBox()
                box.addItems([NUMERIC, CATEGORICAL, IGNORE])
                box.setCurrentText(self.types.get(name, IGNORE))
                box.currentTextChanged.connect(lambda v, n=name: self.types.__setitem__(n, v))
                t.setCellWidget(r, 2, box)
            t.setItem(r, 3, cell(m["missing"], Qt.AlignRight | Qt.AlignVCenter))
            t.setItem(r, 4, cell(f"{m['missing_pct']:.1f}", Qt.AlignRight | Qt.AlignVCenter,
                                 T("danger") if m["missing_pct"] > 20 else None))
            t.setItem(r, 5, cell(m["unique"], Qt.AlignRight | Qt.AlignVCenter))
            t.setItem(r, 6, cell(m["example"]))
        t.resizeColumnsToContents()
        t.setColumnWidth(0, max(t.columnWidth(0), 160))
        t.setColumnWidth(2, 130)
        t.horizontalHeader().setStretchLastSection(True)
        n_num = sum(v == NUMERIC for v in self.types.values())
        n_cat = sum(v == CATEGORICAL for v in self.types.values())
        self.cards["Numeric"].set(n_num)
        self.cards["Categorical"].set(n_cat)

    def _fill_stats(self):
        s: pd.DataFrame = self.info["stats"].dropna(axis=1, how="all")
        t = self.stat_table
        t.clear()
        if s.empty:
            t.setRowCount(0)
            return
        cols = list(s.columns)
        t.setColumnCount(len(cols) + 1)
        t.setHorizontalHeaderLabels(["Column"] + cols)
        t.setRowCount(len(s))
        for r, (name, row) in enumerate(s.iterrows()):
            t.setItem(r, 0, cell(name, bold=True))
            for c, col in enumerate(cols, start=1):
                v = row[col]
                txt = "" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, (float, int)) else str(v)[:24])
                t.setItem(r, c, cell(txt, Qt.AlignRight | Qt.AlignVCenter))
        t.resizeColumnsToContents()

    # ---- accessors
    def target(self) -> str:
        return self.target_box.currentText()

    def feature_types(self) -> Dict[str, str]:
        return {k: v for k, v in self.types.items() if v != IGNORE}

    def detected_task(self) -> str:
        return detect_task(self.df[self.target()]) if self.df is not None else "classification"


class ConfigTab(QWidget):
    task_changed = Signal()

    def __init__(self):
        super().__init__()
        self._detected = "classification"
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(14)

        left = QVBoxLayout()
        g = QGroupBox("Task and split")
        f = QFormLayout(g)
        self.task = QComboBox()
        self.task.addItems(["auto", "classification", "regression"])
        self.task.currentTextChanged.connect(lambda _: self.task_changed.emit())
        self.test_size = QSpinBox(); self.test_size.setRange(5, 50); self.test_size.setValue(20); self.test_size.setSuffix(" %")
        self.seed = QSpinBox(); self.seed.setRange(0, 999999); self.seed.setValue(42)
        f.addRow("Task type", self.task)
        f.addRow("Test set size", self.test_size)
        f.addRow("Random seed", self.seed)
        left.addWidget(g)

        g = QGroupBox("Preprocessing (fitted on training rows only)")
        f = QFormLayout(g)
        self.dups = QCheckBox("Remove duplicate rows (before the split)"); self.dups.setChecked(True)
        self.const = QCheckBox("Remove constant features"); self.const.setChecked(True)
        self.imp_num = QComboBox(); self.imp_num.addItems(["median", "mean", "most_frequent"])
        self.scaling = QComboBox(); self.scaling.addItems(["standard", "minmax", "robust", "none"])
        self.encoding = QComboBox(); self.encoding.addItems(["onehot", "ordinal"])
        self.max_cat = QSpinBox(); self.max_cat.setRange(2, 200); self.max_cat.setValue(30)
        self.fsel = QComboBox(); self.fsel.addItems(["none", "kbest"])
        self.k_best = QSpinBox(); self.k_best.setRange(1, 5000); self.k_best.setValue(20)
        self.cw = QCheckBox("Balance class weights (classification)"); self.cw.setChecked(True)
        f.addRow(self.dups); f.addRow(self.const)
        f.addRow("Numeric missing values", self.imp_num)
        f.addRow("Scaling", self.scaling)
        f.addRow("Categorical encoding", self.encoding)
        f.addRow("Max one-hot levels", self.max_cat)
        f.addRow("Feature selection", self.fsel)
        f.addRow("Keep top K features", self.k_best)
        f.addRow(self.cw)
        left.addWidget(g)

        g = QGroupBox("Execution")
        f = QFormLayout(g)
        self.parallel = QSpinBox(); self.parallel.setRange(1, 64); self.parallel.setValue(min(4, os.cpu_count() or 2))
        self.epochs = QSpinBox(); self.epochs.setRange(1, 500); self.epochs.setValue(40)
        f.addRow("Models trained at once", self.parallel)
        f.addRow("Deep-learning epochs", self.epochs)
        hint = QLabel(f"This machine has {os.cpu_count()} CPU cores; they are shared between running models.")
        hint.setObjectName("muted"); hint.setWordWrap(True)
        f.addRow(hint)
        left.addWidget(g)
        left.addStretch(1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(430)
        root.addWidget(lw)

        right = QVBoxLayout()
        g = QGroupBox("Models (discovered in models/registry.py)")
        v = QVBoxLayout(g)
        btns = QHBoxLayout()
        for text, fn in (("Select all", self.select_all), ("Clear", self.select_none), ("Classical ML only", self.select_ml)):
            b = QPushButton(text); b.clicked.connect(fn); btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Model", "Family", "Status"])
        self.tree.setAlternatingRowColors(True)
        v.addWidget(self.tree)
        self.note = QLabel(""); self.note.setObjectName("muted"); self.note.setWordWrap(True)
        v.addWidget(self.note)
        right.addWidget(g)
        rw = QWidget(); rw.setLayout(right)
        root.addWidget(rw, 1)
        self.refresh_models()

    def set_detected(self, task: str):
        self._detected = task

    def effective_task(self) -> str:
        t = self.task.currentText()
        return self._detected if t == "auto" else t

    def refresh_models(self):
        keep = set(self.selected_keys())
        first = not self.tree.topLevelItemCount()
        self.tree.clear()
        task = self.effective_task()
        groups: Dict[str, QTreeWidgetItem] = {}
        n_unavail = 0
        for e in list_models(task):
            if e.family not in groups:
                g = QTreeWidgetItem([FAMILY_NAMES.get(e.family, e.family), "", ""])
                g.setFlags(g.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
                self.tree.addTopLevelItem(g)
                g.setExpanded(True)
                groups[e.family] = g
            not_for_task = (not e.available) and ("classif" in e.reason)
            status = "ready" if e.available else ("n/a for regression" if not_for_task else "unavailable")
            it = QTreeWidgetItem([e.label, FAMILY_NAMES.get(e.family, e.family), status])
            it.setData(0, Qt.UserRole, e.key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            if e.available:
                on = (e.key in keep) if not first else (e.family == "ml")
                it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            else:
                it.setCheckState(0, Qt.Unchecked)
                it.setFlags(it.flags() & ~Qt.ItemIsEnabled)
                it.setToolTip(0, e.reason); it.setToolTip(2, e.reason)
                it.setForeground(2, QBrush(QColor(T("muted") if not_for_task else T("danger"))))
                n_unavail += 1
            groups[e.family].addChild(it)
        for c in range(3):
            self.tree.resizeColumnToContents(c)
        extra = ""
        if task == "regression":
            extra = "Regression uses the regressor version of each classical model; deep/hybrid networks and LDA are classification-only. "
        else:
            extra = "Deep and hybrid models treat each row as a 1-channel sequence and need PyTorch. "
        self.note.setText(extra + (f"{n_unavail} model(s) unavailable - hover the row for the reason." if n_unavail else ""))

    def _iter_items(self):
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            for j in range(g.childCount()):
                yield g.child(j)

    def selected_keys(self) -> List[str]:
        return [it.data(0, Qt.UserRole) for it in self._iter_items() if it.checkState(0) == Qt.Checked]

    def _set(self, pred):
        for it in self._iter_items():
            if it.flags() & Qt.ItemIsEnabled:
                it.setCheckState(0, Qt.Checked if pred(it) else Qt.Unchecked)

    def select_all(self): self._set(lambda it: True)
    def select_none(self): self._set(lambda it: False)
    def select_ml(self): self._set(lambda it: it.text(1) == FAMILY_NAMES["ml"])

    def options(self) -> PreprocessOptions:
        return PreprocessOptions(task=self.effective_task(), test_size=self.test_size.value() / 100.0,
                                 seed=self.seed.value(), drop_duplicates=self.dups.isChecked(),
                                 drop_constant=self.const.isChecked(), impute_numeric=self.imp_num.currentText(),
                                 scaling=self.scaling.currentText(), encoding=self.encoding.currentText(),
                                 max_categories=self.max_cat.value(), feature_selection=self.fsel.currentText(),
                                 k_best=self.k_best.value(), class_weight=self.cw.isChecked(),
                                 dl_epochs=self.epochs.value())


class TrainTab(QWidget):
    start_clicked = Signal()
    cancel_clicked = Signal()
    cancel_one = Signal(str)

    HEAD = ["Model", "Family", "Status", "Progress", "Duration", "Result", "Exported file"]

    def __init__(self):
        super().__init__()
        self.keys: List[str] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)
        bar = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start training"); self.btn_start.setObjectName("primary")
        self.btn_cancel = QPushButton("■  Cancel all"); self.btn_cancel.setObjectName("danger"); self.btn_cancel.setEnabled(False)
        self.btn_start.clicked.connect(self.start_clicked)
        self.btn_cancel.clicked.connect(self.cancel_clicked)
        bar.addWidget(self.btn_start); bar.addWidget(self.btn_cancel)
        self.summary = QLabel("Choose models on the Configure tab, then start.")
        self.summary.setObjectName("muted")
        bar.addWidget(self.summary, 1)
        self.overall = QProgressBar(); self.overall.setFixedWidth(260)
        bar.addWidget(self.overall)
        root.addLayout(bar)

        split = QSplitter(Qt.Vertical)
        self.table = make_table(self.HEAD)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._menu)
        self.table.cellDoubleClicked.connect(self._open_file)
        split.addWidget(self.table)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo", 11)); self.log.setMaximumBlockCount(5000)
        split.addWidget(self.log)
        split.setSizes([420, 200])
        root.addWidget(split, 1)
        self._bars: Dict[str, QProgressBar] = {}
        self._logged = 0

    def reset(self, jobs: Dict[str, eng.Job]):
        self.keys = list(jobs)
        self.table.setRowCount(len(jobs))
        self._bars.clear()
        self.log.clear()
        self._logged = 0
        for r, (k, j) in enumerate(jobs.items()):
            self.table.setItem(r, 0, cell(j.label, bold=True))
            self.table.setItem(r, 1, cell(FAMILY_NAMES.get(j.family, j.family)))
            for c in (2, 4, 5, 6):
                self.table.setItem(r, c, cell(""))
            bar = QProgressBar(); bar.setRange(0, 100)
            self._bars[k] = bar
            self.table.setCellWidget(r, 3, bar)
        for c, w in enumerate((170, 120, 120, 200, 90, 190)):
            self.table.setColumnWidth(c, w)
        self.table.horizontalHeader().setStretchLastSection(True)

    def refresh(self, e: eng.TrainingEngine):
        done = 0
        for r, k in enumerate(self.keys):
            j = e.jobs[k]
            col = state_color(j.state)
            self.table.setItem(r, 2, cell(("● " + j.state), color=col, bold=True))
            bar = self._bars[k]
            if j.state == eng.TRAINING and j.indeterminate:
                bar.setRange(0, 0)
            else:
                bar.setRange(0, 100)
                bar.setValue(int(100 if j.state == eng.COMPLETED else j.pct))
                bar.setFormat(f"{j.message[:22]}" if j.state == eng.TRAINING else "%p%")
            self.table.setItem(r, 4, cell(fmt_time(j.elapsed) if j.t_start else "-", Qt.AlignRight | Qt.AlignVCenter))
            if j.state == eng.COMPLETED:
                r_ = j.result
                key, name, _ = PRIMARY[r_["task"]]
                self.table.setItem(r, 5, cell(f"{name} {fmt(r_['metrics'].get(key), 4)}", color=T("ok")))
                self.table.setItem(r, 6, cell(Path(r_["artifact"]).name))
                self.table.item(r, 6).setData(Qt.UserRole, r_["artifact"])
            elif j.state == eng.FAILED:
                self.table.setItem(r, 5, cell(j.error[:80], color=T("danger")))
                self.table.item(r, 5).setToolTip(j.error)
            elif j.state == eng.CANCELLED:
                self.table.setItem(r, 5, cell("cancelled", color=T("warn")))
            done += j.state in eng.TERMINAL
        n = len(self.keys)
        running = len(e.running)
        ok = sum(j.state == eng.COMPLETED for j in e.jobs.values())
        bad = sum(j.state == eng.FAILED for j in e.jobs.values())
        self.overall.setRange(0, 100)
        self.overall.setValue(int(sum(100 if j.state in eng.TERMINAL else j.pct for j in e.jobs.values()) / max(n, 1)))
        self.summary.setText(f"{running} training · {n - done - running} pending · {ok} completed · {bad} failed"
                             f" · elapsed {fmt_time((e.t_end or __import__('time').time()) - e.t_start)}")
        new = e.logs[self._logged:]
        if new:
            self.log.appendPlainText("\n".join(new))
            self._logged = len(e.logs)

    def _menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0 or r >= len(self.keys):
            return
        m = QMenu(self)
        a = QAction("Cancel this model", self)
        a.triggered.connect(lambda: self.cancel_one.emit(self.keys[r]))
        m.addAction(a)
        m.exec(self.table.viewport().mapToGlobal(pos))

    def _open_file(self, row, col):
        it = self.table.item(row, 6)
        if it is not None and it.data(Qt.UserRole):
            open_folder(it.data(Qt.UserRole))


class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.results: List[dict] = []
        self.task = "classification"
        self.paths: Dict[str, str] = {}
        self.last = None
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)
        self.banner = QLabel("Results appear here when training has finished.")
        self.banner.setObjectName("bannerIdle")
        root.addWidget(self.banner)

        split = QSplitter(Qt.Vertical)
        self.table = make_table([])
        self.table.itemSelectionChanged.connect(self._select)
        split.addWidget(self.table)
        self.tabs = QTabWidget()
        self.scores = ChartPane(); self.cost = ChartPane(); self.detail = ChartPane()
        self.files = make_table(["Model", "Saved file", "Size", "Metadata (JSON)"], stretch_last=True)
        self.files.cellDoubleClicked.connect(lambda r, c: self.files.item(r, 1) and open_folder(self.files.item(r, 1).data(Qt.UserRole)))
        self.tabs.addTab(self.scores, "Metric comparison")
        self.tabs.addTab(self.cost, "Time && size")
        self.tabs.addTab(self.detail, "Confusion matrix")
        self.tabs.addTab(self.files, "Exported models")
        split.addWidget(self.tabs)
        split.setSizes([230, 420])
        root.addWidget(split, 1)

        row = QHBoxLayout()
        self.b_models = QPushButton("Open trained_models folder")
        self.b_reports = QPushButton("Open evaluation report")
        self.b_plots = QPushButton("Open plots folder")
        self.b_log = QPushButton("Open log file")
        self._targets = {self.b_models: "models", self.b_reports: "reports", self.b_plots: "plots", self.b_log: "log"}
        for b in self._targets:
            b.setEnabled(False)
            b.clicked.connect(lambda _=False, b=b: self.paths.get(self._targets[b]) and open_folder(self.paths[self._targets[b]]))
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

    def show_results(self, e: eng.TrainingEngine, paths: Dict[str, str]):
        self.last = (e, paths)
        self.results, self.task = e.results(), e.plan.prepared.task
        self.paths = {"models": str(e.plan.dirs.trained_models), "reports": paths.get("report", paths.get("reports", "")),
                      "plots": paths.get("plots", ""), "log": str(e.log_path)}
        for b in self._targets:
            b.setEnabled(True)
        t = comparison_table(self.results, self.task)
        n_fail = sum(j.state == eng.FAILED for j in e.jobs.values())
        if t.empty:
            self.banner.setObjectName("bannerIdle"); self.banner.setStyleSheet("")
            self.banner.setText("No model completed successfully - see the Train tab and the log for the reasons.")
            self.table.setRowCount(0); self.table.setColumnCount(0)
            for c in (self.scores, self.cost, self.detail):
                c.clear("no results")
            return
        best = best_model(self.results, self.task)
        key, name, _ = PRIMARY[self.task]
        self.banner.setObjectName("banner")
        self.banner.setStyleSheet("")
        self.banner.setText(f"🏆  Best model: {best['label']}   ·   {name} {best['metrics'][key]:.4f}   ·   "
                            f"trained in {best['train_time_s']:.1f}s   ·   {best['model_size_bytes'] / 1024:.0f} KB"
                            + (f"      ({n_fail} model(s) failed)" if n_fail else ""))
        self.banner.style().unpolish(self.banner); self.banner.style().polish(self.banner)

        cols = [c for c in t.columns if c not in ("_key", "Artifact")]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(t))
        for r, (_, row) in enumerate(t.iterrows()):
            for c, name_ in enumerate(cols):
                v = row[name_]
                if name_ == "Family":
                    v = FAMILY_NAMES.get(v, v)
                txt = fmt(float(v), 4) if isinstance(v, (float,)) else str(v)
                if name_ in ("Train time (s)",):
                    txt = f"{v:.2f}"
                if name_ == "Size (KB)":
                    txt = f"{v:.0f}"
                it = cell(txt, Qt.AlignLeft | Qt.AlignVCenter if c < 3 and name_ in ("Model", "Family") else Qt.AlignRight | Qt.AlignVCenter,
                          bold=(r == 0))
                if r == 0:
                    it.setBackground(QBrush(QColor(T("ok_bg"))))
                it.setData(Qt.UserRole, row["_key"])
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()

        fig_scores(self.results, self.task, self.scores.fig); self.scores.canvas.draw_idle()
        fig_cost(self.results, self.task, self.cost.fig); self.cost.canvas.draw_idle()
        self.tabs.setTabText(2, "Confusion matrix" if self.task == "classification" else "Predicted vs actual")
        self.table.selectRow(0)

        self.files.setRowCount(len(self.results))
        for r, res in enumerate(self.results):
            a = cell(Path(res["artifact"]).name)
            a.setData(Qt.UserRole, res["artifact"])
            a.setToolTip(res["artifact"] + "\n(double-click to reveal)")
            self.files.setItem(r, 0, cell(res["label"], bold=True))
            self.files.setItem(r, 1, a)
            self.files.setItem(r, 2, cell(f"{res['model_size_bytes'] / 1024:.0f} KB", Qt.AlignRight | Qt.AlignVCenter))
            self.files.setItem(r, 3, cell(Path(res["metadata_file"]).name))
        self.files.resizeColumnsToContents()

    def _select(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or not self.results:
            return
        key = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        res = next((r for r in self.results if r["model_key"] == key), None)
        if res is None:
            return
        (fig_confusion if self.task == "classification" else fig_pred_vs_actual)(res, self.detail.fig)
        self.detail.canvas.draw_idle()


# ----------------------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, dirs: Optional[Dirs] = None):
        super().__init__()
        self.dirs = (dirs or Dirs()).ensure()
        self.engine: Optional[eng.TrainingEngine] = None
        self.setWindowTitle("Model Training Studio")
        self.resize(1280, 820)

        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 10, 0, 0)
        head = QHBoxLayout(); head.setContentsMargins(20, 0, 20, 0)
        t = QLabel("Model Training Studio"); t.setObjectName("title")
        s = QLabel("Train and compare several models in parallel and export them as .joblib"); s.setObjectName("muted")
        head.addWidget(t); head.addWidget(s, 1)
        head.addWidget(QLabel("Theme"))
        self.theme_box = QComboBox()
        self.theme_box.addItems(list(THEMES))
        self.theme_box.setMinimumWidth(110)
        head.addWidget(self.theme_box)
        lay.addLayout(head)

        self.tabs = QTabWidget()
        self.data = DataTab(self.dirs); self.config = ConfigTab(); self.train = TrainTab(); self.results = ResultsTab()
        self.tabs.addTab(self.data, "1  Data")
        self.tabs.addTab(self.config, "2  Configure")
        self.tabs.addTab(self.train, "3  Train")
        self.tabs.addTab(self.results, "4  Results")
        lay.addWidget(self.tabs, 1)

        self.data.target_changed.connect(self._task_update)
        self.data.dataset_loaded.connect(lambda: self.tabs.setCurrentIndex(0))
        self.config.task_changed.connect(self.config.refresh_models)
        self.train.start_clicked.connect(self.start_training)
        self.train.cancel_clicked.connect(self.cancel_all)
        self.train.cancel_one.connect(self._cancel_one)
        self.timer = QTimer(self); self.timer.setInterval(200); self.timer.timeout.connect(self._tick)
        self.statusBar().showMessage("Open a CSV file to begin.")
        saved = initial_theme()
        self.theme_box.setCurrentText(saved)
        self.theme_box.currentTextChanged.connect(self.set_theme)

    def set_theme(self, name: str):
        """Re-skin everything: palette + stylesheet, then rebuild the widgets that carry per-cell colours."""
        if name not in THEMES:
            return
        apply_theme(QApplication.instance(), name)
        QSettings("ModelTrainingStudio", "Studio").setValue("theme", name)
        if self.data.df is not None:
            self.data._fill_columns(self.data.target())
        self.config.refresh_models()
        if self.engine is not None:
            self.train.refresh(self.engine)
            if self.engine.finished and self.results.last is not None:
                self.results.show_results(*self.results.last)
        else:
            for c in (self.results.scores, self.results.cost, self.results.detail):
                c.clear("no results yet")

    def _task_update(self):
        self.config.set_detected(self.data.detected_task())
        self.config.refresh_models()
        self.statusBar().showMessage(f"Target '{self.data.target()}' -> {self.config.effective_task()}")

    # ---- training
    def start_training(self):
        if self.engine is not None and not self.engine.finished:
            return
        if self.data.df is None:
            QMessageBox.information(self, "No dataset", "Open a CSV file on the Data tab first.")
            self.tabs.setCurrentIndex(0)
            return
        keys = self.config.selected_keys()
        if not keys:
            QMessageBox.information(self, "No models selected", "Select at least one model on the Configure tab.")
            self.tabs.setCurrentIndex(1)
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            plan = eng.build_run(self.data.df, self.data.target(), self.data.feature_types(), self.config.options(),
                                 keys, self.dirs, self.config.parallel.value(), Path(self.data.path).name)
            self.engine = eng.TrainingEngine(plan)
        except Exception as exc:                                  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot start training", f"{exc}")
            return
        QApplication.restoreOverrideCursor()
        self.train.reset(self.engine.jobs)
        self.train.btn_start.setEnabled(False)
        self.train.btn_cancel.setEnabled(True)
        self.tabs.setTabEnabled(0, False); self.tabs.setTabEnabled(1, False)
        self.tabs.setCurrentIndex(2)
        self.engine.start()
        self.train.refresh(self.engine)
        self.timer.start()
        self.statusBar().showMessage(f"Training {len(keys)} models…")

    def cancel_all(self):
        if self.engine and not self.engine.finished:
            self.engine.cancel_all()
            self._tick()

    def _cancel_one(self, key: str):
        if self.engine:
            self.engine.cancel(key)

    def _tick(self):
        e = self.engine
        if e is None:
            return
        try:
            e.poll()
        except Exception:                                         # noqa: BLE001
            e.log("GUI poll error:\n" + traceback.format_exc())
        self.train.refresh(e)
        if e.finished:
            self.timer.stop()
            self.train.btn_start.setEnabled(True)
            self.train.btn_cancel.setEnabled(False)
            self.tabs.setTabEnabled(0, True); self.tabs.setTabEnabled(1, True)
            try:
                paths = write_reports(e)
            except Exception as exc:                              # noqa: BLE001
                e.log(f"Could not write reports: {exc}\n{traceback.format_exc()}")
                paths = {}
            self.train.refresh(e)
            self.results.show_results(e, paths)
            ok = len(e.results())
            self.statusBar().showMessage(f"Finished: {ok}/{len(e.jobs)} models exported to {self.dirs.trained_models}")
            if ok:
                self.tabs.setCurrentIndex(3)

    def closeEvent(self, ev):
        if self.engine and not self.engine.finished:
            if QMessageBox.question(self, "Training in progress", "Cancel all running training jobs and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.engine.cancel_all()
        ev.accept()


def initial_theme() -> str:
    """Saved choice, else follow the macOS / OS appearance (dark -> Dark, otherwise Light)."""
    try:
        name = QSettings("ModelTrainingStudio", "Studio").value("theme", "")
        if name in THEMES:
            return name
        if QApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark:
            return "Dark"
    except Exception:
        pass
    return "Light"


def run(argv: Optional[List[str]] = None) -> int:
    mp.freeze_support()
    app = QApplication(argv or sys.argv)
    apply_theme(app, initial_theme())
    win = MainWindow()
    win.show()
    for a in (argv or sys.argv)[1:]:
        if a.lower().endswith((".csv", ".tsv", ".txt")) and Path(a).exists():
            win.data.load(a)
    return app.exec()
