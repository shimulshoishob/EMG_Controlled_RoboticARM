"""Log portal: live timing of signal collection, classification and the commands sent to the xArm."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QSizePolicy, QStackedWidget,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...control.gestures import BY_ID
from ...telemetry import PipelineLog
from ..widgets import Card, Segmented, Stat, button, label

ROWS = 120                 # newest rows shown; the CSV export has everything that is kept
RECENT = 40                # the tiles average over this many newest records

# key -> (tab text, one-line explanation, columns as (heading, tooltip), column that takes the spare width)
VIEWS = {
    "gestures": ("Gesture → arm", "One row per gesture. The stages in ms add up to the total.", (
        ("Time", "When the gesture became stable"), ("Gesture", ""),
        ("Wait", "New armband samples arrived → model started"),
        ("Infer", "Preprocessing + classification of the first window that showed this gesture"),
        ("Deliver", "Model finished → control logic received the result"),
        ("Vote", "First vote → gesture counted as stable"),
        ("Dispatch", "Stable → command issued (one-shot gestures include their hold time)"),
        ("USB", "Command issued → packet written to the xArm"),
        ("Total", "Signal arrived → packet written to the xArm (a bracketed value ends at the decision)"),
        ("Sent to xArm", "The first packet this gesture caused, or why there was none")), 9),
    "predictions": ("Every window", "One row per signal window the model classified. Times in ms.", (
        ("Time", ""), ("Model output", ""), ("Conf.", "Confidence"), ("Accepted", "Passed the confidence threshold"),
        ("New signal", "Milliseconds of signal collected since the previous window"),
        ("Wait", "Samples arrived → model started"), ("Preprocess", "Filtering, features and scaling"),
        ("Classify", "The classifier alone"), ("Inference", "Preprocess + classify"),
        ("Signal → result", "Samples arrived → control logic has the result")), 1),
    "packets": ("Sent to xArm", "Every packet written to the arm, whatever caused it.", (
        ("Time", ""), ("Cause", ""), ("Data sent (target positions)", ""),
        ("Move time", "Duration the servos were given, ms"), ("USB write", "Includes the 30 ms minimum gap, ms")), 2),
    "summary": ("Per gesture", "Averages in ms over everything kept since the last Clear.", (
        ("Gesture", ""), ("Windows", "Windows classified as this"), ("Preprocess", "Average"),
        ("Classify", "Average"), ("Inference", "Average"), ("Slowest", "Slowest inference"),
        ("Signal → result", "Average"), ("Signal → decision", "Average until the gesture was stable"),
        ("Sent", "Times it caused a packet to the arm"), ("Signal → arm", "Average until the packet was written")), 0),
}


def num(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def clock(wall: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(wall)) + f".{int(wall * 1000) % 1000:03d}"


def name(gesture: str) -> str:
    return BY_ID[gesture].name if gesture in BY_ID else gesture


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


class LogPage(QWidget):
    def __init__(self, pipeline: PipelineLog, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.frames_per_s = None            # callable set by the main window
        self.window_ms = None
        self.paused = False
        self._shown = {}                    # view -> data version currently in its table

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.tile_rate = Stat("signal, frames / s")
        self.tile_window = Stat("window / step, ms")
        self.tile_infer = Stat("inference")
        self.tile_result = Stat("signal → result")
        self.tile_arm = Stat("signal → xArm")
        self.tile_window.setToolTip("Length of signal the model looks at, and how often new signal gives a new decision.")
        self.tile_infer.setToolTip("Preprocessing + classification, average of the newest windows.")
        self.tile_arm.setToolTip("Signal arrival to the packet written to the arm, average of the newest gestures.")
        for tile in (self.tile_rate, self.tile_window, self.tile_infer, self.tile_result, self.tile_arm):
            tiles.addWidget(tile, 1)
        root.addLayout(tiles)

        card = Card(fill=True)
        bar = QHBoxLayout()
        self.tabs = Segmented([(key, view[0]) for key, view in VIEWS.items()])
        self.tabs.changed.connect(self.show_view)
        card.add(self.tabs)
        self.hint = label("", "caption", wrap=True)
        bar.addWidget(self.hint, 1)
        self.pause_btn = button("Pause", tooltip="Freeze the tables; logging continues in the background.")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._pause)
        bar.addWidget(self.pause_btn)
        bar.addWidget(button("Clear", slot=self.clear))
        bar.addWidget(button("Export CSV…", "primary", slot=self._export))
        card.add(bar)

        self.stack = QStackedWidget()
        self.tables = {}
        for key, (_tab, _hint, columns, stretch) in VIEWS.items():
            table = QTableWidget(0, len(columns))
            for c, (heading, tip) in enumerate(columns):
                item = QTableWidgetItem(heading)
                item.setToolTip(tip)
                table.setHorizontalHeaderItem(c, item)
            header = table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Interactive)
            header.setSectionResizeMode(stretch, QHeaderView.Stretch)
            header.setMinimumSectionSize(48)
            table.verticalHeader().hide()
            table.verticalHeader().setDefaultSectionSize(28)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setWordWrap(False)
            table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
            table.setMinimumHeight(180)
            self.tables[key] = table
            self.stack.addWidget(table)
        card.add(self.stack, 1)
        self.note = label("Not included: the radio or cable delay from the armband, and the time the servos take to move.",
                          "caption", wrap=True)
        card.add(self.note)
        root.addWidget(card, 1)

        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh)
        self.tabs.set_current("gestures")
        self.show_view("gestures")

    # ------------------------------------------------------------------ page lifecycle
    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()
        self.timer.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self.timer.stop()

    def show_view(self, key: str):
        self.current = key
        self.stack.setCurrentWidget(self.tables[key])
        self.hint.setText(VIEWS[key][1])
        self._shown.pop(key, None)
        self.refresh()

    def _pause(self, paused: bool):
        self.paused = paused
        self.pause_btn.setText("Resume" if paused else "Pause")
        if not paused:
            self.refresh()

    def clear(self):
        self.pipeline.clear()
        self._shown.clear()
        paused, self.paused = self.paused, False
        self.refresh()
        self.paused = paused

    def _export(self):
        suggested = Path.home() / f"pipeline_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export pipeline log", str(suggested), "CSV (*.csv)")
        if not path:
            return
        try:
            count = self.pipeline.export_csv(path)
        except OSError as exc:
            self.note.setText(f"Export failed: {exc}")
            return
        self.note.setText(f"Exported {count} rows to {path}")

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        if self.paused:
            return
        self._refresh_tiles()
        versions = self.pipeline.versions
        key = self.current
        version = (versions["predictions"], versions["gestures"]) if key == "summary" else versions[key]
        if self._shown.get(key) == version:
            return
        self._shown[key] = version
        self._fill(self.tables[key], getattr(self, f"_rows_{key}")())

    def _refresh_tiles(self):
        rate = self.frames_per_s() if self.frames_per_s else None
        self.tile_rate.set("–" if rate is None else f"{rate:.0f}")
        recent = self.pipeline.snapshot("predictions", RECENT)
        every = mean([None if p.timing.since_previous is None else p.timing.since_previous * 1000 for p in recent])
        span = self.window_ms() if self.window_ms else None
        self.tile_window.set(f"{num(span, 0)} / {num(every, 0)}")
        for tile, value in ((self.tile_infer, mean([p.timing.inference_ms for p in recent])),
                            (self.tile_result, mean([p.timing.total_ms for p in recent])),
                            (self.tile_arm, mean([e.total_ms for e in self.pipeline.snapshot("gestures", RECENT)]))):
            tile.set("–" if value is None else f"{value:.0f} ms" if value >= 10 else f"{value:.1f} ms")

    def _rows_gestures(self):
        rows = []
        for e in self.pipeline.snapshot("gestures", ROWS):
            t = e.first.timing
            total = num(e.total_ms, 0) if e.total_ms is not None else f"({num(e.decision_ms, 0)})"
            sent = e.data if e.status == "sent" else f"{e.status}: {e.data}" if e.data else e.status
            rows.append((clock(e.wall), name(e.gesture), num(t.wait_ms), num(t.inference_ms), num(t.deliver_ms),
                         num(e.vote_ms, 0), num(e.dispatch_ms, 0), num(e.send_ms), total, sent))
        return rows

    def _rows_predictions(self):
        rows = []
        for p in self.pipeline.snapshot("predictions", ROWS):
            t = p.timing
            new = None if t.since_previous is None else t.since_previous * 1000
            rows.append((clock(p.wall), p.label, f"{p.confidence * 100:.0f} %", "yes" if p.gesture else "no", num(new, 0),
                         num(t.wait_ms), num(t.preprocess_ms), num(t.classify_ms), num(t.inference_ms),
                         num(t.total_ms)))
        return rows

    def _rows_packets(self):
        return [(clock(k.wall), k.cause or "–", k.data, str(k.duration_ms), num(k.write_ms))
                for k in self.pipeline.snapshot("packets", ROWS)]

    def _rows_summary(self):
        return [(name(r["gesture"]), str(r["windows"]), num(r["pre"]), num(r["cls"]), num(r["inf"]), num(r["inf_max"]),
                 num(r["result"]), num(r["decision"], 0), str(r["sent"]), num(r["arm"], 0))
                for r in self.pipeline.per_gesture()]

    @staticmethod
    def _fill(table: QTableWidget, rows):
        first_data = table.rowCount() == 0 and rows
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    table.setItem(r, c, item)
                if item.text() != text:
                    item.setText(text)
                    item.setToolTip(text if len(text) > 24 else "")
        if first_data:                               # size the columns once; the user can drag them afterwards
            table.resizeColumnsToContents()
