"""Logs and diagnostics: packet statistics, loss, latency, and the event / error history."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QPlainTextEdit, QVBoxLayout, QWidget

from ...armband.link import LinkStats
from ...diagnostics_log import DiagnosticsLog
from ..widgets import Card, Stat, button, label


class DiagnosticsPage(QWidget):
    def __init__(self, log: DiagnosticsLog, parent=None):
        super().__init__(parent)
        self.log = log
        self._seen = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        c = Card("Communication", "Statistics of the armband link, updated once per second while connected.")
        row = QHBoxLayout()
        row.setSpacing(28)
        self.s_rate = Stat("frames / s (target 500)")
        self.s_pkts = Stat("packets / s")
        self.s_loss = Stat("packet loss")
        self.s_age = Stat("since last data")
        self.s_lat = Stat("signal → prediction")
        for s in (self.s_rate, self.s_pkts, self.s_loss, self.s_age, self.s_lat):
            row.addWidget(s)
        row.addStretch(1)
        c.add(row)
        self.detail = label("Not connected.", "mono", wrap=True)
        c.add(self.detail)
        root.addWidget(c)

        e = Card("Event log", "Connection changes, arm events, warnings and errors (newest at the bottom). Also written to logs/app.log.")
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(1000)
        self.view.setMinimumHeight(260)
        self.view.setStyleSheet("font-family: Menlo, Consolas, 'Courier New', monospace;")
        e.add(self.view)
        row = QHBoxLayout()
        row.addWidget(button("Copy", slot=self._copy))
        row.addWidget(button("Save log…", slot=self._save))
        row.addWidget(button("Clear view", "ghost", slot=self.view.clear))
        row.addStretch(1)
        self.counts = label("", "caption")
        row.addWidget(self.counts)
        e.add(row)
        root.addWidget(e)

        h = Card("Error history", "Errors only, so they are easy to find after a session.")
        self.errors = QPlainTextEdit()
        self.errors.setReadOnly(True)
        self.errors.setMaximumBlockCount(200)
        self.errors.setMaximumHeight(140)
        h.add(self.errors)
        root.addWidget(h)
        root.addStretch(1)

    def poll(self):
        """Pull new entries from the shared log (called by a GUI timer)."""
        fresh = self.log.since(self._seen)
        if not fresh:
            return
        self._seen = fresh[-1].index
        for entry in fresh:
            line = f"{entry.clock}  {entry.level.upper():7s} {entry.source:9s} {entry.message}"
            self.view.appendPlainText(line)
            if entry.level == "error":
                self.errors.appendPlainText(line)
        c = self.log.counts
        self.counts.setText(f"{c['error']} errors · {c['warning']} warnings · {c['info']} info")

    def set_latency(self, seconds: float):
        self.s_lat.set("-" if seconds < 0 else f"{seconds * 1000:.0f} ms", "good" if 0 <= seconds < 0.1 else "warn")

    def set_stats(self, s: LinkStats):
        self.s_rate.set(f"{s.frames_per_s:.0f}", "good" if s.frames_per_s >= 450 else "warn" if s.frames_per_s else "bad")
        self.s_pkts.set(f"{s.packets_per_s:.0f}")
        self.s_loss.set(f"{s.loss_percent:.1f} %", "good" if s.loss_percent < 1 else "warn" if s.loss_percent < 5 else "bad")
        age = s.last_rx_age_s
        self.s_age.set("-" if age == float("inf") else f"{age * 1000:.0f} ms", "good" if age < 0.5 else "warn")
        self.detail.setText(
            f"{s.transport or '-'}  {s.device_ip}  port {s.stream_port or '-'}  state {s.state.value}\n"
            f"packets {s.total_packets}  lost {s.lost_packets}  malformed {s.bad_packets}  foreign {s.foreign_packets}  "
            f"resyncs {s.resyncs}  late samples {s.late_frames}\n"
            f"armband firmware: queue drops {s.fw_queue_drops}, acquisition overruns {s.fw_acq_overruns}, "
            f"send failures {s.fw_send_failures}, Wi-Fi reconnects {s.fw_wifi_reconnects}")

    def clear_stats(self):
        for s in (self.s_rate, self.s_pkts, self.s_loss, self.s_age):
            s.set("-")
        self.detail.setText("Not connected.")

    def _copy(self):
        QGuiApplication.clipboard().setText(self.log.text())

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save log", str(Path.home() / "emg_arm_log.txt"), "Text (*.txt)")
        if path:
            try:
                Path(path).write_text(self.log.text(), encoding="utf-8")
            except OSError as exc:
                self.log.add("error", "app", f"Could not save the log: {exc}")
