"""Guided dialogs: first-time Wi-Fi setup over USB, and the connection doctor."""

from __future__ import annotations

from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
                             QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget)

from ..armband import diagnostics, netinfo, serial_setup
from . import theme
from .tasks import run_task
from .widgets import Banner, Card, Dot, button, label


class UsbSetupDialog(QDialog):
    """Send the Wi-Fi name and password to the armband through its USB cable. Needed once per network."""
    provisioned = pyqtSignal(str)          # emits the Wi-Fi name

    def __init__(self, parent=None, default_ssid: str = "", default_port: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Set up the armband's Wi-Fi")
        self.setMinimumSize(700, 780)
        self.setModal(True)
        self._busy = False
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(14)
        root.addWidget(label("Set up the armband's Wi-Fi", "h1"))
        root.addWidget(label("Do this once for each Wi-Fi network. The armband remembers it, so afterwards "
                             "you can unplug the USB cable and run it from its battery.", "lead", wrap=True))

        # -- step 1
        c1 = Card("Connect the armband with USB", "Use the armband's USB port (the one used to flash it).", step=1)
        row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(320)
        row.addWidget(self.port_combo, 1)
        row.addWidget(button("Refresh", slot=self.refresh_ports))
        self.check_btn = button("Check armband", "primary", slot=self.check_armband)
        row.addWidget(self.check_btn)
        c1.add(row)
        self.device_banner = Banner("No armband checked yet. Plug it in and press “Check armband”.", "none")
        c1.add(self.device_banner)
        root.addWidget(c1)

        # -- step 2
        c2 = Card("Enter your Wi-Fi", "The same network this computer is on. The armband only supports 2.4 GHz Wi-Fi.", step=2)
        form = QVBoxLayout()
        form.setSpacing(6)
        form.addWidget(label("Wi-Fi name (SSID)", "caption"))
        self.ssid = QLineEdit(default_ssid)
        self.ssid.setPlaceholderText("e.g. LabNetwork-2.4G")
        form.addWidget(self.ssid)
        form.addWidget(label("Password", "caption"))
        prow = QHBoxLayout()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText("leave empty for an open network")
        prow.addWidget(self.password, 1)
        self.show_pw = QCheckBox("Show")
        self.show_pw.toggled.connect(lambda on: self.password.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        prow.addWidget(self.show_pw)
        form.addLayout(prow)
        c2.add(form)
        root.addWidget(c2)

        # -- step 3
        c3 = Card("Send to the armband", "It restarts and joins the network (about 10 seconds).", step=3)
        self.send_btn = button("Send Wi-Fi to armband", "success", slot=self.send)
        c3.header_right.addWidget(self.send_btn)
        self.result = Banner("", "none")
        self.result.hide()
        c3.add(self.result)
        root.addWidget(c3)

        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(button("Close", slot=self.reject))
        root.addLayout(foot)
        self._default_port = default_port
        self.refresh_ports()

    # ----------------------------------------------------------------
    def refresh_ports(self):
        current = self.port_combo.currentData() or self._default_port
        self.port_combo.clear()
        ports = serial_setup.list_ports()
        for p in ports:
            self.port_combo.addItem(p.label, p.device)
        if not ports:
            self.port_combo.addItem("No USB serial ports found - plug the armband in", None)
        idx = self.port_combo.findData(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.check_btn.setEnabled(not busy)
        self.send_btn.setEnabled(not busy)
        QApplication.setOverrideCursor(Qt.BusyCursor) if busy else QApplication.restoreOverrideCursor()

    def check_armband(self):
        port = self.port_combo.currentData()
        if not port:
            self.device_banner.set("Choose the armband's USB port first.", "warn")
            return
        self._set_busy(True)
        self.device_banner.set("Asking the armband… (up to 15 seconds if it was just plugged in)", "accent")
        run_task(serial_setup.probe, port, on_done=self._probed, on_failed=self._probe_failed,
                 on_progress=lambda m: self.device_banner.set(m, "accent"), parent=self)

    def _probed(self, info):
        self._set_busy(False)
        wifi = "Wi-Fi already saved" if info.wifi_saved else "no Wi-Fi saved yet"
        imu = "IMU ready" if info.imu_ready else "IMU not detected (EMG still works)"
        self.device_banner.set(f"Found {info.name}  ·  ID {info.device_id}  ·  firmware {info.firmware}  ·  {imu}  ·  {wifi}",
                               "good" if info.imu_ready else "warn")

    def _probe_failed(self, message):
        self._set_busy(False)
        self.device_banner.set(message, "bad")

    def send(self):
        port = self.port_combo.currentData()
        ssid = self.ssid.text().strip()
        if not port:
            self._show(f"Choose the armband's USB port in step 1.", "warn")
            return
        if not ssid:
            self._show("Type your Wi-Fi name in step 2.", "warn")
            return
        self._set_busy(True)
        self._show("Sending…", "accent")
        run_task(serial_setup.provision, port, ssid, self.password.text(),
                 on_done=lambda _r: self._sent(ssid), on_failed=self._send_failed, parent=self)

    def _sent(self, ssid):
        self._set_busy(False)
        self._show(f"Done. The armband is restarting and joining “{ssid}”. Close this window; the app will now "
                   "search for it. If it is not found, check the Wi-Fi name and password and that the network is 2.4 GHz.",
                   "good")
        self.provisioned.emit(ssid)

    def _send_failed(self, message):
        self._set_busy(False)
        self._show(message, "bad")

    def _show(self, text, tone):
        self.result.show()
        self.result.set(text, tone)


class DoctorDialog(QDialog):
    """Runs the connection checks and lists them with plain-language fixes."""

    ICON = {diagnostics.OK: ("✓", "good"), diagnostics.WARN: ("!", "warn"),
            diagnostics.FAIL: ("✕", "bad"), diagnostics.SKIP: ("–", "none")}

    def __init__(self, parent=None, device_ip: str = "", key: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Connection doctor")
        self.setMinimumSize(660, 560)
        self.device_ip, self.key = device_ip, key
        self.results: List[diagnostics.Check] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(12)
        root.addWidget(label("Connection doctor", "h1"))
        root.addWidget(label("Runs the same steps a person would, one by one, and tells you exactly what to change.",
                             "lead", wrap=True))
        row = QHBoxLayout()
        row.addWidget(label("Armband address", "caption"))
        self.ip_edit = QLineEdit(device_ip)
        self.ip_edit.setPlaceholderText("192.168.1.42")
        self.ip_edit.setMaximumWidth(200)
        row.addWidget(self.ip_edit)
        self.run_btn = button("Run checks", "primary", slot=self.run)
        row.addWidget(self.run_btn)
        row.addStretch(1)
        self.copy_btn = button("Copy report", slot=self.copy_report)
        row.addWidget(self.copy_btn)
        root.addLayout(row)
        self.summary = Banner("Press “Run checks”.", "none")
        root.addWidget(self.summary)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        self.list = QVBoxLayout(host)
        self.list.setContentsMargins(0, 0, 0, 0)
        self.list.setSpacing(8)
        self.list.addStretch(1)
        scroll.setWidget(host)
        root.addWidget(scroll, 1)
        foot = QHBoxLayout()
        foot.addStretch(1)
        foot.addWidget(button("Close", slot=self.accept))
        root.addLayout(foot)

    def run(self):
        while self.list.count() > 1:
            item = self.list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.results = []
        self.run_btn.setEnabled(False)
        self.summary.set("Checking…", "accent")
        ip = self.ip_edit.text().strip()
        run_task(diagnostics.run_checks, ip, self.key, on_done=self._finished, on_failed=self._crashed, parent=self,
                 on_progress=None)
        # results arrive in one batch when the worker finishes (a check can take a few seconds)

    def _crashed(self, message):
        self.run_btn.setEnabled(True)
        self.summary.set(f"The check itself failed: {message}", "bad")

    def _finished(self, results):
        self.run_btn.setEnabled(True)
        self.results = results
        for check in results:
            self.list.insertWidget(self.list.count() - 1, self._row(check))
        bad = [c for c in results if c.status == diagnostics.FAIL]
        warn = [c for c in results if c.status == diagnostics.WARN]
        self.summary.set(diagnostics.summarize(results), "bad" if bad else "warn" if warn else "good")

    def _row(self, check: diagnostics.Check) -> QFrame:
        card = Card()
        theme.set_role(card, tone={"ok": "none", "warn": "warn", "fail": "bad", "skip": "none"}[check.status])
        lay = QHBoxLayout()
        icon, tone = self.ICON[check.status]
        badge = label(icon, "badge", "good" if tone == "good" else "none")
        badge.setAlignment(Qt.AlignCenter)
        if tone in ("warn", "bad"):
            badge.setStyleSheet(f"background: {getattr(theme.tokens(), tone + '_soft')}; "
                                f"color: {getattr(theme.tokens(), tone)};")
        lay.addWidget(badge, 0, Qt.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(label(check.title, "h2"))
        text.addWidget(label(check.detail, wrap=True, selectable=True))
        if check.fix:
            text.addWidget(label("→ " + check.fix, "muted", wrap=True))
        lay.addLayout(text, 1)
        card.add(lay)
        return card

    def copy_report(self):
        lines = [f"[{c.status.upper()}] {c.title}: {c.detail}" + (f"  FIX: {c.fix}" if c.fix else "") for c in self.results]
        QApplication.clipboard().setText("\n".join(lines))
