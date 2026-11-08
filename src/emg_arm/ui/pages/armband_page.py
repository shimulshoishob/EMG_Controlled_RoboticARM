"""Step 2 - Armband: get from "box on the desk" to "live signal", one numbered card at a time."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QVBoxLayout, QWidget)

from ...armband import control, netinfo, serial_setup
from ...armband.link import LinkState, LinkStats
from ...armband.protocol import CONTROL_PORT, DeviceInfo, ProtocolError, check_key, read_key_from_secrets_h
from ...paths import FIRMWARE_DIR
from .. import theme
from ..bridge import ArmbandBridge
from ..dialogs import DoctorDialog, UsbSetupDialog
from ..tasks import run_task
from ..widgets import Banner, Card, Chip, Segmented, SignalScope, Stat, button, label

STATE_TONE = {LinkState.IDLE: "none", LinkState.CONNECTING: "accent", LinkState.STREAMING: "good",
              LinkState.STALLED: "warn", LinkState.RECONNECTING: "warn", LinkState.FAILED: "bad"}


class ArmbandPage(QWidget):
    link_changed = pyqtSignal(str)            # LinkState value (for the sidebar and the bottom bar)

    def __init__(self, bridge: ArmbandBridge, settings, parent=None):
        super().__init__(parent)
        self.bridge, self.settings = bridge, settings
        self.state = LinkState.IDLE
        self._failure = ""
        self._found = {}
        self._scan_task = None
        self._scan_retries = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        # ---- how are you connecting?
        mode_card = Card("How do you want to connect?",
                         "Wi-Fi is wireless and safest for the person wearing the electrodes. USB is a good fallback. "
                         "The virtual armband lets you try everything with no hardware.")
        self.mode = Segmented([("wifi", "Wi-Fi (recommended)"), ("usb", "USB cable"), ("virtual", "Virtual armband (demo)")])
        self.mode.changed.connect(self._mode_changed)
        mode_card.add(self.mode)
        root.addWidget(mode_card)

        # Only the chosen mode's cards are shown (a QStackedWidget would reserve the tallest page's height).
        self.mode_pages = {"wifi": self._build_wifi(), "usb": self._top(self._build_usb(), stretch=False),
                           "virtual": self._top(self._build_virtual(), stretch=False)}
        for page in self.mode_pages.values():
            root.addWidget(page)

        root.addWidget(self._build_connect())
        root.addWidget(self._build_health())
        root.addStretch(1)

        self.mode.set_current(settings.get("connection") if settings.get("connection") in ("wifi", "usb", "virtual") else "wifi")
        self._mode_changed(self.mode.current())
        bridge.state.connect(self.on_state)
        bridge.stats.connect(self.on_stats)
        bridge.frames.connect(self.scope.push)
        bridge.closed.connect(self._update_buttons)
        self._update_key_status()
        self._restore_last_device()

    @staticmethod
    def _top(card: QWidget, stretch: bool = True) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(card)
        if stretch:
            lay.addStretch(1)
        return host

    # ================================================================ builders
    def _build_wifi(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        c1 = Card("Give the armband your Wi-Fi", "First time only. The armband remembers the network, so you can "
                  "skip this step next time.", step=1)
        self.setup_btn = button("Set up Wi-Fi over USB…", "primary", slot=self.open_usb_setup)
        c1.header_right.addWidget(self.setup_btn)
        c1.add(label("You need: the armband plugged into this computer with its USB cable, and the name and "
                     "password of a 2.4 GHz Wi-Fi network that this computer is also connected to.", "muted", wrap=True))
        lay.addWidget(c1)
        self.card_setup = c1

        c2 = Card("Find your armband", "The armband must be powered and on the same Wi-Fi as this computer.", step=2)
        self.scan_btn = button("Scan network", "primary", slot=self.scan)
        c2.header_right.addWidget(self.scan_btn)
        self.found_list = QListWidget()
        self.found_list.setMaximumHeight(120)
        self.found_list.setMinimumHeight(64)
        self.found_list.itemSelectionChanged.connect(self._pick_device)
        c2.add(self.found_list)
        self.scan_banner = Banner("Press “Scan network”. It takes a few seconds.", "none")
        c2.add(self.scan_banner)
        row = QHBoxLayout()
        row.addWidget(label("Or type its IP address:", "muted"))
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("192.168.1.42")
        self.ip_edit.setMaximumWidth(220)
        self.ip_edit.textChanged.connect(self._update_buttons)
        row.addWidget(self.ip_edit)
        row.addStretch(1)
        c2.add(row)
        lay.addWidget(c2)
        self.card_find = c2

        c3 = Card("Enter the access key", "A password shared by the app and the firmware. It is the "
                  "DEVICE_ACCESS_KEY_VALUE inside the secrets.h you flashed.", step=3)
        self.import_btn = button("Import from secrets.h…", slot=self.import_key)
        c3.header_right.addWidget(self.import_btn)
        krow = QHBoxLayout()
        self.key_edit = QLineEdit(self.settings.get("access_key"))
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("paste the key, or use Import")
        self.key_edit.textChanged.connect(self._key_edited)
        krow.addWidget(self.key_edit, 1)
        self.key_show = button("Show", slot=self._toggle_key)
        krow.addWidget(self.key_show)
        c3.add(krow)
        self.key_status = label("", "caption")
        c3.add(self.key_status)
        lay.addWidget(c3)
        self.card_key = c3
        return host

    def _build_usb(self) -> QWidget:
        c = Card("Choose the USB port", "Plug the armband into this computer. Data is sent over the cable, no Wi-Fi needed.", step=1)
        row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(360)
        row.addWidget(self.port_combo, 1)
        row.addWidget(button("Refresh", slot=self.refresh_ports))
        c.add(row)
        c.add(label("Electrical safety: while the electrodes are on a person, run the laptop on battery "
                    "(unplug its charger) when using the USB cable. Close the Arduino Serial Monitor first.", "muted", wrap=True))
        self.refresh_ports()
        return c

    def _build_virtual(self) -> QWidget:
        c = Card("Virtual armband", "A software copy of the armband runs inside this app and sends realistic EMG "
                 "and IMU data over the same network protocol.", step=1)
        c.add(label("Use it to learn the app, to test the arm in simulation, or when the hardware is not at hand. "
                    "It will not produce real gestures, so the model will mostly say Rest.", "muted", wrap=True))
        return c

    def _build_connect(self) -> Card:
        c = Card("Connect and check the signal", "", step=4)
        self.connect_card = c
        self.connect_btn = button("Connect", "success", slot=self.toggle_connection)
        self.connect_btn.setMinimumWidth(150)
        c.header_right.addWidget(self.connect_btn)
        self.banner = Banner("Not connected.", "none")
        c.add(self.banner)
        row = QHBoxLayout()
        self.doctor_btn = button("Connection doctor", "ghost", "Checks network, key and firewall step by step",
                                 slot=self.open_doctor)
        row.addWidget(self.doctor_btn)
        row.addStretch(1)
        c.add(row)
        return c

    def _build_health(self) -> Card:
        c = Card("Live signal", "Eight EMG channels. At rest each line should be calm; squeeze your hand and the "
                 "matching channels should swell.")
        row = QHBoxLayout()
        row.setSpacing(28)
        self.stat_rate = Stat("frames / second  (target 500)")
        self.stat_loss = Stat("packets lost")
        self.stat_age = Stat("since last data")
        self.stat_fw = Stat("armband warnings")
        for s in (self.stat_rate, self.stat_loss, self.stat_age, self.stat_fw):
            row.addWidget(s)
        row.addStretch(1)
        c.add(row)
        self.scope = SignalScope()
        c.add(self.scope)
        self.detail = label("", "caption", wrap=True)
        c.add(self.detail)
        return c

    # ================================================================ lifecycle
    def showEvent(self, e):
        super().showEvent(e)
        self.scope.start()

    def hideEvent(self, e):
        super().hideEvent(e)
        self.scope.stop()

    def restyle(self):
        self.scope.restyle()

    # ================================================================ mode / inputs
    def _mode_changed(self, key: str):
        for name, page in self.mode_pages.items():
            page.setVisible(name == key)
        self.settings.set("connection", key)
        self._update_buttons()

    def refresh_ports(self):
        current = self.port_combo.currentData() or self.settings.get("usb_port")
        self.port_combo.clear()
        ports = serial_setup.list_ports()
        for p in ports:
            self.port_combo.addItem(p.label, p.device)
        if not ports:
            self.port_combo.addItem("No USB serial ports found - plug the armband in, then Refresh", None)
        idx = self.port_combo.findData(current)
        if idx >= 0:
            self.port_combo.setCurrentIndex(idx)

    def _restore_last_device(self):
        ip, did = self.settings.get("device_ip"), self.settings.get("device_id")
        if ip and not ip.startswith("127."):
            self.ip_edit.setText(ip)
            self.scan_banner.set(f"Last used armband: {ip}. Press Connect, or scan again if it moved.", "none")

    def _toggle_key(self):
        hidden = self.key_edit.echoMode() == QLineEdit.Password
        self.key_edit.setEchoMode(QLineEdit.Normal if hidden else QLineEdit.Password)
        self.key_show.setText("Hide" if hidden else "Show")

    def _key_edited(self, text):
        self.settings.set("access_key", text.strip())
        self._update_key_status()
        self._update_buttons()

    def _update_key_status(self):
        key = self.key_edit.text().strip()
        problem = check_key(key)
        if problem:
            self.key_status.setText(problem if key else "No key yet.")
            theme.set_role(self.key_status, tone="warn" if key else "none")
        else:
            self.key_status.setText(f"Key looks valid ({len(key)} characters).")
            theme.set_role(self.key_status, tone="good")
        self.card_key.set_done(problem is None)

    def import_key(self):
        start = ""
        for candidate in (FIRMWARE_DIR / "secrets.h",
                          Path(__file__).resolve().parents[5] / "BioWaveEMG_ArmBand" / "ESP32_S3 Setup" / "EMG_IMU_WIFI_SETUP_V2"):
            if candidate.exists():
                start = str(candidate)
                break
        path, _ = QFileDialog.getOpenFileName(self, "Choose the secrets.h you flashed", start, "secrets.h (secrets.h);;Header files (*.h)")
        if not path:
            return
        try:
            key = read_key_from_secrets_h(Path(path).read_text(encoding="utf-8", errors="replace"))
        except (OSError, ProtocolError) as exc:
            QMessageBox.warning(self, "Access key", str(exc))
            return
        self.key_edit.setText(key)

    # ================================================================ USB setup + scan
    def open_usb_setup(self):
        if self.bridge.active:
            QMessageBox.information(self, "Wi-Fi setup", "Disconnect the armband first (step 4).")
            return
        dlg = UsbSetupDialog(self, default_ssid=self.settings.get("wifi_ssid") or netinfo.current_wifi_ssid() or "",
                             default_port=self.settings.get("usb_port"))
        dlg.provisioned.connect(self._provisioned)
        dlg.exec_()

    def _provisioned(self, ssid: str):
        self.settings.set("wifi_ssid", ssid)
        self.card_setup.set_done(True)
        self._scan_retries = 8
        self.scan_banner.set("The armband is restarting and joining Wi-Fi… searching for it.", "accent")
        QTimer.singleShot(5000, self.scan)

    def scan(self):
        if self._scan_task is not None and self._scan_task.isRunning():
            return
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("Scanning…")
        self.scan_banner.set("Looking for armbands on your network…", "accent")
        self._scan_task = run_task(control.discover, 2.5, on_done=self._scanned, on_failed=self._scan_failed, parent=self)

    def _scan_failed(self, msg):
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan network")
        self.scan_banner.set(f"Scan failed: {msg}", "bad")

    def _scanned(self, report: control.DiscoveryReport):
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("Scan network")
        devices = [d for d in report.found if not d.ip.startswith("127.")]
        self._found = {d.ip: d for d in devices}
        self.found_list.clear()
        for d in devices:
            item = QListWidgetItem(f"{d.name}   ·   {d.ip}   ·   firmware {d.firmware}"
                                   + ("" if d.imu_ready else "   ·   IMU not detected"))
            item.setData(Qt.UserRole, d.ip)
            self.found_list.addItem(item)
        if devices:
            self._scan_retries = 0
            self.found_list.setCurrentRow(0)
            self.scan_banner.set(f"Found {len(devices)} armband(s). Select one, then continue to the access key.", "good")
            self.card_find.set_done(True)
        elif self._scan_retries > 0:
            self._scan_retries -= 1
            self.scan_banner.set(f"Still searching… ({self._scan_retries} tries left)", "accent")
            QTimer.singleShot(2500, self.scan)
        else:
            nets = ", ".join(f"{i.name} ({i.ip})" for i in report.interfaces if not i.is_virtual) or "no network"
            vpn = " A VPN is active - turn it off." if netinfo.vpn_interfaces(report.interfaces) else ""
            self.scan_banner.set(
                f"No armband answered on {nets}.{vpn} Check that it is powered, that it joined the same 2.4 GHz Wi-Fi "
                "(redo step 1 if the Wi-Fi changed), and that the router does not isolate wireless clients. "
                "You can also type its IP address below, or run the Connection doctor.", "warn")

    def _pick_device(self):
        item = self.found_list.currentItem()
        if item:
            ip = item.data(Qt.UserRole)
            self.ip_edit.setText(ip)

    # ================================================================ connect
    def _target_ip(self) -> str:
        return self.ip_edit.text().strip()

    def _update_buttons(self, *_):
        busy = self.bridge.busy
        active = self.bridge.active
        self.connect_btn.setEnabled(not busy)
        for w in (self.setup_btn, self.scan_btn, self.mode, self.ip_edit, self.found_list, self.key_edit, self.import_btn):
            w.setEnabled(not active)
        self.port_combo.setEnabled(not active)
        if busy:
            self.connect_btn.setText("Disconnecting…")
        elif active:
            self.connect_btn.setText("Disconnect")
            theme.set_role(self.connect_btn, "danger")
        else:
            self.connect_btn.setText("Connect")
            theme.set_role(self.connect_btn, "success")

    def toggle_connection(self):
        if self.bridge.active:
            self.bridge.disconnect()
            self._update_buttons()
            return
        self._failure = ""
        mode = self.mode.current()
        self.scope.clear()
        if mode == "wifi":
            ip = self._target_ip()
            if not netinfo.is_valid_ipv4(ip):
                self.banner.set("Step 2 first: scan for the armband or type its IP address (like 192.168.1.42).", "warn")
                return
            problem = check_key(self.key_edit.text())
            if problem:
                self.banner.set("Step 3: " + problem, "warn")
                return
            d = self._found.get(ip)
            self.settings.set("device_ip", ip)
            self.settings.set("device_id", d.device_id if d else self.settings.get("device_id"))
            self.bridge.connect_wifi(ip, self.key_edit.text().strip(), d.device_id if d else self.settings.get("device_id"))
        elif mode == "usb":
            port = self.port_combo.currentData()
            if not port:
                self.banner.set("No USB port selected. Plug the armband in and press Refresh.", "warn")
                return
            self.settings.set("usb_port", port)
            self.bridge.connect_usb(port)
        else:
            self.bridge.connect_virtual()
        self.banner.set("Connecting…", "accent")
        self._update_buttons()

    def open_doctor(self):
        ip = "" if self.mode.current() != "wifi" else self._target_ip()
        DoctorDialog(self, ip, self.key_edit.text().strip()).exec_()

    # ================================================================ link feedback
    def on_state(self, state_value: str, message: str):
        state = LinkState(state_value)
        self.state = state
        if state == LinkState.FAILED:
            self._failure = message
            self.banner.set(message, "bad")
            self.bridge.disconnect()                       # clean up; keep the message on screen
        elif state == LinkState.IDLE and self._failure:
            pass                                           # keep the failure text
        else:
            self.banner.set(message or state.value, STATE_TONE[state])
        if state != LinkState.STREAMING:
            self.stat_age.set("-")
        self.connect_card.set_done(state == LinkState.STREAMING)
        self._update_buttons()
        self.link_changed.emit(state.value)

    def on_stats(self, s: LinkStats):
        rate_ok = s.frames_per_s >= 450
        self.stat_rate.set(f"{s.frames_per_s:.0f}", "good" if rate_ok else "warn" if s.frames_per_s > 0 else "bad")
        self.stat_loss.set(f"{s.loss_percent:.1f} %", "good" if s.loss_percent < 1 else "warn" if s.loss_percent < 5 else "bad")
        age = s.last_rx_age_s
        self.stat_age.set("-" if age == float("inf") else f"{age * 1000:.0f} ms", "good" if age < 0.5 else "warn")
        warnings = s.fw_queue_drops + s.fw_acq_overruns + s.fw_send_failures
        self.stat_fw.set(str(warnings), "good" if warnings == 0 else "warn")
        where = f"{s.transport}  {s.device_ip}" + (f"  ·  data port {s.stream_port}" if s.stream_port else "")
        self.detail.setText(
            f"{where}   |   packets {s.total_packets}   ·   lost {s.lost_packets}   ·   bad {s.bad_packets}   ·   "
            f"armband counters: queue drops {s.fw_queue_drops}, late samples {s.fw_acq_overruns}, send failures "
            f"{s.fw_send_failures}, Wi-Fi reconnects {s.fw_wifi_reconnects}")
