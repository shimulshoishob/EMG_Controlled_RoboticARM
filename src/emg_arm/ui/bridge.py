"""Qt adapter for the armband links: runs connect/disconnect off the GUI thread and re-emits
the link's callbacks as Qt signals."""

from __future__ import annotations

import threading
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from ..armband.fake import FakeArmband
from ..armband.link import BaseLink, LinkState, UsbLink, WifiLink

VIRTUAL_KEY = "virtual-armband-demo-key-000000"


class ArmbandBridge(QObject):
    frames = pyqtSignal(object)          # ndarray (n, 11)
    state = pyqtSignal(str, str)         # LinkState value, message
    stats = pyqtSignal(object)           # LinkStats
    closed = pyqtSignal()                # a disconnect finished

    def __init__(self, parent=None):
        super().__init__(parent)
        self.link: Optional[BaseLink] = None
        self.virtual: Optional[FakeArmband] = None
        self._closing = False
        self.recorder = None             # session.RawRecorder or None
        self.mode = ""

    # ---------------------------------------------------------- state
    @property
    def active(self) -> bool:
        return self.link is not None

    @property
    def busy(self) -> bool:
        return self._closing

    def _emit_state(self, state: LinkState, message: str):
        self.state.emit(state.value, message)

    def _hook_packets(self, link: BaseLink):
        link.packet_log_hook = lambda pkt: self.recorder.add(pkt) if self.recorder else None

    def _common(self):
        return (self.frames.emit, self._emit_state, self.stats.emit)

    # -------------------------------------------------------- connect
    def connect_wifi(self, ip: str, key: str, device_id: str = ""):
        self._begin("wifi")
        self.link = WifiLink(ip, key, *self._common(), device_id=device_id)
        self._hook_packets(self.link)
        self.link.start()

    def connect_usb(self, port: str):
        self._begin("usb")
        self.link = UsbLink(port, *self._common())
        self._hook_packets(self.link)
        self.link.start()

    def connect_virtual(self):
        self._begin("virtual")
        try:
            self.virtual = FakeArmband(VIRTUAL_KEY, bind_ip="127.0.0.1", name="Virtual armband (demo)",
                                       device_id="000000VIRTUAL").start()
        except OSError as exc:
            self.mode = ""
            self.state.emit(LinkState.FAILED.value,
                            f"Could not start the virtual armband ({exc}). Is another copy of this app using "
                            "UDP port 5001?")
            return
        self.link = WifiLink("127.0.0.1", VIRTUAL_KEY, *self._common(), device_id="000000VIRTUAL")
        self._hook_packets(self.link)
        self.link.start()

    def _begin(self, mode: str):
        if self.link is not None:
            raise RuntimeError("already connected")
        self.mode = mode
        self._closing = False

    # ----------------------------------------------------- disconnect
    def disconnect(self):
        """Stop in the background (it talks to the network); ``closed`` fires when done."""
        link, virtual = self.link, self.virtual
        if link is None or self._closing:
            return
        self._closing = True

        def work():
            try:
                link.stop()
            finally:
                if virtual is not None:
                    virtual.stop()
                self.link = None
                self.virtual = None
                self.mode = ""
                self._closing = False
                self.state.emit(LinkState.IDLE.value, "Disconnected")
                self.closed.emit()

        threading.Thread(target=work, name="armband-disconnect", daemon=True).start()

    def shutdown(self):
        """Blocking stop for application exit."""
        link, virtual = self.link, self.virtual
        self.link = self.virtual = None
        if link is not None:
            try:
                link.stop()
            except Exception:
                pass
        if virtual is not None:
            virtual.stop()
