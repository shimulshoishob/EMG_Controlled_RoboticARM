"""Qt adapter for the armband links: runs connect/disconnect off the GUI thread and re-emits
the link's callbacks as Qt signals."""

from __future__ import annotations

import threading
import time
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from ..armband.link import BaseLink, LinkState, UsbLink, WifiLink


class ArmbandBridge(QObject):
    frames = pyqtSignal(object)          # ndarray (n, 11)
    stamped = pyqtSignal(object, float)  # the same frames with their arrival time (time.perf_counter)
    state = pyqtSignal(str, str)         # LinkState value, message
    stats = pyqtSignal(object)           # LinkStats
    closed = pyqtSignal()                # a disconnect finished

    def __init__(self, parent=None):
        super().__init__(parent)
        self.link: Optional[BaseLink] = None
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

    def _emit_frames(self, frames):
        arrived = time.perf_counter()    # taken in the link thread, before any waiting for the GUI
        self.frames.emit(frames)
        self.stamped.emit(frames, arrived)

    def _common(self):
        return (self._emit_frames, self._emit_state, self.stats.emit)

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

    def _begin(self, mode: str):
        if self.link is not None:
            raise RuntimeError("already connected")
        self.mode = mode
        self._closing = False

    # ----------------------------------------------------- disconnect
    def disconnect(self):
        """Stop in the background (it talks to the network); ``closed`` fires when done."""
        link = self.link
        if link is None or self._closing:
            return
        self._closing = True

        def work():
            try:
                link.stop()
            finally:
                self.link = None
                self.mode = ""
                self._closing = False
                self.state.emit(LinkState.IDLE.value, "Disconnected")
                self.closed.emit()

        threading.Thread(target=work, name="armband-disconnect", daemon=True).start()

    def shutdown(self):
        """Blocking stop for application exit."""
        link, self.link = self.link, None
        if link is not None:
            try:
                link.stop()
            except Exception:
                pass
