"""USB serial helpers: list ports, probe the armband (``INFO``), send Wi-Fi credentials (``PROVISION``).

Improvements over the old probe:
* DTR/RTS are held low *before* the port opens, so opening it does not reset the ESP32 on
  most boards; if it does reset (some USB-UART bridges), the probe simply keeps asking until
  the firmware has booted instead of sleeping a fixed 3.5 s and hoping.
* ``USBSTOP`` is sent first, so a board that was left streaming binary data still answers.
* Replies are located inside the byte stream, so a binary frame glued in front of the text
  line cannot hide it.
* Passwords are encoded minimally so the 255-byte firmware line limit is not exceeded.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .protocol import (USB_BAUD, DeviceInfo, ProtocolError, build_provision_line, parse_info)

# USB vendor ids of the chips found on ESP32 boards.
ESPRESSIF = 0x303A
BRIDGES = {0x10C4: "Silicon Labs CP210x", 0x1A86: "WCH CH340/CH9102", 0x0403: "FTDI"}


class SerialSetupError(RuntimeError):
    """Message is written for the user."""


@dataclass
class PortInfo:
    device: str
    description: str
    vid: Optional[int] = None
    pid: Optional[int] = None
    likely_esp32: bool = False

    @property
    def label(self) -> str:
        star = "★ " if self.likely_esp32 else ""
        return f"{star}{self.device} - {self.description}"


def list_ports() -> List[PortInfo]:
    """Serial ports, most likely armband first. Hides macOS ``tty.*`` twins and system ports."""
    import serial.tools.list_ports
    ports: List[PortInfo] = []
    for p in serial.tools.list_ports.comports():
        dev = p.device
        if sys.platform == "darwin":
            if dev.startswith("/dev/tty.") or "Bluetooth" in dev or "debug-console" in dev:
                continue
        desc = p.description if p.description and p.description != "n/a" else "Serial port"
        likely = p.vid == ESPRESSIF or p.vid in BRIDGES
        if p.vid == ESPRESSIF:
            desc += "  (ESP32-S3 native USB)"
        elif p.vid in BRIDGES:
            desc += f"  ({BRIDGES[p.vid]})"
        ports.append(PortInfo(dev, desc, p.vid, p.pid, likely))
    ports.sort(key=lambda x: (not x.likely_esp32, x.device))
    return ports


def _open(port: str, baud: int = USB_BAUD):
    import serial
    ser = serial.Serial()
    ser.port, ser.baudrate = port, baud
    ser.timeout, ser.write_timeout = 0.2, 1.0
    ser.dtr = False
    ser.rts = False
    try:
        ser.open()
    except (serial.SerialException, OSError) as exc:
        text = str(exc)
        if "denied" in text.lower() or "busy" in text.lower() or "in use" in text.lower():
            raise SerialSetupError(f"{port} is busy. Close the Arduino IDE Serial Monitor (or any other "
                                   "program using the armband) and try again.") from exc
        if "no such file" in text.lower() or "could not open" in text.lower() or "filenotfound" in text.lower():
            raise SerialSetupError(f"{port} is not available. Re-plug the USB cable and press Refresh.") from exc
        raise SerialSetupError(f"Could not open {port}: {exc}") from exc
    return ser


def _converse(port: str, command: str, pattern: str, timeout: float, resend_every: float = 0.8,
              progress: Optional[Callable[[str], None]] = None, baud: int = USB_BAUD) -> str:
    """Send ``command`` repeatedly until a reply line matching ``pattern`` arrives."""
    import serial
    rx = re.compile(pattern.encode())
    ser = _open(port, baud)
    try:
        ser.write(b"USBSTOP\n")
        end = time.monotonic() + timeout
        last_send = 0.0
        buf = b""
        while time.monotonic() < end:
            now = time.monotonic()
            if now - last_send >= resend_every:
                ser.write(command.encode("utf-8") + b"\n")
                ser.flush()
                last_send = now
                if progress and last_send - (end - timeout) > 2.5:
                    progress("Waiting for the armband to answer… (it may still be starting up)")
            try:
                buf += ser.read(ser.in_waiting or 1)
            except (serial.SerialException, OSError) as exc:
                raise SerialSetupError(f"The USB connection dropped ({exc}). Re-plug the cable.") from exc
            buf = buf[-4096:]
            match = rx.search(buf)
            if match:
                line_end = buf.find(b"\n", match.start())
                if line_end < 0:
                    continue                      # line not complete yet
                return buf[match.start():line_end].decode("utf-8", errors="replace").strip()
        raise SerialSetupError(
            "The armband did not answer. Check that the firmware 'EMG_IMU_WIFI_SETUP_V2' is flashed, that "
            "this is the armband's USB port, and that the Arduino Serial Monitor is closed.")
    finally:
        ser.close()


def probe(port: str, timeout: float = 15.0, progress: Optional[Callable[[str], None]] = None,
          baud: int = USB_BAUD) -> DeviceInfo:
    line = _converse(port, "INFO", r"INFO\|[^\r\n]*", timeout, progress=progress, baud=baud)
    try:
        return parse_info(line, port)
    except ProtocolError as exc:
        raise SerialSetupError(f"Unexpected answer from the armband: {line[:80]!r}") from exc


def provision(port: str, ssid: str, password: str, timeout: float = 10.0, baud: int = USB_BAUD) -> None:
    """Store Wi-Fi credentials on the armband. It restarts afterwards and joins that network."""
    try:
        line = build_provision_line(ssid, password)
    except ProtocolError as exc:
        raise SerialSetupError(str(exc)) from exc
    reply = _converse(port, line, r"(ACK\|PROVISIONED|ERR\|[A-Z_]+)", timeout, resend_every=3.0, baud=baud)
    if reply.startswith("ACK"):
        return
    code = reply.split("|", 1)[1] if "|" in reply else reply
    raise SerialSetupError({
        "MISSING_SSID": "The armband did not receive a Wi-Fi name.",
        "BAD_FORMAT": "The armband could not read the Wi-Fi details. Avoid the | character in the name or "
                      "password, or use a shorter password."}.get(code, f"The armband refused the Wi-Fi details ({code})."))
