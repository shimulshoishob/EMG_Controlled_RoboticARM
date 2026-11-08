"""Live data links to the armband (Wi-Fi/UDP and USB/serial) with automatic recovery.  No Qt.

Both links share one contract: ``start()`` returns at once, then the link reports through three
callbacks (always from a background thread):

* ``on_frames(ndarray[n, 11])``  - 8 raw EMG counts + roll/pitch/yaw, in order, no duplicates
* ``on_state(LinkState, message)`` - what is happening, in words a user can act on
* ``on_stats(LinkStats)``        - once per second

The links never give up by themselves after a network problem; they back off and retry until
``stop()`` is called, except for errors that retrying cannot fix (wrong access key).
"""

from __future__ import annotations

import select
import socket
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from . import netinfo
from .control import (AuthFailed, ControlClient, ControlError, DeviceRejected, DeviceTimeout, NoRoute,
                      discover)
from .protocol import (CONTROL_PORT, STREAM_PORT, TOTAL_CHANNELS, UsbFrameReader, Packet, ProtocolError,
                       parse_packet)

KEEPALIVE_INTERVAL_S = 3.0        # firmware drops the stream after 15 s without an authorised command
STALL_AFTER_S = 1.5               # no data for this long -> STALLED
FIRST_DATA_WAIT_S = 4.0           # after an accepted START, data should arrive within this
USB_STALL_AFTER_S = 2.0


class LinkState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    STALLED = "stalled"             # connected but data stopped; recovering
    RECONNECTING = "reconnecting"   # lost the device; searching / retrying
    FAILED = "failed"               # needs the user (wrong key, port busy, ...)


@dataclass
class LinkStats:
    state: LinkState = LinkState.IDLE
    frames_per_s: float = 0.0
    packets_per_s: float = 0.0
    total_frames: int = 0
    total_packets: int = 0
    lost_packets: int = 0
    bad_packets: int = 0
    foreign_packets: int = 0
    resyncs: int = 0
    last_rx_age_s: float = float("inf")
    late_frames: int = 0
    # firmware's own counters (cumulative since its boot)
    fw_queue_drops: int = 0
    fw_acq_overruns: int = 0
    fw_send_failures: int = 0
    fw_wifi_reconnects: int = 0
    device_ip: str = ""
    stream_port: int = 0
    transport: str = ""

    @property
    def loss_percent(self) -> float:
        total = self.total_packets + self.lost_packets
        return 100.0 * self.lost_packets / total if total else 0.0


FramesCallback = Callable[[np.ndarray], None]
StateCallback = Callable[[LinkState, str], None]
StatsCallback = Callable[[LinkStats], None]


class BaseLink:
    transport = ""

    def __init__(self, on_frames: FramesCallback, on_state: StateCallback, on_stats: StatsCallback):
        self._on_frames, self._on_state, self._on_stats = on_frames, on_state, on_stats
        self.stats = LinkStats(transport=self.transport)
        self.state = LinkState.IDLE
        self.message = ""
        self.fatal: Optional[str] = None
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._last_rx = 0.0
        self._expected_seq: Optional[int] = None
        self._win_t = time.monotonic()
        self._win_frames = 0
        self._win_packets = 0
        self.packet_log_hook: Optional[Callable[[Packet], None]] = None

    # ------------------------------------------------------------- public
    def start(self):
        raise NotImplementedError

    def stop(self):
        self._stop.set()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(6)
        self._threads.clear()
        self._set_state(LinkState.IDLE, "Disconnected")

    @property
    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def data_age(self) -> float:
        return time.monotonic() - self._last_rx if self._last_rx else float("inf")

    def wait_for(self, state: LinkState, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.state == state:
                return True
            time.sleep(0.02)
        return self.state == state

    # ----------------------------------------------------------- internals
    def _set_state(self, state: LinkState, message: str = ""):
        changed = state != self.state or message != self.message
        self.state, self.message = state, message
        self.stats.state = state
        if changed:
            try:
                self._on_state(state, message)
            except Exception:
                pass

    def _spawn(self, target, name):
        t = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(t)
        t.start()

    def _ingest(self, packet: Packet):
        """Sequence bookkeeping + delivery. Returns False for duplicates / stale packets."""
        s = self.stats
        if self._expected_seq is not None:
            delta = (packet.sequence - self._expected_seq) & 0xFFFFFFFF
            if delta >= 0x80000000:                 # older than what we already have
                return False
            if delta > 2000:                        # firmware rebooted (sequence restarted)
                s.resyncs += 1
            else:
                s.lost_packets += delta
        self._expected_seq = (packet.sequence + 1) & 0xFFFFFFFF
        s.total_packets += 1
        s.total_frames += packet.n_frames
        s.late_frames += packet.late_frames
        if packet.version >= 2:
            s.fw_queue_drops, s.fw_acq_overruns = packet.queue_drops, packet.acq_overruns
            s.fw_send_failures, s.fw_wifi_reconnects = packet.send_failures, packet.wifi_reconnects
        self._last_rx = time.monotonic()
        self._win_frames += packet.n_frames
        self._win_packets += 1
        if self.packet_log_hook:
            try:
                self.packet_log_hook(packet)
            except Exception:
                pass
        try:
            self._on_frames(packet.frames)
        except Exception:
            pass
        return True

    def _tick_stats(self):
        now = time.monotonic()
        if now - self._win_t >= 1.0:
            span = now - self._win_t
            self.stats.frames_per_s = self._win_frames / span
            self.stats.packets_per_s = self._win_packets / span
            self._win_frames = self._win_packets = 0
            self._win_t = now
            self.stats.last_rx_age_s = self.data_age()
            try:
                self._on_stats(self.stats)
            except Exception:
                pass

    def _reset_stream_tracking(self):
        self._expected_seq = None


# =============================================================== Wi-Fi / UDP
class WifiLink(BaseLink):
    transport = "Wi-Fi"

    def __init__(self, device_ip: str, key: str, on_frames, on_state, on_stats, device_id: str = "",
                 stream_port: int = STREAM_PORT, control_port: int = CONTROL_PORT,
                 keepalive_interval_s: float = KEEPALIVE_INTERVAL_S, local_ip: Optional[str] = None):
        super().__init__(on_frames, on_state, on_stats)
        self.device_ip = device_ip
        self.device_id = device_id
        self.key = key
        self.preferred_port = stream_port
        self.control_port = control_port
        self.keepalive_interval_s = keepalive_interval_s
        self.local_ip_override = local_ip
        self.client = ControlClient(device_ip, key, port=control_port)
        self._sock: Optional[socket.socket] = None
        self.stats.device_ip = device_ip
        self.diagnosis = ""             # last specific advice ("firewall", "wrong network", ...)
        self.last_error_kind = ""       # "", "auth", "timeout", "route", "firewall", "port"

    # --------------------------------------------------------------- start
    def start(self):
        self._stop.clear()
        self._set_state(LinkState.CONNECTING, f"Connecting to {self.device_ip}…")
        try:
            self._open_stream_socket()
        except OSError as exc:
            self.last_error_kind = "port"
            self._fail(f"Could not open a UDP port for the data stream ({exc}). "
                       "Close other copies of this app and try again.")
            return
        self._spawn(self._receive_loop, "armband-rx")
        self._spawn(self._control_loop, "armband-control")

    def stop(self):
        super().stop()            # joins our threads first, so no late START can follow the STOP below
        # Tell the armband to stop sending; best effort and quick.
        try:
            self.client.timeout = 0.5
            self.client.stop_stream()
        except Exception:
            pass
        sock, self._sock = self._sock, None
        if sock:
            try:
                sock.close()
            except OSError:
                pass
        self.client.close()

    def _open_stream_socket(self):
        last: Optional[OSError] = None
        for port in range(self.preferred_port, self.preferred_port + 10):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
                except OSError:
                    pass
                sock.bind(("0.0.0.0", port))
            except OSError as exc:          # port taken (another instance): try the next one
                sock.close()
                last = exc
                continue
            self._sock = sock
            self.stats.stream_port = port
            return
        raise last or OSError("no free port")

    # --------------------------------------------------------- receive path
    def _receive_loop(self):
        while not self._stop.is_set():
            sock = self._sock
            if sock is None:
                return
            try:
                ready = select.select([sock], [], [], 0.2)[0]
                if ready:
                    data, addr = sock.recvfrom(2048)
                else:
                    self._tick_stats()
                    continue
            except (OSError, ValueError):
                if self._stop.is_set():
                    return
                time.sleep(0.05)
                continue
            if addr[0] != self.device_ip:
                self.stats.foreign_packets += 1
                continue
            try:
                packet = parse_packet(data)
            except ProtocolError:
                self.stats.bad_packets += 1
                continue
            self._ingest(packet)
            self._tick_stats()

    # --------------------------------------------------------- control path
    def _client_ip(self) -> str:
        return self.local_ip_override or netinfo.local_ip_for(self.device_ip)

    def _fail(self, message: str):
        self.fatal = message
        self._set_state(LinkState.FAILED, message)

    def _try_start(self) -> bool:
        """START the stream. Returns True when the armband accepted it."""
        self.client.start_stream(self._client_ip(), self.stats.stream_port)
        self._reset_stream_tracking()
        return True

    def _wait_data(self, seconds: float) -> bool:
        end = time.monotonic() + seconds
        mark = self.stats.total_packets
        while time.monotonic() < end and not self._stop.is_set():
            if self.stats.total_packets > mark:
                return True
            time.sleep(0.02)
        return self.stats.total_packets > mark

    def _relocate(self) -> bool:
        """The armband may have a new DHCP address: look it up again by device id."""
        if not self.device_id:
            return False
        try:
            report = discover(timeout=2.0, include_loopback=self.device_ip.startswith("127."), sweep=False)
        except Exception:
            return False
        for ip, info in report.devices.items():
            if info.device_id == self.device_id and ip != self.device_ip:
                self.device_ip = ip
                self.stats.device_ip = ip
                self.client.set_target(ip)
                return True
        return False

    def _control_loop(self):
        backoff = 1.0
        failures = 0
        while not self._stop.is_set():
            # ---------------- (re)start the stream
            try:
                self._try_start()
            except AuthFailed as exc:
                self.last_error_kind = "auth"
                return self._fail(
                    f"{exc} Open the firmware's secrets.h and copy DEVICE_ACCESS_KEY_VALUE into the app.")
            except DeviceRejected as exc:
                self.last_error_kind = "rejected"
                from .control import ERROR_TEXT
                return self._fail(ERROR_TEXT.get(exc.code, str(exc)))
            except (DeviceTimeout, NoRoute) as exc:
                failures += 1
                self.last_error_kind = "route" if isinstance(exc, NoRoute) else "timeout"
                state = LinkState.CONNECTING if failures == 1 and self.state == LinkState.CONNECTING \
                    else LinkState.RECONNECTING
                where = netinfo.same_subnet(self.device_ip)
                hint = "" if where else " This computer is on a different network than the armband."
                self._set_state(state, f"No answer from the armband at {self.device_ip}.{hint} "
                                       f"Retrying in {backoff:.0f} s…")
                if failures >= 2 and self._relocate():
                    self._set_state(LinkState.RECONNECTING, f"The armband moved to {self.device_ip}; reconnecting…")
                    backoff = 1.0
                    continue
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2, 8.0)
                continue

            # ---------------- START accepted: does data arrive?
            if not self._wait_data(FIRST_DATA_WAIT_S):
                failures += 1
                self.last_error_kind = "firewall"
                self.diagnosis = (
                    f"The armband accepted the connection but no data arrives on UDP port "
                    f"{self.stats.stream_port}. A firewall on this computer is probably blocking incoming "
                    "UDP (allow Python / this app in Windows Defender Firewall or the macOS firewall), "
                    "or a VPN / guest-Wi-Fi 'client isolation' is in the way.")
                self._set_state(LinkState.STALLED, self.diagnosis)
                if self._stop.wait(min(2.0 * failures, 8.0)):
                    return
                continue

            failures, backoff = 0, 1.0
            self.last_error_kind = ""
            self.diagnosis = ""
            self._set_state(LinkState.STREAMING, f"Streaming from {self.device_ip}")

            # ---------------- steady state: keepalive + watchdog
            next_ping = time.monotonic() + self.keepalive_interval_s
            ping_failures = 0
            while not self._stop.is_set():
                if self._stop.wait(0.1):
                    return
                now = time.monotonic()
                if self.data_age() > STALL_AFTER_S:
                    self._set_state(LinkState.STALLED, "Data stopped - restarting the stream…")
                    break                                   # -> START again
                if self.state == LinkState.STALLED:
                    self._set_state(LinkState.STREAMING, f"Streaming from {self.device_ip}")
                if now >= next_ping:
                    next_ping = now + self.keepalive_interval_s
                    try:
                        self.client.ping()
                        ping_failures = 0
                    except AuthFailed as exc:
                        return self._fail(str(exc))
                    except ControlError:
                        ping_failures += 1       # data is still flowing; the watchdog handles real loss
        # loop exits only on stop()

    # ------------------------------------------------------------ helpers
    def restart(self):
        """Ask the control loop to START again (used after changing network)."""
        self._last_rx = 0.0


# ===================================================================== USB
class UsbLink(BaseLink):
    transport = "USB"

    def __init__(self, port: str, on_frames, on_state, on_stats, baud: int = 921600):
        super().__init__(on_frames, on_state, on_stats)
        self.port = port
        self.baud = baud
        self.reader = UsbFrameReader()
        self._serial = None
        self.stats.device_ip = port
        self.log_lines: List[str] = []

    def start(self):
        self._stop.clear()
        self._set_state(LinkState.CONNECTING, f"Opening {self.port}…")
        self._spawn(self._run, "armband-usb")

    def stop(self):
        self._stop.set()
        ser = self._serial
        try:
            if ser is not None and ser.is_open:
                ser.write(b"USBSTOP\n")
                ser.flush()
        except Exception:
            pass
        super().stop()
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass

    def _open(self):
        import serial
        ser = serial.Serial()
        ser.port = self.port
        ser.baudrate = self.baud
        ser.timeout = 0.05
        ser.write_timeout = 1.0
        # Opening an ESP32-S3 USB port normally pulses DTR/RTS, which RESETS the board
        # (3 s of silence). Holding both low before opening avoids that on macOS/Linux/Windows.
        ser.dtr = False
        ser.rts = False
        ser.open()
        return ser

    def _run(self):
        import serial
        retry = 0
        while not self._stop.is_set():
            try:
                ser = self._open()
            except (serial.SerialException, OSError) as exc:
                retry += 1
                msg = str(exc)
                hint = ""
                if "denied" in msg.lower() or "busy" in msg.lower() or "in use" in msg.lower():
                    hint = " Another program (Arduino Serial Monitor?) may be using the port."
                self._set_state(LinkState.RECONNECTING if retry > 1 else LinkState.CONNECTING,
                                f"Cannot open {self.port}: {msg}.{hint} Retrying…")
                if self._stop.wait(1.5):
                    return
                continue
            retry = 0
            self._serial = ser
            self.reader = UsbFrameReader()
            self._reset_stream_tracking()
            try:
                self._session(ser)
            except (serial.SerialException, OSError) as exc:
                if self._stop.is_set():
                    return
                self._set_state(LinkState.RECONNECTING, f"USB connection lost ({exc}). Plug the cable in again…")
            finally:
                try:
                    ser.close()
                except Exception:
                    pass
            if self._stop.wait(1.0):
                return

    def _session(self, ser):
        ser.reset_input_buffer()
        started = time.monotonic()
        last_cmd = 0.0
        self._last_rx = 0.0
        self._set_state(LinkState.CONNECTING, f"Waiting for data on {self.port}…")
        while not self._stop.is_set():
            now = time.monotonic()
            # USBSTART repeats as keepalive (firmware ends the session after 5 s of silence).
            if now - last_cmd >= (0.5 if self.state != LinkState.STREAMING else 1.0):
                ser.write(b"USBSTART\n" if self.state != LinkState.STREAMING else b"USBPING\n")
                last_cmd = now
            waiting = ser.in_waiting
            chunk = ser.read(waiting if waiting else 1)
            if chunk:
                for packet in self.reader.feed(chunk):
                    self._ingest(packet)
                self.stats.bad_packets = self.reader.bad_frames
                self.log_lines += self.reader.take_text()[-20:]
                del self.log_lines[:-50]
            self._tick_stats()
            age = self.data_age()
            if self.stats.total_packets and age < USB_STALL_AFTER_S:
                if self.state != LinkState.STREAMING:
                    self._set_state(LinkState.STREAMING, f"Streaming over USB ({self.port})")
            elif self.state == LinkState.STREAMING:
                self._set_state(LinkState.STALLED, "USB data stopped - restarting the stream…")
                last_cmd = 0.0
                ser.write(b"USBSTOP\n")
            elif now - started > 10 and self.state == LinkState.CONNECTING:
                self._set_state(
                    LinkState.CONNECTING,
                    "The port opened but the armband sends nothing. Is this the armband's USB port "
                    "(not the UART port) and is the firmware EMG_IMU_WIFI_SETUP_V2 flashed?")
