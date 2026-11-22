"""UDP control channel and device discovery for the armband.  Pure Python, no Qt.

Why this replaces the old ``ControlProtocol``:

* The firmware remembers a challenge per remote **(IP, source port)**. The old code opened a new
  socket for ``CHALLENGE`` and another for ``START``, so the second one came from a different
  source port and the firmware always answered ``ERR|AUTH_FAILED``.  ``ControlClient`` keeps ONE
  socket for its whole life and runs each authenticated command on it.
* UDP replies get lost (about 1 in 5 on busy Wi-Fi).  Every transaction is retried from the start
  with a fresh challenge, and stale datagrams are discarded before each attempt.
* Discovery enumerates the real interfaces and broadcasts on each one (directed broadcast, bound
  to that adapter), repeats the request, and falls back to a unicast sweep of the local /24 when
  the router drops broadcasts.
"""

from __future__ import annotations

import ipaddress
import select
import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

from . import netinfo
from .protocol import (CONTROL_PORT, DeviceInfo, ProtocolError, build_authed, parse_hello)


class ControlError(RuntimeError):
    """Base class: ``str(exc)`` is a plain-language message that can be shown to the user."""


class DeviceTimeout(ControlError):
    pass


class AuthFailed(ControlError):
    pass


class DeviceRejected(ControlError):
    def __init__(self, code: str):
        super().__init__(f"The armband refused the command ({code}).")
        self.code = code


class NoRoute(ControlError):
    pass


ERROR_TEXT = {
    "AUTH_FAILED": "The access key does not match the one flashed into the armband.",
    "BAD_FORMAT": "The armband could not read the command (firmware/app version mismatch?).",
    "BAD_IP": "The armband rejected this computer's address.",
    "BAD_PORT": "The armband rejected the data port.",
    "UNKNOWN_COMMAND": "The armband does not know this command (old firmware?).",
}


class ControlClient:
    """Talks to one armband. Thread-safe; keeps a single UDP socket (see module docstring)."""

    def __init__(self, device_ip: str, key: str, port: int = CONTROL_PORT,
                 timeout: float = 1.0, attempts: int = 4):
        self.device_ip = device_ip
        self.key = key
        self.port = port
        self.timeout = timeout
        self.attempts = attempts
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------ socket
    def _socket(self) -> socket.socket:
        if self._sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", 0))
            self._sock = sock
        return self._sock

    def close(self):
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                finally:
                    self._sock = None

    def set_target(self, device_ip: str):
        """The armband got a new DHCP address: keep the client, change where it points."""
        with self._lock:
            self.device_ip = device_ip

    def _drain(self, sock: socket.socket):
        while select.select([sock], [], [], 0)[0]:
            try:
                sock.recvfrom(2048)
            except OSError:
                break

    def _send_wait(self, sock: socket.socket, message: str, prefixes: Sequence[str], timeout: float) -> str:
        """Send one datagram, return the first reply from the device whose text starts with a prefix."""
        try:
            sock.sendto(message.encode("utf-8"), (self.device_ip, self.port))
        except OSError as exc:
            raise NoRoute(f"This computer cannot reach {self.device_ip} ({exc.strerror or exc}). "
                          "Is it on the same Wi-Fi network (and is any VPN off)?") from exc
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise socket.timeout()
            ready = select.select([sock], [], [], left)[0]
            if not ready:
                raise socket.timeout()
            try:
                data, addr = sock.recvfrom(2048)
            except ConnectionResetError:        # Windows reports ICMP "port unreachable" this way
                continue
            except OSError as exc:
                raise NoRoute(str(exc)) from exc
            if addr[0] != self.device_ip:
                continue
            text = data.decode("utf-8", errors="replace").strip()
            if any(text.startswith(p) for p in prefixes):
                return text

    # ------------------------------------------------------- transactions
    def authed(self, command: str, *payload, attempts: Optional[int] = None) -> List[str]:
        """CHALLENGE + signed ``command`` on the same socket. Returns the fields after ``ACK``."""
        if not self.key:
            raise AuthFailed("No access key is set.")
        attempts = attempts or self.attempts
        last: Exception = DeviceTimeout("")
        with self._lock:
            sock = self._socket()
            for attempt in range(attempts):
                self._drain(sock)
                try:
                    reply = self._send_wait(sock, "CHALLENGE", ("CHALLENGE|",), self.timeout)
                    challenge = reply.split("|", 1)[1]
                    reply = self._send_wait(sock, build_authed(command, challenge, self.key, *payload),
                                            ("ACK|", "ERR|"), self.timeout)
                except socket.timeout:
                    last = DeviceTimeout(
                        f"No answer from the armband at {self.device_ip} (UDP port {self.port}).")
                    continue
                parts = reply.split("|")
                if parts[0] == "ACK":
                    return parts[1:]
                code = parts[1] if len(parts) > 1 else "UNKNOWN"
                if code == "AUTH_FAILED":
                    # Either the key is wrong, or the challenge expired/was evicted. Retry once more
                    # with a fresh challenge before blaming the key.
                    last = AuthFailed(ERROR_TEXT["AUTH_FAILED"])
                    if attempt >= 1:
                        raise last
                    continue
                raise DeviceRejected(code)
        raise last

    def start_stream(self, client_ip: str, client_port: int) -> List[str]:
        return self.authed("START", client_ip, client_port)

    def stop_stream(self) -> List[str]:
        return self.authed("STOP", attempts=2)

    def ping(self) -> List[str]:
        return self.authed("PING", attempts=2)

    def hello(self, attempts: int = 3) -> DeviceInfo:
        """Unauthenticated ``DISCOVER`` straight to this device (is it there at all?)."""
        with self._lock:
            sock = self._socket()
            for _ in range(attempts):
                self._drain(sock)
                try:
                    reply = self._send_wait(sock, "DISCOVER", ("HELLO|",), self.timeout)
                    return parse_hello(reply, self.device_ip)
                except (socket.timeout, ProtocolError):
                    continue
        raise DeviceTimeout(f"No answer from {self.device_ip}.")


# ------------------------------------------------------------------ discovery
class DiscoveryReport:
    """Result of a scan, with enough detail to explain an empty result to the user."""

    def __init__(self):
        self.devices: Dict[str, DeviceInfo] = {}
        self.interfaces: List[netinfo.Interface] = []
        self.sent_broadcast = False
        self.sent_sweep = False
        self.errors: List[str] = []

    @property
    def found(self) -> List[DeviceInfo]:
        return list(self.devices.values())


def _open_scan_socket(bind_ip: str) -> Optional[socket.socket]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((bind_ip, 0))
        sock.setblocking(False)
        return sock
    except OSError:
        return None


def discover(timeout: float = 2.5, port: int = CONTROL_PORT, include_loopback: bool = True,
             sweep: bool = True, progress: Optional[Callable[[str], None]] = None,
             extra_targets: Sequence[str] = ()) -> DiscoveryReport:
    """Find armbands on this computer's networks.

    1. Broadcast ``DISCOVER`` on every real interface (3 rounds, so one lost datagram is harmless).
    2. If nothing answered and ``sweep`` is set, send ``DISCOVER`` to every address of each
       local /24 (some routers filter broadcast but pass unicast).
    """
    report = DiscoveryReport()
    report.interfaces = netinfo.list_interfaces(include_loopback=False)
    say = progress or (lambda _m: None)

    sockets: Dict[socket.socket, str] = {}
    for iface in report.interfaces:
        sock = _open_scan_socket(iface.ip)
        if sock:
            sockets[sock] = iface.ip
        else:
            report.errors.append(f"Could not open a socket on {iface.name} ({iface.ip}).")
    wildcard = _open_scan_socket("0.0.0.0")
    if wildcard:
        sockets[wildcard] = "0.0.0.0"
    loop = _open_scan_socket("127.0.0.1") if include_loopback else None
    if loop:
        sockets[loop] = "127.0.0.1"
    if not sockets:
        report.errors.append("No usable network interface (is Wi-Fi switched on?).")
        return report

    def send_round(unicast_targets: Sequence[str] = ()):
        payload = b"DISCOVER"
        for sock, bind_ip in list(sockets.items()):
            targets: List[str] = []
            if bind_ip == "127.0.0.1":
                targets = ["127.0.0.1"]
            elif bind_ip == "0.0.0.0":
                targets = ["255.255.255.255"]
            else:
                iface = next((i for i in report.interfaces if i.ip == bind_ip), None)
                targets = ["255.255.255.255"] + ([iface.broadcast] if iface else [])
            for target in targets + list(unicast_targets):
                try:
                    sock.sendto(payload, (target, port))
                    if target != "127.0.0.1":
                        report.sent_broadcast = report.sent_broadcast or not unicast_targets
                except OSError:
                    pass

    def collect(seconds: float):
        end = time.monotonic() + seconds
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            ready = select.select(list(sockets), [], [], min(left, 0.1))[0]
            for sock in ready:
                try:
                    data, addr = sock.recvfrom(2048)
                except (BlockingIOError, ConnectionResetError, OSError):
                    continue
                try:
                    info = parse_hello(data.decode("utf-8", errors="replace"), addr[0])
                except ProtocolError:
                    continue
                if addr[0] not in report.devices:
                    report.devices[addr[0]] = info
                    say(f"Found {info.name} at {addr[0]}")

    try:
        say("Looking for armbands on " + (", ".join(i.name for i in report.interfaces) or "this computer") + "…")
        per_round = timeout / 3.0
        for _ in range(3):
            send_round(extra_targets)
            collect(per_round)
            if report.devices:
                collect(0.3)
                break
        if not report.devices and sweep:
            hosts: List[str] = []
            for iface in report.interfaces:
                if iface.is_virtual:
                    continue
                net = iface.network
                if net.prefixlen >= 22:                      # up to 1022 hosts; skip huge networks
                    hosts += [str(h) for h in net.hosts() if str(h) != iface.ip]
            if hosts:
                say(f"No broadcast reply - trying each of {len(hosts)} addresses one by one…")
                report.sent_sweep = True
                for rnd in range(2):
                    for i in range(0, len(hosts), 64):
                        for sock, bind_ip in sockets.items():
                            if bind_ip in ("127.0.0.1", "0.0.0.0"):
                                continue
                            for host in hosts[i:i + 64]:
                                try:
                                    sock.sendto(b"DISCOVER", (host, port))
                                except OSError:
                                    pass
                        collect(0.05)
                    collect(0.6)
                    if report.devices:
                        break
    finally:
        for sock in sockets:
            try:
                sock.close()
            except OSError:
                pass
    return report
