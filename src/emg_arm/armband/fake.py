"""A software copy of the armband firmware (``emg-imu-2.0.0``), speaking the exact same protocol.

Used for the in-app **Virtual armband** (try the whole app with no hardware) and by the tests,
which can make it behave badly on purpose: lose replies, lose data, reboot, change its keepalive.

It reproduces the firmware's rules, including the awkward ones:
* a challenge belongs to one remote (IP, source port) and is single-use, 30 s lifetime, 4 slots;
* ``START`` locks the stream target, ``STOP``/keepalive timeout (15 s) unlocks it;
* ``PING`` does not re-lock a target after a reboot.

Run it stand-alone:  ``python -m emg_arm.armband.fake --key <key>``
"""

from __future__ import annotations

import argparse
import math
import os
import random
import select
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .protocol import (CONTROL_PORT, EMG_CHANNELS, FRAME_V2_DTYPE, FRAMES_PER_PACKET, MAGIC,
                       SAMPLE_RATE_HZ, V2_HEADER, V2_PACKET_SIZE, encode_usb_frame, sign)

FIRMWARE_VERSION = "emg-imu-2.0.0"
CHALLENGE_SLOTS = 4
CHALLENGE_TTL_S = 30.0


class SignalSource:
    """Plausible EMG + IMU: quiet rest, then bursts on one channel after another."""

    PHASES = [("REST", 3.0, ()), ("CH1+2", 2.5, (0, 1)), ("CH3+4", 2.5, (2, 3)),
              ("CH5+6", 2.5, (4, 5)), ("CH7+8", 2.5, (6, 7)), ("ALL", 2.5, tuple(range(8)))]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.frame_id = 0
        self.t0 = time.monotonic()
        self.cycle = sum(p[1] for p in self.PHASES)

    def phase_at(self, t: float) -> Tuple[str, Tuple[int, ...]]:
        t %= self.cycle
        for name, length, active in self.PHASES:
            if t < length:
                return name, active
            t -= length
        return "REST", ()

    def frames(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=FRAME_V2_DTYPE)
        now_us = int((time.monotonic() - self.t0) * 1e6)
        for i in range(n):
            t = self.frame_id / SAMPLE_RATE_HZ
            _name, active = self.phase_at(t)
            base = 1980 + 25 * np.arange(EMG_CHANNELS)
            emg = base + self.rng.normal(0, 6, EMG_CHANNELS)
            if active:
                carrier = 0.6 * math.sin(2 * math.pi * 55 * t) + 0.4 * math.sin(2 * math.pi * 110 * t + 1)
                envelope = 0.6 + 0.4 * math.sin(2 * math.pi * 0.7 * t)
                for ch in active:
                    emg[ch] += 220 * envelope * carrier * (1 + 0.15 * self.rng.standard_normal())
            out["frame_id"][i] = self.frame_id
            out["ts_us"][i] = max(0, now_us - (n - 1 - i) * (1_000_000 // SAMPLE_RATE_HZ))
            out["imu_id"][i] = self.frame_id // 2
            out["imu_ts_us"][i] = out["ts_us"][i]
            out["emg"][i] = np.clip(emg, 0, 4095).astype(np.uint16)
            out["rpy"][i] = (10 * math.sin(0.5 * t), 5 * math.sin(0.3 * t), (20 * t) % 360 - 180)
            out["imu_fresh"][i] = 1 if self.frame_id % 2 == 0 else 0
            self.frame_id += 1
        return out


def build_packet(seq: int, frames: np.ndarray, counters=(0, 0, 0, 0)) -> bytes:
    q, o, s, w = counters
    header = V2_HEADER.pack(MAGIC, 2, len(frames), FRAME_V2_DTYPE.itemsize, seq & 0xFFFFFFFF, q, o, s, w, 0)
    return header + frames.tobytes()


class FakeArmband:
    """Firmware emulation over UDP. ``start()`` runs it on a background thread."""

    def __init__(self, key: str, bind_ip: str = "127.0.0.1", control_port: int = CONTROL_PORT,
                 name: str = "BioWave-EMG-IMU-2026", device_id: str = "AABBCCDDEEFF",
                 keepalive_timeout_s: float = 15.0, seed: Optional[int] = None):
        self.key = key
        self.bind_ip = bind_ip
        self.control_port = control_port
        self.name = name
        self.device_id = device_id
        self.keepalive_timeout_s = keepalive_timeout_s
        self.source = SignalSource(seed)
        # fault injection
        self.reply_drop_probability = 0.0
        self.data_drop_probability = 0.0
        self.silent = False                       # ignore everything (device "off the network")
        # state
        self.target: Optional[Tuple[str, int]] = None
        self.last_authorized = 0.0
        self.sequence = 0
        self.packets_sent = 0
        self._slots: List[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._rng = random.Random(seed)
        self.log: List[str] = []

    # ------------------------------------------------------------- control
    def start(self) -> "FakeArmband":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((self.bind_ip if self.bind_ip != "127.0.0.1" else "127.0.0.1", self.control_port))
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fake-armband", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(2)
        if self._sock:
            self._sock.close()
            self._sock = None

    def reboot(self):
        """Power-cycle: forget the stream target and all challenges (what a real reset does)."""
        self.target = None
        self._slots.clear()
        self.sequence = 0
        self.source = SignalSource()

    @property
    def streaming(self) -> bool:
        return self.target is not None

    # ------------------------------------------------------------ protocol
    def _reply(self, addr, text: str):
        if self._rng.random() < self.reply_drop_probability:
            return
        try:
            self._sock.sendto(text.encode("utf-8"), addr)
        except OSError:
            pass

    def _hello(self) -> str:
        return "|".join(["HELLO", self.device_id, self.name, "STA", self.bind_ip, "1",
                         "1" if self.streaming else "0", FIRMWARE_VERSION, str(SAMPLE_RATE_HZ)])

    def _issue_challenge(self, addr) -> str:
        now = time.monotonic()
        slot = next((s for s in self._slots if s["addr"] == addr), None)
        if slot is None:
            self._slots = [s for s in self._slots if now - s["t"] <= CHALLENGE_TTL_S]
            if len(self._slots) >= CHALLENGE_SLOTS:
                self._slots.remove(min(self._slots, key=lambda s: s["t"]))
            slot = {"addr": addr}
            self._slots.append(slot)
        slot["challenge"] = "%016x" % self._rng.getrandbits(64)
        slot["t"] = now
        return slot["challenge"]

    def _verify(self, command: str, challenge: str, payload: str, auth: str, addr) -> bool:
        slot = next((s for s in self._slots if s["addr"] == addr), None)
        if slot is None or time.monotonic() - slot["t"] > CHALLENGE_TTL_S or slot["challenge"] != challenge:
            return False
        parts = [command, challenge] + ([payload] if payload else [])
        ok = sign(self.key, *parts) == auth
        self._slots.remove(slot)                      # single use
        if ok:
            self.last_authorized = time.monotonic()
        return ok

    def _handle(self, text: str, addr):
        if self.silent:
            return
        if text == "DISCOVER":
            return self._reply(addr, self._hello())
        if text == "CHALLENGE":
            return self._reply(addr, "CHALLENGE|" + self._issue_challenge(addr))
        parts = text.split("|")
        cmd = parts[0]
        if cmd == "START":
            if len(parts) != 5:
                return self._reply(addr, "ERR|BAD_FORMAT")
            _c, challenge, ip, port, auth = parts
            if not self._verify("START", challenge, f"{ip}|{port}", auth, addr):
                return self._reply(addr, "ERR|AUTH_FAILED")
            try:
                socket.inet_aton(ip)
            except OSError:
                return self._reply(addr, "ERR|BAD_IP")
            if not port.isdigit() or not 0 < int(port) < 65536:
                return self._reply(addr, "ERR|BAD_PORT")
            self.target = (ip, int(port))
            self.log.append(f"START {ip}:{port}")
            return self._reply(addr, "ACK|STARTED")
        if cmd in ("STOP", "PING"):
            if len(parts) != 3:
                return self._reply(addr, "ERR|BAD_FORMAT")
            if not self._verify(cmd, parts[1], "", parts[2], addr):
                return self._reply(addr, "ERR|AUTH_FAILED")
            if cmd == "STOP":
                self.target = None
                self.log.append("STOP")
                return self._reply(addr, "ACK|STOPPED")
            return self._reply(addr, "ACK|PONG")
        self._reply(addr, "ERR|UNKNOWN_COMMAND")

    # ------------------------------------------------------------- running
    def _run(self):
        assert self._sock is not None
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # data goes out from here
        next_packet = time.monotonic()
        period = FRAMES_PER_PACKET / SAMPLE_RATE_HZ
        try:
            while not self._stop.is_set():
                ready = select.select([self._sock], [], [], 0.002)[0]
                if ready:
                    try:
                        data, addr = self._sock.recvfrom(1024)
                        self._handle(data.decode("utf-8", errors="replace"), addr)
                    except (OSError, ConnectionResetError):
                        pass
                now = time.monotonic()
                if self.target and now - self.last_authorized > self.keepalive_timeout_s:
                    self.target = None
                    self.log.append("keepalive timeout")
                if now < next_packet:
                    continue
                if now - next_packet > 0.25:
                    next_packet = now
                while next_packet <= now:
                    next_packet += period
                    if self.target is None or self.silent:
                        continue
                    frames = self.source.frames(FRAMES_PER_PACKET)
                    packet = build_packet(self.sequence, frames)
                    self.sequence += 1
                    if self._rng.random() < self.data_drop_probability:
                        continue
                    try:
                        out.sendto(packet, self.target)
                        self.packets_sent += 1
                    except OSError:
                        pass
        finally:
            out.close()


class FakeUsbDevice:
    """Serial side of the firmware on a file descriptor (a pty in tests)."""

    def __init__(self, fd: int, seed: Optional[int] = None, boot_delay_s: float = 0.0):
        self.fd = fd
        self.source = SignalSource(seed)
        self.sequence = 0
        self.streaming = False
        self.csv = False
        self.last_cmd = 0.0
        self.boot_delay_s = boot_delay_s
        self.provisioned: Optional[Tuple[str, str]] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(2)

    def _write(self, data: bytes):
        try:
            os.write(self.fd, data)
        except OSError:
            pass

    def _line(self, line: str):
        if self.streaming and not self.csv:
            self._write(b"\x00")
        if line.startswith("INFO"):
            self._write(f"INFO|AABBCCDDEEFF|BioWave-EMG-IMU-2026|1|1|{FIRMWARE_VERSION}|{SAMPLE_RATE_HZ}\n".encode())
        elif line in ("USBSTART", "USBSTARTCSV", "USBPING"):
            if not self.streaming and line != "USBPING":
                self.csv = line == "USBSTARTCSV"
                self.streaming = True
            self.last_cmd = time.monotonic()
        elif line == "USBSTOP":
            self.streaming = False
        elif line.startswith("PROVISION|"):
            from urllib.parse import unquote_plus
            _p, ssid, pw = line.split("|")
            self.provisioned = (unquote_plus(ssid), unquote_plus(pw))
            self._write(b"ACK|PROVISIONED\n")
        else:
            self._write(b"ERR|UNKNOWN_COMMAND\n")

    def _run(self):
        time.sleep(self.boot_delay_s)
        buf = b""
        next_packet = time.monotonic()
        period = FRAMES_PER_PACKET / SAMPLE_RATE_HZ
        while not self._stop.is_set():
            if select.select([self.fd], [], [], 0.002)[0]:
                try:
                    buf += os.read(self.fd, 256)
                except OSError:
                    break
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._line(line.decode(errors="replace").strip())
            now = time.monotonic()
            if self.streaming and now - self.last_cmd > 5.0:
                self.streaming = False
            if now < next_packet:
                continue
            if now - next_packet > 0.25:
                next_packet = now
            while next_packet <= now:
                next_packet += period
                if self.streaming and not self.csv:
                    packet = build_packet(self.sequence, self.source.frames(FRAMES_PER_PACKET))
                    self.sequence += 1
                    self._write(encode_usb_frame(packet))


def main():
    parser = argparse.ArgumentParser(description="Software copy of the BioWave armband firmware.")
    parser.add_argument("--key", required=True, help="device access key (16+ characters)")
    parser.add_argument("--bind", default="0.0.0.0", help="address to listen on (default all interfaces)")
    parser.add_argument("--drop-replies", type=float, default=0.0, help="probability of losing a reply")
    parser.add_argument("--drop-data", type=float, default=0.0, help="probability of losing a data packet")
    args = parser.parse_args()
    device = FakeArmband(args.key, bind_ip=args.bind)
    device.reply_drop_probability = args.drop_replies
    device.data_drop_probability = args.drop_data
    device.start()
    print(f"Fake armband listening on {args.bind}:{CONTROL_PORT}; Ctrl+C to quit")
    try:
        while True:
            time.sleep(1)
            print(f"streaming={device.streaming} packets_sent={device.packets_sent}", end="\r")
    except KeyboardInterrupt:
        device.stop()


if __name__ == "__main__":
    main()
