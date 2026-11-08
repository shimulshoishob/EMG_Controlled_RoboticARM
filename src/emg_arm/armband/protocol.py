"""BioWave armband wire protocol (firmware ``emg-imu-2.x``).  Pure Python, no Qt.

Everything the firmware says or sends is described here once:

* **Data packets** - ``BWIM`` v2 (28-byte header + 5 x 56-byte frames = 308 bytes) over UDP,
  or the same bytes + CRC-16, COBS encoded and ``0x00`` terminated over USB serial.
  BWIM v1 (252 bytes) is still parsed so old firmware keeps working.
* **Control messages** - ``|`` separated text over UDP port 5001, authenticated with an
  HMAC-SHA256 of ``COMMAND|challenge[|payload]`` keyed with the device access key.
* **Serial commands** - ``INFO``, ``PROVISION|ssid|pass``, ``USBSTART``, ``USBPING``, ``USBSTOP``.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import struct
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ------------------------------------------------------------------ constants
STREAM_PORT = 5000
CONTROL_PORT = 5001
USB_BAUD = 921600
EMG_CHANNELS = 8
IMU_CHANNELS = 3
TOTAL_CHANNELS = EMG_CHANNELS + IMU_CHANNELS          # 8 EMG + roll, pitch, yaw
SAMPLE_RATE_HZ = 500
FRAMES_PER_PACKET = 5
ADC_MAX = 4095

MAGIC = b"BWIM"
V2_HEADER = struct.Struct("<4sBBHIIIIHH")             # 28 bytes
V1_HEADER = struct.Struct("<4sBBHI")                  # 12 bytes
FRAME_V2_DTYPE = np.dtype([
    ("frame_id", "<u4"), ("ts_us", "<u8"), ("imu_id", "<u4"), ("imu_ts_us", "<u8"),
    ("emg", "<u2", (EMG_CHANNELS,)), ("rpy", "<f4", (3,)),
    ("imu_fresh", "u1"), ("flags", "u1"), ("reserved", "<u2"),
])
FRAME_V1_DTYPE = np.dtype([
    ("frame_id", "<u4"), ("ts_us", "<u4"), ("imu_id", "<u4"), ("imu_ts_us", "<u4"),
    ("emg", "<u2", (EMG_CHANNELS,)), ("rpy", "<f4", (3,)),
    ("imu_fresh", "u1"), ("pad", "u1", (3,)),
])
assert V2_HEADER.size == 28 and FRAME_V2_DTYPE.itemsize == 56
assert V1_HEADER.size == 12 and FRAME_V1_DTYPE.itemsize == 48
V2_PACKET_SIZE = V2_HEADER.size + FRAMES_PER_PACKET * FRAME_V2_DTYPE.itemsize   # 308
V1_PACKET_SIZE = V1_HEADER.size + FRAMES_PER_PACKET * FRAME_V1_DTYPE.itemsize   # 252
FRAME_FLAG_LATE = 0x01


class ProtocolError(ValueError):
    """A packet or message does not follow the protocol."""


# --------------------------------------------------------------- data packets
@dataclass
class Packet:
    version: int
    sequence: int
    frames: np.ndarray                    # (n, 11) float32: emg1..emg8, roll, pitch, yaw
    frame_ids: np.ndarray                 # (n,) uint32
    timestamps_us: np.ndarray             # (n,) uint64 (v1: uint32 wrap)
    imu_fresh: np.ndarray                 # (n,) bool
    late_frames: int = 0                  # frames sampled after their deadline
    queue_drops: int = 0                  # cumulative counters from the firmware (v2 only)
    acq_overruns: int = 0
    send_failures: int = 0
    wifi_reconnects: int = 0

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


def parse_packet(data: bytes) -> Packet:
    """Parse one BWIM datagram (v1 or v2). Raises ProtocolError on anything else."""
    if len(data) < V1_HEADER.size or data[:4] != MAGIC:
        raise ProtocolError("not a BWIM packet")
    version = data[4]
    if version == 2:
        if len(data) < V2_HEADER.size:
            raise ProtocolError("truncated v2 header")
        (_m, _v, count, frame_size, seq, q_drops, overruns, send_fail, wifi_re,
         _flags) = V2_HEADER.unpack_from(data, 0)
        dtype, offset = FRAME_V2_DTYPE, V2_HEADER.size
    elif version == 1:
        _m, _v, count, frame_size, seq = V1_HEADER.unpack_from(data, 0)
        q_drops = overruns = send_fail = wifi_re = 0
        dtype, offset = FRAME_V1_DTYPE, V1_HEADER.size
    else:
        raise ProtocolError(f"unsupported BWIM version {version}")
    if frame_size != dtype.itemsize:
        raise ProtocolError(f"unexpected frame size {frame_size}")
    if count == 0 or count > 32:
        raise ProtocolError(f"unreasonable frame count {count}")
    if len(data) != offset + count * frame_size:
        raise ProtocolError(f"length {len(data)} does not match {count} frames")
    raw = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
    frames = np.empty((count, TOTAL_CHANNELS), dtype=np.float32)
    frames[:, :EMG_CHANNELS] = raw["emg"]
    frames[:, EMG_CHANNELS:] = raw["rpy"]
    late = int(np.count_nonzero(raw["flags"] & FRAME_FLAG_LATE)) if version == 2 else 0
    return Packet(version, seq, frames, raw["frame_id"].copy(), raw["ts_us"].copy(),
                  raw["imu_fresh"].astype(bool), late, q_drops, overruns, send_fail, wifi_re)


# ------------------------------------------------------------ USB framing
def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) - same as the firmware."""
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def cobs_encode(data: bytes) -> bytes:
    out = bytearray([0])
    code_index, code = 0, 1
    for byte in data:
        if byte == 0:
            out[code_index] = code
            code_index, code = len(out), 1
            out.append(0)
        else:
            out.append(byte)
            code += 1
            if code == 0xFF:
                out[code_index] = code
                code_index, code = len(out), 1
                out.append(0)
    out[code_index] = code
    return bytes(out)


def cobs_decode(data: bytes) -> bytes:
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        code = data[i]
        if code == 0:
            raise ProtocolError("zero byte inside COBS block")
        end = i + code
        if end > n:
            raise ProtocolError("COBS block runs past the end")
        out += data[i + 1:end]
        i = end
        if code != 0xFF and i < n:
            out.append(0)
    return bytes(out)


def encode_usb_frame(packet_bytes: bytes) -> bytes:
    """What the firmware writes for one packet (used by the fake armband and tests)."""
    crc = crc16_ccitt(packet_bytes)
    return cobs_encode(packet_bytes + struct.pack("<H", crc)) + b"\x00"


def decode_usb_frame(encoded: bytes) -> bytes:
    """COBS-decode and CRC-check one 0x00-delimited chunk. Returns the BWIM packet bytes."""
    raw = cobs_decode(encoded)
    if len(raw) < 3:
        raise ProtocolError("frame too short")
    body, (crc,) = raw[:-2], struct.unpack("<H", raw[-2:])
    if crc16_ccitt(body) != crc:
        raise ProtocolError("CRC mismatch")
    return body


class UsbFrameReader:
    """Splits the serial byte stream into frames and text lines.

    The firmware interleaves binary frames (no 0x00 inside, 0x00 terminated) with the
    occasional text reply (``INFO|...``) and boot log lines. A chunk that is not a valid
    frame is kept as text if it looks printable, otherwise counted as a bad frame.
    """

    MAX_CHUNK = 1024

    def __init__(self):
        self._buf = bytearray()
        self.bad_frames = 0
        self.good_frames = 0
        self.text_lines: List[str] = []

    def feed(self, data: bytes) -> List[Packet]:
        packets: List[Packet] = []
        self._buf += data
        while True:
            end = self._buf.find(0)
            if end < 0:
                if len(self._buf) > self.MAX_CHUNK:      # garbage with no delimiter
                    del self._buf[:-self.MAX_CHUNK // 2]
                    self.bad_frames += 1
                break
            chunk = bytes(self._buf[:end])
            del self._buf[:end + 1]
            if not chunk:
                continue
            try:
                packets.append(parse_packet(decode_usb_frame(chunk)))
                self.good_frames += 1
                continue
            except ProtocolError:
                pass
            # A text line may be glued in front of a frame (the firmware only breaks the stream
            # before its own replies): peel printable lines off the front, then retry the rest.
            if self._peel_text_then_frame(chunk, packets):
                continue
            if re.fullmatch(rb"[\x09\x0a\x0d\x20-\x7e]+", chunk):
                self.text_lines.extend(t.strip() for t in chunk.decode("ascii").splitlines() if t.strip())
            else:
                self.bad_frames += 1
        return packets

    _PRINTABLE = re.compile(rb"[\x09\x0a\x0d\x20-\x7e]*")

    def _peel_text_then_frame(self, chunk: bytes, packets: List[Packet]) -> bool:
        pos = chunk.find(b"\n")
        while pos >= 0 and pos + 1 < len(chunk):
            head, tail = chunk[:pos + 1], chunk[pos + 1:]
            if not self._PRINTABLE.fullmatch(head):
                return False
            try:
                packets.append(parse_packet(decode_usb_frame(tail)))
            except ProtocolError:
                pos = chunk.find(b"\n", pos + 1)
                continue
            self.good_frames += 1
            self.text_lines.extend(t.strip() for t in head.decode("ascii").splitlines() if t.strip())
            return True
        return False

    def take_text(self) -> List[str]:
        lines, self.text_lines = self.text_lines, []
        return lines


# ------------------------------------------------------------ control messages
@dataclass
class DeviceInfo:
    """What a device says about itself (UDP ``HELLO`` or serial ``INFO``)."""
    device_id: str
    name: str
    imu_ready: bool
    firmware: str
    sample_rate: int = SAMPLE_RATE_HZ
    ip: str = ""                       # address we reached it at (UDP) or ""
    wifi_mode: str = ""
    streaming: bool = False            # already locked to a client
    wifi_saved: Optional[bool] = None  # serial INFO only
    port: str = ""                     # serial port name (serial INFO only)
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        where = self.ip or self.port or "?"
        return f"{self.name}  ·  {where}"


def parse_hello(message: str, source_ip: str) -> DeviceInfo:
    """``HELLO|id|name|mode|ip|imu|locked|fw[|rate]`` (v1 firmware has no rate)."""
    parts = message.strip().split("|")
    if parts[0] != "HELLO" or len(parts) < 8:
        raise ProtocolError(f"not a HELLO message: {message[:60]!r}")
    rate = int(parts[8]) if len(parts) > 8 and parts[8].isdigit() else SAMPLE_RATE_HZ
    return DeviceInfo(device_id=parts[1], name=parts[2], wifi_mode=parts[3], ip=source_ip,
                      imu_ready=parts[5] == "1", streaming=parts[6] == "1", firmware=parts[7],
                      sample_rate=rate, extra={"reported_ip": parts[4]})


def parse_info(line: str, port: str = "") -> DeviceInfo:
    """``INFO|id|name|imu|wifiSaved|fw[|rate]``."""
    parts = line.strip().split("|")
    if parts[0] != "INFO" or len(parts) < 6:
        raise ProtocolError(f"not an INFO line: {line[:60]!r}")
    rate = int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else SAMPLE_RATE_HZ
    return DeviceInfo(device_id=parts[1], name=parts[2], imu_ready=parts[3] == "1",
                      wifi_saved=parts[4] == "1", firmware=parts[5], sample_rate=rate, port=port)


def sign(key: str, *parts) -> str:
    """HMAC-SHA256 (hex) over ``part|part|...`` keyed with the device access key."""
    message = "|".join(str(p) for p in parts)
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def build_authed(command: str, challenge: str, key: str, *payload) -> str:
    return "|".join([command, challenge, *map(str, payload), sign(key, command, challenge, *payload)])


# ------------------------------------------------------------ serial helpers
def quote_for_firmware(text: str) -> str:
    """Percent-encode only what the firmware's decoder needs (``% | + CR LF`` and non-ASCII).

    The firmware drops serial lines longer than 255 bytes, so ``urllib.parse.quote(safe='')``
    (which escapes every symbol, ~3x growth) can overflow with a long password.
    """
    out = []
    for byte in text.encode("utf-8"):
        ch = chr(byte)
        if byte < 0x20 or byte >= 0x7F or ch in "%|+":
            out.append(f"%{byte:02X}")
        else:
            out.append(ch)
    return "".join(out)


MAX_SERIAL_LINE = 250
MAX_SSID_BYTES = 32
MAX_PASSWORD_BYTES = 63


def build_provision_line(ssid: str, password: str) -> str:
    if not ssid:
        raise ProtocolError("The Wi-Fi name (SSID) is empty.")
    if len(ssid.encode("utf-8")) > MAX_SSID_BYTES:
        raise ProtocolError(f"A Wi-Fi name can be at most {MAX_SSID_BYTES} bytes.")
    if password and not 8 <= len(password.encode("utf-8")) <= MAX_PASSWORD_BYTES:
        raise ProtocolError("A Wi-Fi password must be 8 to 63 characters (or empty for an open network).")
    line = f"PROVISION|{quote_for_firmware(ssid)}|{quote_for_firmware(password)}"
    if len(line) > MAX_SERIAL_LINE:
        raise ProtocolError("The Wi-Fi name and password are too long to send to the armband.")
    return line


_KEY_RE = re.compile(r'#\s*define\s+DEVICE_ACCESS_KEY_VALUE\s+"([^"]*)"')
PLACEHOLDER_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_KEY"


def read_key_from_secrets_h(text: str) -> str:
    """Extract the access key from the text of the firmware's ``secrets.h``."""
    match = _KEY_RE.search(text)
    if not match:
        raise ProtocolError("No DEVICE_ACCESS_KEY_VALUE found in that file.")
    return match.group(1)


def check_key(key: str) -> Optional[str]:
    """Return a human-readable problem with the access key, or None if it looks usable."""
    key = key.strip()
    if not key:
        return "Enter the device access key (the value in the firmware's secrets.h)."
    if key == PLACEHOLDER_KEY:
        return "That is the placeholder key. Use the key from the secrets.h you flashed."
    if len(key) < 16:
        return "The firmware requires a key of at least 16 characters."
    if key != key.strip() or any(c.isspace() for c in key):
        return "The key must not contain spaces."
    return None
