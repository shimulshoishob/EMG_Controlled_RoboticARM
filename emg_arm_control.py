#!/usr/bin/env python3
"""
BioWave EMG armband -> Hiwonder xArm 1S controller (single file, browser UI).

Armband links (pick one in the UI; they are the same links BioWave main.py offers):
  * BioWave app   - listens to the gesture predictions main.py already publishes on
                    UDP 127.0.0.1:5005 (easiest: no device setup here, uses your trained model)
  * BioWave Wi-Fi - talks to the ESP32-S3 directly (DISCOVER / signed START / keep-alive, UDP 5000)
  * BioWave USB   - reads the ESP32-S3 text stream over USB serial (921600 baud)
  * Myo / Simulated
Arm: xArm over USB-HID (auto-detected) or a COM port, or simulated.

    gestures or EMG envelope -> decoder -> joint goals -> rate limiter -> xArm servos

Run:   python emg_arm_control.py          (UI at http://127.0.0.1:8765)
Only the standard library is required; see requirements.txt for the hardware packages.
"""
import argparse
import collections
import csv
import hashlib
import hmac
import json
import math
import os
import random
import socket
import struct
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# CONFIG  (edit here)
# =============================================================================
# id = servo id on the xArm controller. "+" moves the servo value up unless inverted
# (also a "flip" checkbox per joint in the UI).
JOINTS = [
    dict(id=6, name="Base",    min=100, max=850, home=500, invert=False),
    dict(id=5, name="Reach",   min=275, max=700, home=700, invert=False),
    dict(id=4, name="Elbow",   min=560, max=950, home=820, invert=False),
    dict(id=2, name="Wrist",   min=200, max=800, home=500, invert=False),
    dict(id=1, name="Gripper", min=100, max=510, home=100, invert=False),  # 100=open, 510=close
]
IMU_JOINTS = (6, 5)              # driven by arm tilt in "EMG + Tilt" mode (roll->Base, pitch->Reach)
IMU_RANGE_DEG = 40.0

TICK_HZ = 25
SEND_DURATION_MS = 90            # servo travel time per command (a bit > tick so moves blend)
MAX_RATE = 260.0                 # servo units / s at full activation, speed = 1.0
MANUAL_RATE = 0.6                # manual jog rate as fraction of MAX_RATE
CHASE_RATE = 1.6                 # how fast commanded pos may chase the goal (x MAX_RATE)
HOME_SPEED = 1.5

EMG_WINDOW_S = 0.20              # RMS window
THR_ON, THR_OFF = 0.30, 0.15     # normalized activation hysteresis
CO_HOLD = 0.45                   # s of co-contraction to switch joint
STALE_S = 0.5                    # no raw EMG for this long -> EMG motion blocked
CAL_SECONDS = 3.0

# gesture link (BioWave main.py publishes ~12 Hz)
GESTURE_PORT = 5005
GESTURE_STALE_S = 0.8            # no prediction for this long -> arm stops
GESTURE_DWELL_S = 0.15           # a gesture must persist this long before it moves the arm
ONE_SHOT_HOLD_S = 0.6            # Home / E-stop gestures must be held this long
DEFAULT_MIN_CONF = 35.0          # % confidence needed to act on a gesture

# BioWave device protocol (same constants as BioWave main.py)
BW_STREAM_PORT, BW_CONTROL_PORT = 5000, 5001
BW_EMG_CH, BW_IMU_CH = 8, 3
BW_HEADER = struct.Struct("<4sBBHI")
BW_FRAME = struct.Struct("<IIII8HfffB3x")
BW_FRAMES_PER_PACKET = 5
BW_RATE = 500
BW_USB_BAUD = 921600
BW_KEEPALIVE_S = 2.0

XARM_VID, XARM_PID = 0x0483, 0x5750

SETTINGS_FILE = os.path.join(HERE, "settings.json")
POSE_FILE = os.path.join(HERE, "poses.json")
LOG_DIR = os.path.join(HERE, "logs")

# Gesture -> action. Actions: "none", "home", "estop" or "<servo id><+|->" (e.g. "6+" = Base +).
# Keys are normalised labels (lower-case letters/digits only). Editable in the UI.
GESTURE_LABELS = ["Left", "Right", "fist_close", "Up", "Down", "Double_Closed_Fist", "Victory_Sign",
                  "Thumbs_Up", "Pinch_using_thumb_and_index_finger", "Rest"]
DEFAULT_GMAP = {"Left": "6+", "Right": "6-", "Up": "4+", "Down": "4-", "fist_close": "1+",
                "Pinch_using_thumb_and_index_finger": "1-", "Victory_Sign": "5+", "Thumbs_Up": "5-",
                "Double_Closed_Fist": "home", "Rest": "none"}


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def norm_label(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def action_options():
    out = [("none", "do nothing")]
    for j in JOINTS:
        extra = {"1": (" (close)", " (open)")}.get(str(j["id"]), ("", ""))
        out += [("%d+" % j["id"], j["name"] + " +" + extra[0]), ("%d-" % j["id"], j["name"] + " -" + extra[1])]
    return out + [("home", "Go HOME (hold)"), ("estop", "E-STOP (hold)")]


# =============================================================================
# ARM DRIVERS
# =============================================================================
class SimArm:
    kind = "sim"

    def open(self):
        pass

    def read_positions(self):
        return None

    def send(self, cmds, dur):
        pass

    def battery(self):
        return 7.4

    def close(self):
        pass


class XArm:
    """Real xArm over USB-HID ("USB") or serial ("COM3")."""

    def __init__(self, target):
        self.target = target
        self.kind = "usb" if target.upper().startswith("USB") else "serial"
        self.c = None

    def open(self):
        try:
            import xarm
        except ImportError:
            raise RuntimeError("Python package 'xarm' is missing for %s. Run:  %s -m pip install xarm hidapi"
                               % (sys.executable, sys.executable))
        if self.kind == "usb":
            try:
                import hid
            except ImportError:
                raise RuntimeError("Python package 'hidapi' is missing for %s. Run:  %s -m pip install hidapi"
                                   % (sys.executable, sys.executable))
            if not hid.enumerate(XARM_VID, XARM_PID):
                raise RuntimeError("xArm controller not found on USB. Check: controller switched ON, a data-capable "
                                   "USB cable, try another port, and unplug/replug it.")
        try:
            self.c = xarm.Controller(self.target)
        except Exception as e:
            if self.kind == "usb":
                raise RuntimeError("xArm found but cannot be opened (%s). Close other programs using it (the "
                                   "Vision-Guided app, other Python windows), then replug the USB cable." % e)
            raise RuntimeError("cannot open %s: %s" % (self.target, e))

    def read_positions(self):
        out = {}
        for j in JOINTS:
            for _ in range(5):                      # macOS HID reads fail intermittently
                try:
                    p = self.c.getPosition(j["id"])
                    if p is not None:
                        out[j["id"]] = p
                        break
                except Exception:
                    pass
                time.sleep(0.05)
            else:
                return None
        return out

    def send(self, cmds, dur):
        self.c.setPosition([[sid, int(pos)] for sid, pos in cmds], duration=int(dur), wait=False)

    def battery(self):
        return self.c.getBatteryVoltage()

    def close(self):
        self.c = None


def make_arm(kind, port):
    if kind == "sim":
        return SimArm()
    if kind == "usb":
        return XArm("USB")
    if kind == "serial":
        if not port:
            raise RuntimeError("choose a serial port (e.g. COM3)")
        return XArm(port)
    raise RuntimeError("unknown arm type " + kind)


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================
class Envelope:
    """Per-channel DC removal + sliding-window RMS."""

    def __init__(self, nch, rate):
        self.nch = nch
        win = max(8, int(EMG_WINDOW_S * rate))
        self.mean = [None] * nch
        self.sq = [collections.deque(maxlen=win) for _ in range(nch)]
        self.alpha = 2.0 / rate                      # ~0.5 s baseline time constant

    def push(self, sample):
        a = min(0.5, self.alpha)
        for i in range(self.nch):
            v = float(sample[i])
            m = self.mean[i]
            m = v if m is None else m + a * (v - m)
            self.mean[i] = m
            x = v - m
            self.sq[i].append(x * x)

    def value(self):
        return [math.sqrt(sum(q) / len(q)) if q else 0.0 for q in self.sq]


class Calibration:
    def __init__(self, nch, default_mvc):
        self.rest = [0.0] * nch
        self.mvc = [default_mvc] * nch

    def norm(self, env):
        return [clamp((e - r) / max(m - r, 1e-6), 0.0, 1.0)
                for e, r, m in zip(env, self.rest, self.mvc)]


# =============================================================================
# BIOWAVE DEVICE PROTOCOL (Wi-Fi control channel)
# =============================================================================
def bw_sign(secret, *parts):
    return hmac.new(secret.encode(), "|".join(str(p) for p in parts).encode(), hashlib.sha256).hexdigest()


def bw_exchange(msg, ip, timeout=2.0, many=False, broadcast=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.2 if many else timeout)
    if broadcast:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        s.sendto(msg.encode(), (ip, BW_CONTROL_PORT))
        if not many:
            data, addr = s.recvfrom(2048)
            return data.decode("utf-8", "replace"), addr[0]
        out, end = [], time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                data, addr = s.recvfrom(2048)
                out.append((data.decode("utf-8", "replace"), addr[0]))
            except socket.timeout:
                pass
        return out
    finally:
        s.close()


def bw_discover(timeout=1.5):
    found = {}
    try:
        for text, ip in bw_exchange("DISCOVER", "255.255.255.255", timeout, many=True, broadcast=True):
            p = text.strip().split("|")
            if len(p) == 8 and p[0] == "HELLO":
                found[ip] = dict(ip=ip, id=p[1], name=p[2], streaming=p[6] == "1", fw=p[7])
    except OSError:
        pass
    return list(found.values())


def bw_command(ip, key, cmd, *payload):
    if not key:
        raise RuntimeError("enter the device access key")
    try:
        resp, _ = bw_exchange("CHALLENGE", ip)
        p = resp.strip().split("|")
        if len(p) != 2 or p[0] != "CHALLENGE":
            raise RuntimeError("device sent an invalid challenge")
        msg = "|".join([cmd, p[1], *payload, bw_sign(key, cmd, p[1], *payload)])
        resp, _ = bw_exchange(msg, ip)
    except socket.timeout:
        raise RuntimeError("no reply from %s. Same Wi-Fi network? Allow Python through the firewall "
                           "(UDP 5000/5001). Is BioWave app streaming from it? Close it first." % ip)
    p = resp.strip().split("|")
    if p[0] == "ERR":
        raise RuntimeError("device rejected %s: %s (wrong access key?)" % (cmd, p[1] if len(p) > 1 else "?"))
    if p[0] != "ACK":
        raise RuntimeError("unexpected reply from device")
    return p[1:]


def local_ip_for(target):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 1))
        return s.getsockname()[0]
    finally:
        s.close()


# =============================================================================
# ARMBAND SOURCES
# =============================================================================
class Band:
    kind = "none"
    nch = 8
    rate = 200.0               # samples / s
    default_mvc = 40.0         # envelope at full effort before calibration
    gestures = False           # True: delivers labelled gestures instead of raw EMG

    def __init__(self, ctl):
        self.ctl = ctl
        self.stop = threading.Event()

    def start(self):
        threading.Thread(target=self._guard, daemon=True).start()

    def _guard(self):
        try:
            self.run()
        except Exception as e:
            if not self.stop.is_set():
                self.ctl.band_error(self, str(e))

    def close(self):
        self.stop.set()

    def run(self):
        raise NotImplementedError


class SimBand(Band):
    """Virtual armband. Levels come from the UI buttons / keyboard (F, E, B)."""
    kind = "sim"

    def run(self):
        self.ctl.band_ready(self)
        t0, n = time.time(), 0
        while not self.stop.is_set():
            a, b = self.ctl.sim["a"], self.ctl.sim["b"]
            half = self.nch // 2
            emg = [random.gauss(0, 2 + 38 * (a if i < half else b)) for i in range(self.nch)]
            self.ctl.on_sample(emg, (self.ctl.sim["roll"], self.ctl.sim["pitch"]))
            n += 1
            delay = t0 + n / self.rate - time.time()
            if delay > 0:
                time.sleep(delay)


class BridgeBand(Band):
    """Gesture predictions published by BioWave main.py on UDP 127.0.0.1:5005."""
    kind = "bridge"
    gestures = True

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", GESTURE_PORT))
        except OSError as e:
            s.close()
            raise RuntimeError("UDP port %d is busy (%s). Close gesture_game_app.py or any other listener "
                               "on that port." % (GESTURE_PORT, e))
        s.settimeout(0.2)
        self.ctl.band_ready(self)
        try:
            while not self.stop.is_set():
                try:
                    data, _ = s.recvfrom(1024)
                except socket.timeout:
                    continue
                try:
                    m = json.loads(data.decode("utf-8"))
                    self.ctl.on_gesture(str(m.get("label", "")), float(m.get("conf_pct", 0.0)))
                except (ValueError, TypeError):
                    pass
        finally:
            s.close()


def usb_port_candidates(strict=False):
    """USB serial ports, most likely armband first. strict=True returns only confident matches."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    good, rest = [], []
    for p in lp.comports():
        dev = p.device
        text = ("%s %s %s" % (dev, p.description, p.hwid)).lower()
        if any(k in text for k in ("bluetooth", "debug", "wlan", "soundcore", "airpods")) or dev.startswith("/dev/tty."):
            continue
        likely = (dev.upper().startswith("COM") or any(k in dev.lower() for k in ("usbmodem", "usbserial", "ttyacm", "ttyusb"))
                  or p.vid in (0x303A, 0x10C4, 0x1A86, 0x0403))
        (good if likely else rest).append(dev)
    return good if strict else good + rest


def list_ports():
    return usb_port_candidates()


class SerialBand(Band):
    """BioWave ESP32-S3 over USB: text lines `c1 c2 ... c8 [i1 i2 i3]` (spaces or commas)."""
    kind = "serial"
    rate = BW_RATE
    default_mvc = 150.0

    def __init__(self, ctl, port, baud, nch):
        super().__init__(ctl)
        self.port, self.baud, self.nch = port, baud, nch

    def run(self):
        try:
            import serial
        except ImportError:
            raise RuntimeError("pyserial is missing for %s. Run:  %s -m pip install pyserial"
                               % (sys.executable, sys.executable))
        port = self.port
        if port in ("", "auto"):
            cands = usb_port_candidates(strict=True)
            if not cands:
                raise RuntimeError("no USB serial port found. Plug in the BioWave armband (data cable) and retry.")
            port = cands[0]
        try:
            s = serial.Serial(port, self.baud, timeout=0.05)
        except Exception as e:
            msg = str(e)
            if "busy" in msg.lower() or "Errno 16" in msg or "denied" in msg.lower():
                msg += " - the port is in use: close the BioWave app / serial monitor first."
            raise RuntimeError("cannot open %s: %s" % (port, msg))
        nch, tail, count = self.nch, b"", 0
        with s:
            try:
                s.reset_input_buffer()
            except Exception:
                pass
            self.ctl.band_ready(self, "%s @ %d" % (port, self.baud))
            while not self.stop.is_set():
                chunk = s.read(max(1, s.in_waiting))
                if not chunk:
                    continue
                lines = (tail + chunk).split(b"\n")
                tail = lines.pop()
                for raw in lines:
                    parts = raw.replace(b",", b" ").split()
                    if len(parts) < nch:
                        continue
                    try:
                        v = [float(x) for x in parts[:nch + 3]]
                    except ValueError:
                        continue
                    self.ctl.on_sample(v[:nch], (v[nch], v[nch + 1]) if len(v) >= nch + 2 else None)


class WifiBand(Band):
    """BioWave ESP32-S3 over Wi-Fi: signed START, UDP frames on port 5000, keep-alive + self-heal."""
    kind = "wifi"
    rate = BW_RATE
    default_mvc = 150.0
    nch = BW_EMG_CH

    def __init__(self, ctl, ip, key):
        super().__init__(ctl)
        self.ip, self.key = ip, key
        self.last_rx = 0.0

    def _start(self):
        bw_command(self.ip, self.key, "START", local_ip_for(self.ip), str(BW_STREAM_PORT))

    def run(self):
        if not self.ip:
            raise RuntimeError("scan and pick a BioWave device first")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        try:
            s.bind(("0.0.0.0", BW_STREAM_PORT))
        except OSError as e:
            s.close()
            raise RuntimeError("UDP port %d is busy (%s). Close the BioWave app (it uses the same stream)."
                               % (BW_STREAM_PORT, e))
        s.settimeout(0.1)
        try:
            self._start()
            self.last_rx = time.monotonic()
            self.ctl.band_ready(self, "device " + self.ip)
            threading.Thread(target=self._keepalive, daemon=True).start()
            size = BW_HEADER.size + BW_FRAME.size * BW_FRAMES_PER_PACKET
            while not self.stop.is_set():
                try:
                    data, _ = s.recvfrom(2048)
                except socket.timeout:
                    continue
                if len(data) != size:
                    continue
                magic, ver, nfr, fsz, _seq = BW_HEADER.unpack_from(data, 0)
                if magic != b"BWIM" or ver != 1 or fsz != BW_FRAME.size:
                    continue
                self.last_rx = time.monotonic()
                for k in range(nfr):
                    f = BW_FRAME.unpack_from(data, BW_HEADER.size + k * fsz)
                    self.ctl.on_sample(f[4:12], (f[12], f[13]))
        finally:
            s.close()
            try:
                bw_command(self.ip, self.key, "STOP")
            except Exception:
                pass

    def _keepalive(self):
        while not self.stop.wait(BW_KEEPALIVE_S):
            try:
                if time.monotonic() - self.last_rx > 3.0:
                    self._start()                       # stream stalled: ask the device to resume
                else:
                    bw_command(self.ip, self.key, "PING")
            except Exception as e:
                self.ctl.band_warn(self, str(e))


class MyoBand(Band):
    """Thalmic Myo via the `pyomyo` package + Myo BLE dongle. (Not hardware-tested.)"""
    kind = "myo"
    nch = 8

    def run(self):
        try:
            from pyomyo import Myo, emg_mode
        except ImportError:
            raise RuntimeError("pyomyo is missing for %s. Run:  %s -m pip install pyomyo"
                               % (sys.executable, sys.executable))
        m = Myo(mode=emg_mode.RAW)
        m.connect()
        imu = [0.0, 0.0]

        def on_imu(quat, acc, gyro):
            w, x, y, z = [q / 16384.0 for q in quat]
            imu[0] = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            imu[1] = math.degrees(math.asin(clamp(2 * (w * y - z * x), -1, 1)))

        m.add_emg_handler(lambda emg, mov: self.ctl.on_sample(list(emg), tuple(imu)))
        m.add_imu_handler(on_imu)
        self.ctl.band_ready(self)
        try:
            while not self.stop.is_set():
                m.run()
        finally:
            try:
                m.disconnect()
            except Exception:
                pass


# =============================================================================
# CONTROLLER  (state + decoder + output loop)
# =============================================================================
class Controller:
    RETRY_S = 4.0

    def __init__(self):
        self.lk = threading.RLock()
        self.arm = None
        self.arm_status, self.arm_msg = "disconnected", ""     # disconnected|connecting|connected|error
        self.arm_cfg, self.want_arm, self.arm_retry = None, False, 0.0
        self.band = None
        self.band_status, self.band_msg = "disconnected", ""
        self.band_cfg, self.want_band, self.band_retry = None, False, 0.0
        self.nch = 8
        self.env = Envelope(self.nch, 200)
        self.cal = Calibration(self.nch, 40.0)
        self.swap = False
        self.last_sample = 0.0
        self.rate = 0.0
        self._rate_t, self._rate_n = time.time(), 0
        self.imu = None
        self.imu0 = (0.0, 0.0)

        self.mode = "emg"                      # emg | imu | manual
        self.speed = 0.6
        self.estop = False
        self.active = 2                        # joint id; start on Wrist
        self.pos = {j["id"]: float(j["home"]) for j in JOINTS}
        self.goal = dict(self.pos)
        self.last_sent = {}
        self.jog = {}                          # id -> (dir, expiry)
        self.homing_until = 0.0
        self.first_send_dur = None
        self.battery = None
        self._bat_t = time.time()
        self.invert = {j["id"]: j["invert"] for j in JOINTS}
        self.cmd_errors = 0

        self.a_on = self.b_on = False
        self.co_t = 0.0
        self.co_fired = False
        self.gesture = "Rest"
        self.norm = [0.0] * self.nch
        self.a = self.b = 0.0
        self.cal_job = None
        self.cal_msg = ""
        self.sim = dict(a=0.0, b=0.0, roll=0.0, pitch=0.0)
        self.events = collections.deque(maxlen=14)
        self.poses = self._load_poses()
        self.logf = self.logw = None

        # gesture link
        self.gmap = {norm_label(k): v for k, v in DEFAULT_GMAP.items()}
        self.glabels = {norm_label(k): k for k in GESTURE_LABELS}
        self.min_conf = DEFAULT_MIN_CONF
        self.g_label, self.g_conf, self.g_ts = "", 0.0, 0.0
        self.g_cur, self.g_since, self.g_fired = None, 0.0, False

        # wifi discovery / saved settings
        self.wifi_devices, self.scanning = [], False
        self._err = {"arm": "", "band": ""}      # last reported error (dedupes the event log)
        self.key = os.environ.get("BIOWAVE_KEY", "")
        self.saved = {}
        self._load_settings()

    # ------------------------------------------------------------------ utils
    def ev(self, msg):
        self.events.appendleft("%s  %s" % (time.strftime("%H:%M:%S"), msg))

    def _load_poses(self):
        try:
            with open(POSE_FILE) as f:
                return {str(k): {int(a): b for a, b in v.items()} for k, v in json.load(f).items()}
        except Exception:
            return {}

    def _save_poses(self):
        with open(POSE_FILE, "w") as f:
            json.dump(self.poses, f)

    def _load_settings(self):
        try:
            with open(SETTINGS_FILE) as f:
                s = json.load(f)
        except Exception:
            return
        self.saved = s
        self.gmap.update(s.get("gmap", {}))
        self.speed = float(s.get("speed", self.speed))
        self.min_conf = float(s.get("min_conf", self.min_conf))
        self.swap = bool(s.get("swap", False))
        self.key = self.key or s.get("key", "")
        for k, v in s.get("invert", {}).items():
            if int(k) in self.invert:
                self.invert[int(k)] = bool(v)

    def _save_settings(self):
        s = {k: v for k, v in self.saved.items() if k != "key"}
        s.update(gmap=self.gmap, speed=self.speed, min_conf=self.min_conf, swap=self.swap,
                 invert=self.invert)
        self.saved = dict(s)
        s["key"] = self.key
        try:
            fd = os.open(SETTINGS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # holds the device key
            with os.fdopen(fd, "w") as f:
                json.dump(s, f, indent=1)
        except OSError:
            pass

    def jdef(self, jid):
        return next(j for j in JOINTS if j["id"] == jid)

    def cycle_ids(self):
        ids = [j["id"] for j in JOINTS]
        return [i for i in ids if i not in IMU_JOINTS] if self.mode == "imu" else ids

    def diagnostics(self):
        def has(m):
            try:
                __import__(m)
                return True
            except Exception:
                return False
        hid_n = None
        if has("hid"):
            try:
                import hid
                hid_n = len(hid.enumerate(XARM_VID, XARM_PID))
            except Exception:
                hid_n = None
        return dict(python=sys.executable, xarm=has("xarm"), hid=has("hid"), serial=has("serial"),
                    pyomyo=has("pyomyo"), xarm_usb_devices=hid_n, ports=usb_port_candidates())

    # ---------------------------------------------------------- arm connection
    def connect_arm(self, kind, port="", user=True):
        with self.lk:
            self._drop_arm()
            self.arm_cfg = (kind, port)
            self.want_arm = True
            self.arm_status, self.arm_msg = "connecting", ""
            if user:
                self.saved.update(arm_kind=kind, arm_port=port)
                self._save_settings()
        threading.Thread(target=self._connect_arm, args=(kind, port), daemon=True).start()

    def _connect_arm(self, kind, port):
        try:
            arm = make_arm(kind, port)
            arm.open()
            posn = arm.read_positions()
            with self.lk:
                if self.arm_cfg != (kind, port) or not self.want_arm:      # superseded while connecting
                    arm.close()
                    return
                self.arm = arm
                self.cmd_errors = 0
                self.last_sent = {}
                if posn:
                    for k, v in posn.items():
                        self.pos[k] = self.goal[k] = float(v)
                    self.first_send_dur = None
                    self.ev("arm connected, positions read")
                else:
                    for j in JOINTS:
                        self.pos[j["id"]] = self.goal[j["id"]] = float(j["home"])
                    self.first_send_dur = 1500 if kind != "sim" else None
                    self.ev("arm connected - moving to HOME" if kind != "sim" else "simulated arm ready")
                self.arm_status, self.arm_msg = "connected", ""
                self._err["arm"] = ""
        except Exception as e:
            with self.lk:
                if str(e) != self._err["arm"]:
                    self.ev("arm: " + str(e))
                    self._err["arm"] = str(e)
                self.arm_status, self.arm_msg = "error", str(e)
                self.arm_retry = time.time() + self.RETRY_S

    def _drop_arm(self):
        if self.arm:
            try:
                self.arm.close()
            except Exception:
                pass
        self.arm = None
        self.arm_status = "disconnected"

    def disconnect_arm(self):
        with self.lk:
            self.want_arm = False
            self._drop_arm()

    # -------------------------------------------------------- band connection
    def connect_band(self, kind, port="", baud=BW_USB_BAUD, nch=8, ip="", key="", user=True):
        with self.lk:
            self._drop_band()
            self.want_band = True
            if key:
                self.key = key
            self.band_cfg = dict(kind=kind, port=port, baud=baud, nch=nch, ip=ip)
            if user:
                self.saved.update(band=self.band_cfg)
                self._save_settings()
            try:
                if kind == "sim":
                    band = SimBand(self)
                elif kind == "bridge":
                    band = BridgeBand(self)
                elif kind == "serial":
                    band = SerialBand(self, port, int(baud), int(clamp(int(nch), 2, 16)))
                elif kind == "wifi":
                    band = WifiBand(self, ip, self.key)
                elif kind == "myo":
                    band = MyoBand(self)
                else:
                    raise RuntimeError("unknown band " + kind)
            except Exception as e:
                self.band_status, self.band_msg = "error", str(e)
                self.band_retry = time.time() + self.RETRY_S
                return
            self.nch = band.nch
            self.env = Envelope(self.nch, band.rate)
            self.cal = Calibration(self.nch, band.default_mvc)
            self.norm = [0.0] * self.nch
            self.imu = None
            self.last_sample = 0.0
            self.g_ts = 0.0
            self.g_cur = None
            self.band = band
            self.band_status, self.band_msg = "connecting", ""
            band.start()

    def _drop_band(self):
        if self.band:
            self.band.close()
        self.band = None
        self.band_status = "disconnected"
        self.a_on = self.b_on = False

    def disconnect_band(self):
        with self.lk:
            self.want_band = False
            self._drop_band()

    def band_ready(self, band, info=""):
        with self.lk:
            if band is self.band:
                self.band_status, self.band_msg = "connected", ""
                self._err["band"] = ""
                self.ev("armband connected: %s %s" % (band.kind, info))

    def band_error(self, band, msg):
        with self.lk:
            if band is self.band:
                if msg != self._err["band"]:
                    self.ev("armband: " + msg)
                    self._err["band"] = msg
                self.band_status, self.band_msg = "error", msg
                self.band_retry = time.time() + self.RETRY_S
                band.stop.set()

    def band_warn(self, band, msg):
        with self.lk:
            if band is self.band and msg != self._err["band"]:
                self._err["band"] = msg
                self.band_msg = msg
                self.ev("armband: " + msg)

    def on_sample(self, emg, imu):
        with self.lk:
            self.env.push(emg)
            self.last_sample = time.time()
            self._rate_n += 1
            if imu is not None:
                self.imu = imu

    def on_gesture(self, label, conf):
        with self.lk:
            key = norm_label(label)
            if key and key != "na" and key not in self.glabels:
                self.glabels[key] = label
                self.gmap.setdefault(key, "none")
            self.g_label, self.g_conf, self.g_ts = key, conf, time.time()
            self.last_sample = self.g_ts
            self._rate_n += 1

    # ------------------------------------------------------------- wifi scan
    def scan_wifi(self):
        with self.lk:
            if self.scanning:
                return
            self.scanning = True

        def work():
            devs = bw_discover()
            with self.lk:
                self.wifi_devices, self.scanning = devs, False
                self.ev("Wi-Fi scan: %d BioWave device(s)" % len(devs))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ calibration
    def start_cal(self, kind):
        with self.lk:
            if kind == "clear":
                b = self.band
                self.cal = Calibration(self.nch, b.default_mvc if b else 40.0)
                self.cal_msg = "calibration cleared"
                return
            self.cal_job = dict(kind=kind, t0=time.time(), samples=[])
            self.cal_msg = {"rest": "Relax your arm...", "flex": "Flex (wrist toward palm) firmly...",
                            "ext": "Extend (wrist back) firmly..."}[kind]

    def _cal_step(self, env):
        job = self.cal_job
        if not job:
            return
        el = time.time() - job["t0"]
        if el > 0.6:
            job["samples"].append(env)
        if el >= CAL_SECONDS:
            S = job["samples"]
            half = self.nch // 2
            if S:
                if job["kind"] == "rest":
                    self.cal.rest = [sum(s[i] for s in S) / len(S) for i in range(self.nch)]
                else:
                    idx = range(0, half) if (job["kind"] == "flex") != self.swap else range(half, self.nch)
                    for i in idx:
                        v = sorted(s[i] for s in S)
                        self.cal.mvc[i] = max(v[int(0.9 * (len(v) - 1))], self.cal.rest[i] + 1.0)
            self.ev("calibration '%s' done" % job["kind"])
            self.cal_msg = "'%s' calibrated" % job["kind"]
            self.cal_job = None

    # ---------------------------------------------------------- user commands
    def _do_estop(self):
        self.estop = True
        self.goal = dict(self.pos)
        self.jog.clear()
        self.ev("EMERGENCY STOP - commands frozen")

    def _do_home(self):
        self.goal = {j["id"]: float(j["home"]) for j in JOINTS}
        self.homing_until = time.time() + 4
        self.ev("going HOME")

    def command(self, c):
        cmd = c.get("cmd")
        with self.lk:
            if cmd == "connect_arm":
                self.connect_arm(c.get("kind", "sim"), c.get("port", ""))
            elif cmd == "disconnect_arm":
                self.disconnect_arm()
            elif cmd == "connect_band":
                self.connect_band(c.get("kind", "sim"), c.get("port", ""), c.get("baud", BW_USB_BAUD),
                                  c.get("channels", 8), c.get("ip", ""), c.get("key", ""))
            elif cmd == "disconnect_band":
                self.disconnect_band()
            elif cmd == "scan_wifi":
                self.scan_wifi()
            elif cmd == "estop":
                self._do_estop()
            elif cmd == "reset":
                self.estop = False
                self.goal = dict(self.pos)
                self.ev("e-stop released")
            elif cmd == "home":
                self._do_home()
            elif cmd == "gripper":
                g = self.jdef(1)
                self.goal[1] = float(g["min"] if c.get("state") == "open" else g["max"])
            elif cmd == "goal":
                j = self.jdef(int(c["joint"]))
                self.goal[j["id"]] = clamp(float(c["value"]), j["min"], j["max"])
            elif cmd == "jog":
                self.jog[int(c["joint"])] = (int(c["dir"]), time.time() + 0.25)
            elif cmd == "select":
                self.active = int(c["joint"])
            elif cmd == "mode":
                self.mode = c["mode"]
                if self.active not in self.cycle_ids():
                    self.active = self.cycle_ids()[0]
                self.ev("mode: " + self.mode)
            elif cmd == "speed":
                self.speed = clamp(float(c["value"]), 0.1, 1.5)
                self._save_settings()
            elif cmd == "min_conf":
                self.min_conf = clamp(float(c["value"]), 0, 100)
                self._save_settings()
            elif cmd == "gmap":
                self.gmap[norm_label(c["label"])] = c["action"]
                self._save_settings()
            elif cmd == "cal":
                self.start_cal(c["kind"])
            elif cmd == "swap":
                self.swap = bool(c["value"])
                self._save_settings()
            elif cmd == "invert":
                self.invert[int(c["joint"])] = bool(c["value"])
                self._save_settings()
            elif cmd == "zero_imu":
                if self.imu:
                    self.imu0 = tuple(self.imu)
                    self.ev("IMU zeroed")
            elif cmd == "pose_save":
                self.poses[str(c["slot"])] = dict(self.pos)
                self._save_poses()
                self.ev("pose %s saved" % c["slot"])
            elif cmd == "pose_load":
                p = self.poses.get(str(c["slot"]))
                if p:
                    self.goal = {int(k): float(v) for k, v in p.items()}
                    self.homing_until = time.time() + 4
                    self.ev("pose %s" % c["slot"])
            elif cmd == "sim":
                for k in ("a", "b", "roll", "pitch"):
                    if k in c:
                        self.sim[k] = float(c[k])
            elif cmd == "log":
                self._set_log(bool(c["on"]))

    def _set_log(self, on):
        if on and not self.logf:
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, time.strftime("session_%Y%m%d_%H%M%S.csv"))
            self.logf = open(path, "w", newline="")
            self.logw = csv.writer(self.logf)
            self.logw.writerow(["t", "flex", "ext", "gesture", "active_joint"] +
                               ["j%d" % j["id"] for j in JOINTS] + ["imu_roll", "imu_pitch"])
            self.ev("logging -> " + os.path.relpath(path, HERE))
        elif not on and self.logf:
            self.logf.close()
            self.logf = self.logw = None
            self.ev("logging stopped")

    # ------------------------------------------------------------ control loop
    def run(self):
        dt = 1.0 / TICK_HZ
        nxt = time.time()
        while True:
            try:
                self._auto_reconnect()
                self.tick(dt)
            except Exception as e:
                with self.lk:
                    self.ev("loop error: %r" % e)
            nxt += dt
            time.sleep(max(0.0, nxt - time.time()))

    def _auto_reconnect(self):
        now = time.time()
        with self.lk:
            if self.want_arm and self.arm_status == "error" and now >= self.arm_retry and self.arm_cfg:
                self.arm_retry = now + self.RETRY_S
                self.connect_arm(*self.arm_cfg, user=False)
            if self.want_band and self.band_status == "error" and now >= self.band_retry and self.band_cfg:
                self.band_retry = now + self.RETRY_S
                cfg = self.band_cfg
                msg = self.band_msg
                self.connect_band(cfg["kind"], cfg["port"], cfg["baud"], cfg["nch"], cfg["ip"], user=False)
                self.band_msg = msg                     # keep the last error visible while retrying

    def _apply_action(self, act, now, dt, label):
        """Run a gesture action; returns text for the status line."""
        if act in ("home", "estop"):
            if now - self.g_since >= ONE_SHOT_HOLD_S:
                if not self.g_fired:
                    self.g_fired = True
                    (self._do_home if act == "home" else self._do_estop)()
                return "%s -> %s" % (label, act.upper())
            return "%s -> hold for %s (%d%%)" % (label, act, int(100 * (now - self.g_since) / ONE_SHOT_HOLD_S))
        jid, sign = int(act[:-1]), (1 if act[-1] == "+" else -1)
        if self.invert[jid]:
            sign = -sign
        self.goal[jid] += sign * MAX_RATE * self.speed * dt
        self.active = jid
        return "%s -> %s %s" % (label, self.jdef(jid)["name"], act[-1])

    def tick(self, dt):
        now = time.time()
        with self.lk:
            if now - self._rate_t >= 1.0:
                self.rate = self._rate_n / (now - self._rate_t)
                self._rate_t, self._rate_n = now, 0
            band = self.band
            gest = bool(band and band.gestures)
            if gest:
                live = self.band_status == "connected" and (now - self.g_ts) < GESTURE_STALE_S
                if self.band_status == "connected":
                    self.band_msg = "" if live else "waiting for BioWave app predictions (start Realtime prediction)"
            else:
                stale = (now - self.last_sample) > STALE_S
                live = self.band_status == "connected" and not stale
                if self.band_status == "connected":
                    self.band_msg = "no data from armband" if (stale and self.last_sample) else ""

            direction, level = 0, 0.0
            self.a = self.b = 0.0
            if live and not gest:
                env = self.env.value()
                self._cal_step(env)
                self.norm = self.cal.norm(env)
                half = self.nch // 2
                ga, gb = self.norm[:half], self.norm[half:]
                if self.swap:
                    ga, gb = gb, ga
                self.a, self.b = sum(ga) / len(ga), sum(gb) / len(gb)
                self.a_on = self.a > THR_ON or (self.a_on and self.a > THR_OFF)
                self.b_on = self.b > THR_ON or (self.b_on and self.b > THR_OFF)
            else:
                self.a_on = self.b_on = False
                self.cal_job = None
                if not gest:
                    self.norm = [0.0] * self.nch

            active_ok = self.mode != "manual" and live and not self.estop
            name = self.jdef(self.active)["name"]
            gest_action = None
            if gest:
                self.co_t, self.co_fired = 0.0, False
                act = self.gmap.get(self.g_label, "none") if active_ok and self.g_conf >= self.min_conf else "none"
                if act == "none":
                    self.g_cur, self.g_fired = None, False
                    self.gesture = ("E-STOP" if self.estop else "manual mode" if self.mode == "manual" else
                                    "no gesture data" if not live else
                                    "%s (%d%%) - no action" % (self.glabels.get(self.g_label, self.g_label), self.g_conf))
                else:
                    if self.g_label != self.g_cur:
                        self.g_cur, self.g_since, self.g_fired = self.g_label, now, False
                    if now - self.g_since >= GESTURE_DWELL_S:
                        gest_action = act
                    else:
                        self.gesture = "%s ..." % self.glabels.get(self.g_label, self.g_label)
            elif active_ok:
                if self.a_on and self.b_on:
                    self.co_t += dt
                    self.gesture = "BOTH - hold to switch joint (%d%%)" % min(100, int(100 * self.co_t / CO_HOLD))
                    if self.co_t >= CO_HOLD and not self.co_fired:
                        ids = self.cycle_ids()
                        self.active = ids[(ids.index(self.active) + 1) % len(ids)] if self.active in ids else ids[0]
                        self.co_fired = True
                        self.ev("joint -> " + self.jdef(self.active)["name"])
                elif self.a_on or self.b_on:
                    self.co_t = 0.0
                    if not self.co_fired:
                        direction = 1 if self.a_on else -1
                        v = self.a if self.a_on else self.b
                        level = clamp((v - THR_OFF) / (1 - THR_OFF), 0.0, 1.0)
                        self.gesture = "%s -> %s %s" % ("FLEX" if self.a_on else "EXTEND", name,
                                                         "+" if direction > 0 else "-")
                    else:
                        self.gesture = "release to continue"
                else:
                    self.co_t, self.co_fired = 0.0, False
                    self.gesture = "Rest"
            else:
                self.co_t, self.co_fired = 0.0, False
                self.gesture = ("E-STOP" if self.estop else "manual mode" if self.mode == "manual" else
                                "no armband data" if not live else "Rest")

            if self.homing_until and all(abs(self.goal[k] - self.pos[k]) < 1.5 for k in self.pos):
                self.homing_until = 0.0
            homing = now < self.homing_until
            if self.cal_job or homing:                 # no operator input while calibrating / going home
                direction, gest_action = 0, None

            if not self.estop:
                if direction:
                    s = -1 if self.invert[self.active] else 1
                    self.goal[self.active] += s * direction * level * MAX_RATE * self.speed * dt
                if gest_action:
                    self.gesture = self._apply_action(gest_action, now, dt,
                                                      self.glabels.get(self.g_label, self.g_label))
                if self.mode == "imu" and live and self.imu and not gest and not homing and not self.cal_job:
                    for jid, raw, zero in ((IMU_JOINTS[0], self.imu[0], self.imu0[0]),
                                           (IMU_JOINTS[1], self.imu[1], self.imu0[1])):
                        j = self.jdef(jid)
                        k = clamp((raw - zero) / IMU_RANGE_DEG, -1, 1) * (-1 if self.invert[jid] else 1)
                        mid, half_r = (j["min"] + j["max"]) / 2, (j["max"] - j["min"]) / 2
                        self.goal[jid] = mid + k * half_r
                for jid, (d, exp) in list(self.jog.items()):
                    if exp < now:
                        del self.jog[jid]
                    elif d and not homing:
                        s = -1 if self.invert[jid] else 1
                        self.goal[jid] += s * d * MANUAL_RATE * MAX_RATE * self.speed * dt
                for j in JOINTS:
                    self.goal[j["id"]] = clamp(self.goal[j["id"]], j["min"], j["max"])
                rate = (HOME_SPEED if homing else CHASE_RATE) * MAX_RATE * (1 if homing else self.speed)
                for jid in self.pos:
                    step = rate * dt
                    self.pos[jid] += clamp(self.goal[jid] - self.pos[jid], -step, step)
                self._send()

            if self.logw and live:
                self.logw.writerow([round(now, 3), round(self.a, 3), round(self.b, 3), self.gesture,
                                    self.active] + [round(self.pos[j["id"]], 1) for j in JOINTS] +
                                   (list(self.imu) if self.imu else ["", ""]))

    def _send(self):
        if self.arm is None or self.arm_status != "connected":
            return
        first = self.first_send_dur
        cmds = [(sid, int(round(p))) for sid, p in self.pos.items()
                if first or abs(int(round(p)) - self.last_sent.get(sid, -9999)) >= 1]
        if not cmds:
            if time.time() - self._bat_t > 20:
                self._bat_t = time.time()
                try:
                    self.battery = self.arm.battery()
                except Exception:
                    pass
            return
        try:
            self.arm.send(cmds, first or SEND_DURATION_MS)
            for sid, p in cmds:
                self.last_sent[sid] = p
            self.first_send_dur = None
            self.cmd_errors = 0
        except Exception as e:
            self.cmd_errors += 1
            if self.cmd_errors >= 5:
                self.arm_status, self.arm_msg = "error", "lost connection: %s (auto-retrying)" % e
                self.arm_retry = time.time() + self.RETRY_S
                self.ev("arm: connection lost")

    # ------------------------------------------------------------------ state
    def state(self, with_diag=False):
        with self.lk:
            gl = self.glabels
            st = dict(
                arm=dict(status=self.arm_status, msg=self.arm_msg, battery=self.battery,
                         kind=getattr(self.arm, "kind", "")),
                band=dict(status=self.band_status, msg=self.band_msg, rate=round(self.rate),
                          kind=getattr(self.band, "kind", ""), nch=self.nch, imu=bool(self.imu),
                          gestures=bool(self.band and self.band.gestures)),
                mode=self.mode, speed=self.speed, estop=self.estop, active=self.active, swap=self.swap,
                joints=[dict(id=j["id"], name=j["name"], min=j["min"], max=j["max"],
                             pos=round(self.pos[j["id"]]), goal=round(self.goal[j["id"]]),
                             invert=self.invert[j["id"]]) for j in JOINTS],
                a=round(self.a, 3), b=round(self.b, 3), norm=[round(x, 3) for x in self.norm],
                gesture=self.gesture,
                gest=dict(label=gl.get(self.g_label, self.g_label), conf=round(self.g_conf),
                          online=(time.time() - self.g_ts) < GESTURE_STALE_S, min_conf=self.min_conf),
                gmap=[dict(key=k, label=gl.get(k, k), action=self.gmap.get(k, "none")) for k in gl],
                actions=[dict(v=v, l=l) for v, l in action_options()],
                cal=dict(msg=self.cal_msg, busy=bool(self.cal_job),
                         prog=(time.time() - self.cal_job["t0"]) / CAL_SECONDS if self.cal_job else 0),
                poses=sorted(self.poses.keys()), logging=bool(self.logf),
                events=list(self.events), thr=[THR_OFF, THR_ON],
                wifi=dict(devices=self.wifi_devices, scanning=self.scanning, has_key=bool(self.key)),
                saved={k: v for k, v in self.saved.items() if k != "key"},
            )
        if with_diag:
            st["diag"] = self.diagnostics()
        return st


# =============================================================================
# WEB UI
# =============================================================================
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMG Arm Control</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#1c2230;--mut:#6b7385;--line:#e1e4ea;--acc:#2563eb;--ok:#16a34a;--warn:#d97706;--bad:#dc2626;--fa:#f97316;--fb:#0ea5e9}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b202b;--ink:#e8ebf2;--mut:#8f98ab;--line:#2c3342;--acc:#5b8cff;--ok:#34d399;--warn:#fbbf24;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,system-ui,Segoe UI,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:16px}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
h1{font-size:18px;margin:0;flex:1}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 10px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.row:last-child{margin-bottom:0}
.lbl{width:70px;font-weight:600}
select,input[type=text],input[type=password],input[type=number]{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 8px;font:inherit;max-width:100%}
button{background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font:inherit;cursor:pointer;user-select:none;touch-action:manipulation}
button:hover{border-color:var(--acc)}button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}button:disabled{opacity:.45;cursor:default}
#estop{background:var(--bad);color:#fff;border-color:var(--bad);font-weight:700;padding:10px 18px}
#estop.latched{background:var(--ok);border-color:var(--ok)}
.pill{display:inline-flex;gap:6px;align-items:center;padding:3px 10px;border-radius:99px;border:1px solid var(--line);font-size:13px;color:var(--mut)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mut)}
.connected .dot{background:var(--ok)}.connecting .dot{background:var(--warn)}.error .dot{background:var(--bad)}
.seg{display:inline-flex}.seg button{border-radius:0;margin-left:-1px}.seg button:first-child{border-radius:8px 0 0 8px;margin:0}.seg button:last-child{border-radius:0 8px 8px 0}
.gesture{font-size:22px;font-weight:700;text-align:center;padding:10px;border-radius:10px;background:var(--bg);margin-bottom:10px}
.bars{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px}
.bar{height:22px;background:var(--bg);border-radius:6px;position:relative;overflow:hidden;border:1px solid var(--line)}
.bar i{position:absolute;left:0;top:0;bottom:0;width:0}.bar b{position:absolute;top:0;bottom:0;width:2px;background:var(--ink);opacity:.5}
.ch{display:flex;gap:3px;height:34px;align-items:flex-end;margin-top:6px}.ch i{flex:1;background:var(--mut);opacity:.55;border-radius:2px 2px 0 0;min-height:2px}
.joint{display:grid;grid-template-columns:96px 1fr auto auto;gap:8px;align-items:center;padding:6px;border-radius:10px;border:2px solid transparent}
.joint.active{border-color:var(--acc);background:color-mix(in srgb,var(--acc) 8%,transparent)}
.joint .nm{font-weight:600;cursor:pointer}.joint .nm small{display:block;color:var(--mut);font-weight:400}
.joint input[type=range]{width:100%}
.jog button{width:38px;padding:8px 0}
.small{font-size:12px;color:var(--mut)}.err{color:var(--bad)}
.ev{font:12px ui-monospace,Menlo,monospace;color:var(--mut);max-height:130px;overflow:auto}
details summary{cursor:pointer;color:var(--mut);font-size:13px;margin-bottom:8px}
.gm{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px}.gm div{display:flex;gap:6px;align-items:center;justify-content:space-between}
.hide{display:none!important}
@media(max-width:560px){.joint{grid-template-columns:70px 1fr auto}.joint .inv{display:none}.gm{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<header>
  <h1>EMG Arm Control</h1>
  <span class="pill" id="pArm"><span class="dot"></span><span id="tArm">arm</span></span>
  <span class="pill" id="pBand"><span class="dot"></span><span id="tBand">armband</span></span>
  <button id="estop" onclick="estop()">E-STOP</button>
</header>

<div class="card"><h2>1 · Connect</h2>
  <div class="row"><span class="lbl">Arm</span>
    <select id="armKind" onchange="kindUI()"><option value="usb">xArm USB (auto-detect)</option><option value="serial">xArm serial (COM)</option><option value="sim">Simulated</option></select>
    <select id="armPort" class="ports" style="display:none"></select>
    <button class="pri" onclick="connArm()" id="bArm">Connect</button></div>
  <div class="small err" id="mArm"></div>
  <div class="row"><span class="lbl">Armband</span>
    <select id="bandKind" onchange="kindUI()">
      <option value="bridge">BioWave app (gestures)</option><option value="wifi">BioWave Wi-Fi (direct)</option>
      <option value="serial">BioWave USB cable (direct)</option><option value="myo">Myo (BLE dongle)</option><option value="sim">Simulated</option></select>
    <button class="pri" onclick="connBand()" id="bBand">Connect</button></div>
  <div class="row" id="rowWifi"><span class="lbl"></span>
    <button onclick="post({cmd:'scan_wifi'})" id="bScan">Scan</button>
    <select id="wifiDev"></select>
    <input type="password" id="wifiKey" placeholder="device access key" autocomplete="off" style="width:190px"></div>
  <div class="row" id="rowUsb"><span class="lbl"></span>
    <select id="bandPort" class="ports"></select>
    <input type="number" id="bandCh" value="8" min="2" max="16" style="width:64px" title="EMG channels">
    <button onclick="refreshPorts()">↻ ports</button></div>
  <div class="small err" id="mBand"></div>
  <div class="small" id="bandHelp"></div>
  <div class="row"><button onclick="connAll()">Connect both</button>
    <span class="small">Failed links retry automatically every few seconds.</span></div>
</div>

<div class="card"><h2>2 · Control</h2>
  <div class="row">
    <span class="seg" id="modes">
      <button data-m="emg" onclick="setMode('emg')" id="mEmg">EMG</button>
      <button data-m="imu" onclick="setMode('imu')" id="mImu">EMG + Tilt</button>
      <button data-m="manual" onclick="setMode('manual')">Manual</button></span>
    <span class="small" style="flex:1" id="modeHelp"></span>
    <label class="small">Speed <input type="range" id="speed" min="0.1" max="1.5" step="0.05" style="width:110px" oninput="post({cmd:'speed',value:this.value})"></label>
  </div>
  <div class="gesture" id="gesture">Rest</div>
  <div id="emgBox">
    <div class="bars">
      <div><div class="small">Flex <span id="vA"></span></div><div class="bar"><i id="barA" style="background:var(--fa)"></i><b id="thA"></b></div></div>
      <div><div class="small">Extend <span id="vB"></span></div><div class="bar"><i id="barB" style="background:var(--fb)"></i><b id="thB"></b></div></div>
    </div>
    <div class="ch" id="chs"></div>
    <div class="row" style="margin-top:10px">
      <span class="small">Calibrate:</span>
      <button onclick="cal('rest')">1 Rest</button><button onclick="cal('flex')">2 Flex</button><button onclick="cal('ext')">3 Extend</button>
      <span class="small" id="calMsg"></span></div>
  </div>
  <div id="gestBox" class="hide">
    <div class="bars"><div style="grid-column:1/3"><div class="small">Prediction <span id="gLab"></span> <span id="gConf"></span></div>
      <div class="bar"><i id="gBar" style="background:var(--acc)"></i></div></div></div>
    <details><summary>Gesture → arm action</summary><div class="gm" id="gmap"></div>
      <div class="row" style="margin-top:8px"><label class="small">Min confidence <input type="range" id="minConf" min="0" max="100" step="5" style="width:130px" oninput="post({cmd:'min_conf',value:this.value});$('mcv').textContent=this.value+'%'"> <span id="mcv"></span></label></div></details>
  </div>
  <div id="joints" style="margin-top:8px"></div>
  <div class="row" style="margin-top:8px">
    <button class="pri" onclick="post({cmd:'home'})">⌂ Home</button>
    <button onclick="post({cmd:'gripper',state:'open'})">Open gripper</button>
    <button onclick="post({cmd:'gripper',state:'close'})">Close gripper</button>
    <button onclick="post({cmd:'select',joint:nextJ()})">Next joint ⇥</button>
  </div>
</div>

<details class="card"><summary>More: poses, tilt, simulator, logging, diagnostics</summary>
  <div class="row"><span class="lbl">Poses</span><span id="poses"></span></div>
  <div class="row"><span class="lbl">Tilt</span><button onclick="post({cmd:'zero_imu'})">Zero tilt</button>
    <label class="small"><input type="checkbox" id="swap" onchange="post({cmd:'swap',value:this.checked})"> swap flex/extend channel groups</label></div>
  <div class="row"><span class="lbl">Logging</span><button id="logBtn" onclick="post({cmd:'log',on:!S.logging})">Record CSV</button><span class="small">saved in ./logs</span></div>
  <div id="simBox"><div class="row"><span class="lbl">Sim band</span>
    <button id="sF" class="hold" data-k="f">Hold: Flex (F)</button><button id="sE" class="hold" data-k="e">Hold: Extend (E)</button><button id="sB" class="hold" data-k="b">Hold: Both (B)</button></div>
    <div class="row"><span class="lbl"></span><label class="small">roll <input type="range" min="-60" max="60" value="0" oninput="post({cmd:'sim',roll:this.value})"></label>
    <label class="small">pitch <input type="range" min="-60" max="60" value="0" oninput="post({cmd:'sim',pitch:this.value})"></label></div></div>
  <div class="row"><span class="lbl">Diagnose</span><button onclick="diag()">Run checks</button></div>
  <pre class="ev" id="diag" style="white-space:pre-wrap"></pre>
</details>

<div class="card"><h2>Log</h2><div class="ev" id="ev"></div></div>
<div class="small">Keys: Space = E-stop · H = home · Tab = next joint · 1-5 = select joint · ←/→ jog selected joint · F/E/B = simulated armband</div>
</div>
<script>
let S=null, drag=null, inited=false;
const $=id=>document.getElementById(id);
function post(o){return fetch('/api/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)}).catch(()=>{})}
function estop(){post({cmd:S&&S.estop?'reset':'estop'})}
function setMode(m){post({cmd:'mode',mode:m})}
function cal(k){post({cmd:'cal',kind:k})}
function nextJ(){const ids=S.joints.map(j=>j.id).filter(i=>S.mode!=='imu'||(i!==6&&i!==5));return ids[(ids.indexOf(S.active)+1)%ids.length]||ids[0]}
const bandHelp={bridge:'Start BioWave main.py → load your model → Realtime prediction. This app then follows its gestures (UDP 5005). Keep BioWave app open.',
 wifi:'Close the BioWave app first (only one program can stream). Scan, pick the device, enter its access key.',
 serial:'Close the BioWave app first (it holds the USB port). Baud 921600.',myo:'Needs the Myo BLE dongle and pyomyo.',sim:'Virtual armband for testing (keys F / E / B).'};
function kindUI(){
  const a=$('armKind').value,b=$('bandKind').value;
  $('armPort').style.display=a==='serial'?'':'none';
  $('rowWifi').classList.toggle('hide',b!=='wifi');$('rowUsb').classList.toggle('hide',b!=='serial');
  $('bandHelp').textContent=bandHelp[b];
}
async function refreshPorts(){const d=await (await fetch('/api/diag')).json();fillPorts(d.ports)}
function fillPorts(ports){
  const o=ports.map(p=>`<option>${p}</option>`).join('');
  $('armPort').innerHTML=o||'<option value="">no ports</option>';
  $('bandPort').innerHTML='<option value="auto">Auto-detect</option>'+o;
  if(S&&S.saved.band&&S.saved.band.port)$('bandPort').value=S.saved.band.port;
}
function connArm(){const c=S&&S.arm.status==='connected';post(c?{cmd:'disconnect_arm'}:{cmd:'connect_arm',kind:$('armKind').value,port:$('armPort').value})}
function connBand(){const c=S&&S.band.status==='connected';post(c?{cmd:'disconnect_band'}:{cmd:'connect_band',kind:$('bandKind').value,port:$('bandPort').value,channels:$('bandCh').value,ip:$('wifiDev').value,key:$('wifiKey').value})}
function connAll(){post({cmd:'connect_arm',kind:$('armKind').value,port:$('armPort').value});connBand2()}
function connBand2(){post({cmd:'connect_band',kind:$('bandKind').value,port:$('bandPort').value,channels:$('bandCh').value,ip:$('wifiDev').value,key:$('wifiKey').value})}
async function diag(){const d=await (await fetch('/api/diag')).json();
  $('diag').textContent=['python: '+d.python,'xarm package: '+d.xarm,'hidapi package: '+d.hid,'pyserial package: '+d.serial,'pyomyo package: '+d.pyomyo,
   'xArm USB devices seen: '+(d.xarm_usb_devices===null?'n/a (hidapi missing)':d.xarm_usb_devices),'serial ports: '+(d.ports.join(', ')||'none')].join('\n');fillPorts(d.ports)}
function buildJoints(){
  $('joints').innerHTML=S.joints.map(j=>`<div class="joint" id="j${j.id}">
   <div class="nm" onclick="post({cmd:'select',joint:${j.id}})">${j.name}<small id="jv${j.id}"></small></div>
   <input type="range" id="js${j.id}" min="${j.min}" max="${j.max}">
   <span class="jog"><button data-j="${j.id}" data-d="-1">−</button><button data-j="${j.id}" data-d="1">+</button></span>
   <label class="small inv"><input type="checkbox" id="ji${j.id}" onchange="post({cmd:'invert',joint:${j.id},value:this.checked})"> flip</label></div>`).join('');
  S.joints.forEach(j=>{const s=$('js'+j.id);
    s.addEventListener('pointerdown',()=>drag=j.id);addEventListener('pointerup',()=>drag=null);
    s.addEventListener('input',()=>post({cmd:'goal',joint:j.id,value:s.value}));});
  document.querySelectorAll('.jog button').forEach(b=>{
    let t=null;const j=+b.dataset.j,d=+b.dataset.d;
    const go=()=>{post({cmd:'select',joint:j});const f=()=>post({cmd:'jog',joint:j,dir:d});f();t=setInterval(f,80)};
    const end=()=>{if(t){clearInterval(t);t=null;post({cmd:'jog',joint:j,dir:0})}};
    b.addEventListener('pointerdown',go);b.addEventListener('pointerup',end);b.addEventListener('pointerleave',end);});
}
function pill(id,t,st,txt){$(id).className='pill '+st;$(t).textContent=txt}
const helps={emg:'Flex = + · Extend = − · both = next joint',imu:'Tilt arm: roll→Base, pitch→Reach · EMG drives the rest',manual:'Armband ignored'};
let gsig='';
function renderGmap(s){
  const sig=JSON.stringify(s.gmap)+JSON.stringify(s.actions);
  if(sig===gsig||document.activeElement&&document.activeElement.classList.contains('gsel'))return;gsig=sig;
  $('gmap').innerHTML=s.gmap.map(g=>`<div><span>${g.label}</span><select class="gsel" onchange="post({cmd:'gmap',label:'${g.key}',action:this.value})">${s.actions.map(a=>`<option value="${a.v}" ${a.v===g.action?'selected':''}>${a.l}</option>`).join('')}</select></div>`).join('');
}
function initFromSaved(s){
  const sv=s.saved||{};
  if(sv.arm_kind)$('armKind').value=sv.arm_kind;
  if(sv.band){$('bandKind').value=sv.band.kind;$('bandCh').value=sv.band.nch||8}
  kindUI();
  if(s.speed)$('speed').value=s.speed;
  $('minConf').value=s.gest.min_conf;$('mcv').textContent=Math.round(s.gest.min_conf)+'%';
  fillPorts([]);refreshPorts();
}
let devSig='';
function render(s){
  if(!S){S=s;buildJoints();initFromSaved(s)}S=s;
  pill('pArm','tArm',s.arm.status,'Arm: '+s.arm.status+(s.arm.battery?' · '+s.arm.battery.toFixed(1)+'V':''));
  pill('pBand','tBand',s.band.status,'Armband: '+s.band.status+(s.band.status==='connected'?' · '+s.band.rate+(s.band.gestures?' msg/s':' Hz'):''));
  $('bArm').textContent=s.arm.status==='connected'?'Disconnect':'Connect';
  $('bBand').textContent=s.band.status==='connected'?'Disconnect':'Connect';
  $('mArm').textContent=s.arm.msg;$('mBand').textContent=s.band.msg;
  const e=$('estop');e.textContent=s.estop?'RESET E-STOP':'E-STOP';e.className=s.estop?'latched':'';
  document.querySelectorAll('#modes button').forEach(b=>b.classList.toggle('on',b.dataset.m===s.mode));
  const g=s.band.gestures;
  $('emgBox').classList.toggle('hide',g);$('gestBox').classList.toggle('hide',!g);$('mImu').classList.toggle('hide',g);
  $('modeHelp').textContent=g?'Gestures from BioWave app drive the arm':helps[s.mode];
  $('gesture').textContent=s.gesture;
  if(g){$('gLab').textContent=s.gest.online?s.gest.label:'(none)';$('gConf').textContent=s.gest.online?s.gest.conf+'%':'';$('gBar').style.width=(s.gest.online?s.gest.conf:0)+'%';renderGmap(s)}
  else{
    $('barA').style.width=(s.a*100)+'%';$('barB').style.width=(s.b*100)+'%';
    $('thA').style.left=(s.thr[1]*100)+'%';$('thB').style.left=(s.thr[1]*100)+'%';
    $('vA').textContent=Math.round(s.a*100)+'%';$('vB').textContent=Math.round(s.b*100)+'%';
    const ch=$('chs');if(ch.children.length!==s.norm.length){ch.innerHTML=s.norm.map(()=>'<i></i>').join('')}
    s.norm.forEach((v,i)=>ch.children[i].style.height=Math.max(2,v*34)+'px');
    $('calMsg').textContent=s.cal.busy?s.cal.msg+' '+Math.round(s.cal.prog*100)+'%':s.cal.msg;}
  s.joints.forEach(j=>{
    $('j'+j.id).classList.toggle('active',j.id===s.active);
    $('jv'+j.id).textContent=j.pos;
    if(drag!==j.id)$('js'+j.id).value=j.goal;
    const c=$('ji'+j.id);if(document.activeElement!==c)c.checked=j.invert;});
  $('swap').checked=s.swap;
  $('logBtn').classList.toggle('on',s.logging);
  $('simBox').style.display=s.band.kind==='sim'?'':'none';
  $('bScan').textContent=s.wifi.scanning?'Scanning…':'Scan';
  const ds=JSON.stringify(s.wifi.devices);
  if(ds!==devSig){devSig=ds;const sel=s.wifi.devices.length?s.wifi.devices.map(d=>`<option value="${d.ip}">${d.name} (${d.ip})${d.streaming?' · streaming':''}</option>`).join(''):'<option value="">no devices - press Scan</option>';
    $('wifiDev').innerHTML=sel;if(!s.wifi.devices.length&&s.saved.band&&s.saved.band.ip)$('wifiDev').innerHTML=`<option value="${s.saved.band.ip}">${s.saved.band.ip} (saved)</option>`}
  if(s.wifi.has_key&&!$('wifiKey').value&&document.activeElement!==$('wifiKey'))$('wifiKey').placeholder='key saved - leave blank';
  $('poses').innerHTML=[1,2,3].map(n=>`<button onclick="post({cmd:'pose_save',slot:${n}})">Save ${n}</button><button ${s.poses.includes(''+n)?'':'disabled'} onclick="post({cmd:'pose_load',slot:${n}})">Go ${n}</button> `).join('');
  $('ev').innerHTML=s.events.map(x=>'<div>'+x.replace(/</g,'&lt;')+'</div>').join('');
}
async function poll(){try{render(await (await fetch('/api/state')).json())}catch(e){}setTimeout(poll,100)}
poll();kindUI();
const held={};
function simKey(k,down){const m={f:[1,0],e:[0,1],b:[1,1]}[k];if(!m)return;
  post({cmd:'sim',a:down?m[0]:0,b:down?m[1]:0});
  document.querySelector(`.hold[data-k=${k}]`).classList.toggle('on',down)}
document.querySelectorAll('.hold').forEach(b=>{
  b.addEventListener('pointerdown',()=>simKey(b.dataset.k,true));
  b.addEventListener('pointerup',()=>simKey(b.dataset.k,false));
  b.addEventListener('pointerleave',()=>simKey(b.dataset.k,false));});
addEventListener('keydown',ev=>{
  const t=ev.target.tagName;if(t==='INPUT'&&ev.target.type!=='range'||t==='SELECT')return;
  if(!S||ev.repeat&&!['ArrowLeft','ArrowRight'].includes(ev.key))return;
  const k=ev.key;
  if(k===' '){ev.preventDefault();estop()}
  else if(k==='h')post({cmd:'home'});
  else if(k==='Tab'){ev.preventDefault();post({cmd:'select',joint:nextJ()})}
  else if(/^[1-5]$/.test(k))post({cmd:'select',joint:S.joints[+k-1].id});
  else if(k==='ArrowLeft'||k==='ArrowRight'){ev.preventDefault();post({cmd:'jog',joint:S.active,dir:k==='ArrowRight'?1:-1})}
  else if(S.band.kind==='sim'&&'feb'.includes(k)&&!held[k]){held[k]=1;simKey(k,true)}});
addEventListener('keyup',ev=>{const k=ev.key;
  if(held[k]){held[k]=0;simKey(k,false)}});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    ctl = None
    allowed_hosts = ()

    def log_message(self, *a):
        pass

    def _ok_host(self):
        return self.headers.get("Host", "") in self.allowed_hosts      # blocks DNS rebinding

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._ok_host():
            return self._send(403, "{}")
        if self.path == "/api/state":
            return self._send(200, json.dumps(self.ctl.state()))
        if self.path == "/api/diag":
            return self._send(200, json.dumps(self.ctl.diagnostics()))
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        self._send(404, "{}")

    def do_POST(self):
        # JSON content-type forces a CORS preflight from other sites, which we never answer.
        if (not self._ok_host() or self.path != "/api/cmd"
                or "application/json" not in self.headers.get("Content-Type", "")):
            return self._send(403, "{}")
        try:
            n = int(self.headers.get("Content-Length", 0))
            self.ctl.command(json.loads(self.rfile.read(n) or b"{}"))
            self._send(200, "{}")
        except Exception as e:
            self._send(400, json.dumps({"error": str(e)}))


def main():
    ap = argparse.ArgumentParser(description="BioWave EMG armband -> xArm controller")
    ap.add_argument("--arm", help="sim | usb | COMx   (default: last used, else usb)")
    ap.add_argument("--band", help="bridge | wifi:<ip> | serial[:<port>] | myo | sim   (default: last used, else bridge)")
    ap.add_argument("--key", help="BioWave device access key (or env BIOWAVE_KEY)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-autoconnect", action="store_true", help="do not connect at startup")
    args = ap.parse_args()

    ctl = Controller()
    if args.key:
        ctl.key = args.key
    threading.Thread(target=ctl.run, daemon=True).start()

    sv = ctl.saved
    if args.arm or (sv.get("arm_kind") and not args.no_autoconnect):
        a = args.arm or sv["arm_kind"]
        if a in ("sim", "usb"):
            ctl.connect_arm(a, user=bool(args.arm))
        else:
            ctl.connect_arm("serial", a if args.arm else sv.get("arm_port", ""), user=bool(args.arm))
    if args.band or (sv.get("band") and not args.no_autoconnect):
        if args.band:
            kind, _, val = args.band.partition(":")
            ctl.connect_band(kind, port=val if kind == "serial" else "", ip=val if kind == "wifi" else "")
        else:
            c = sv["band"]
            ctl.connect_band(c["kind"], c.get("port", ""), c.get("baud", BW_USB_BAUD), c.get("nch", 8),
                             c.get("ip", ""), user=False)

    Handler.ctl = ctl
    Handler.allowed_hosts = ("127.0.0.1:%d" % args.port, "localhost:%d" % args.port)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://127.0.0.1:%d" % args.port
    print("EMG arm control running at", url, " (Ctrl+C to quit)  python:", sys.executable)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.disconnect_band()          # sends STOP to the armband when streaming over Wi-Fi
        time.sleep(0.3)
        ctl._set_log(False)


if __name__ == "__main__":
    main()
