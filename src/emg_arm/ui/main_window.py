"""Main window: step-by-step navigation, always-visible STOP button, and the control logic that ties
armband -> model -> gesture filter -> motion streamer -> arm together."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QFrame, QHBoxLayout, QLineEdit, QMainWindow, QMessageBox,
                             QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from .. import __version__
from ..arm.driver import ArmSafetyError, SafeArm, SimulatedBackend, XArmBackend
from ..arm.motion import MotionStreamer
from ..arm.teleop import GestureFilter, Teleop, canonical_gesture
from ..armband.link import LinkState
from ..ml.inference import InferenceWorker, ModelError, load_model
from ..paths import LOG_DIR
from ..session import RawRecorder, SessionLog
from . import theme
from .bridge import ArmbandBridge
from .pages.arm_page import ArmPage
from .pages.armband_page import ArmbandPage
from .pages.control_page import ControlPage
from .pages.data_page import DataPage
from .pages.model_page import ModelPage
from .widgets import Chip, Dot, button, label

SAMPLE_RATE = 500

KEY_MOTIONS = {Qt.Key_Left: "left", Qt.Key_Right: "right", Qt.Key_Up: "up", Qt.Key_Down: "down",
               Qt.Key_W: "shoulder_up", Qt.Key_S: "shoulder_down", Qt.Key_E: "elbow_up", Qt.Key_D: "elbow_down"}
KEY_NAMES = {Qt.Key_Left: "←", Qt.Key_Right: "→", Qt.Key_Up: "↑", Qt.Key_Down: "↓",
             Qt.Key_W: "W", Qt.Key_S: "S", Qt.Key_E: "E", Qt.Key_D: "D"}
STOP_KEYS = (Qt.Key_Space, Qt.Key_Escape)

PAGES = [
    ("model", "1  Model", "Choose the gesture model",
     "The trained Random Forest that recognises your gestures from the EMG signal."),
    ("armband", "2  Armband", "Connect the armband",
     "Four short steps: Wi-Fi once, find it, enter the key, connect. Then check the live signal."),
    ("arm", "3  Robotic arm", "Connect the robotic arm",
     "Connect the xArm 1S (or simulate it) and send it to its home pose."),
    ("control", "4  Control", "Control the arm",
     "Arm the robot, then move it with gestures or the keyboard. Stop at any time with Space or the red button."),
    ("data", "5  Data && settings", "Data & settings",
     "Record raw signals for analysis, find your session logs, and choose the appearance."),
]


def console(message):
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass


def xarm_plugged_in():
    try:
        import hid
        return any(d["vendor_id"] == 0x0483 and d["product_id"] == 0x5750 for d in hid.enumerate())
    except Exception:
        return False


class KeyboardFilter(QObject):
    """App-wide key handling so STOP works whichever widget has focus."""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype in (QEvent.KeyPress, QEvent.KeyRelease):
            armed = bool(self.window.teleop and self.window.teleop.armed)
            if QApplication.activeWindow() is not self.window:
                return False
            if not armed and isinstance(QApplication.focusWidget(), QLineEdit):
                return False
            return self.window.handle_key(event, etype == QEvent.KeyPress)
        if etype == QEvent.WindowDeactivate:
            self.window.pressed_keys.clear()
        return False


class MainWindow(QMainWindow):
    def __init__(self, config: dict, settings):
        super().__init__()
        self.config, self.settings = config, settings
        self.setWindowTitle("EMG Robotic Arm - BioWave research console")
        self.resize(1240, 860)
        self.setMinimumSize(980, 680)

        # ---- model / armband state
        self.loaded = None
        self.model_loaded = False
        self.num_channels = 11
        self.window_samples = 100
        self.stride_samples = 25
        self.sample_count = 0
        self.data_buffer = None
        self.baseline = None
        self.last_batch_t = 0.0
        self.link_state = LinkState.IDLE
        self.bridge = ArmbandBridge(self)
        self.recorder = None
        self.inference = InferenceWorker(SAMPLE_RATE)
        self.inference.prediction_ready.connect(self.on_prediction)
        self.inference.error.connect(lambda m: self.pages["control"].note_label.setText(m))
        self.inference.start()

        # ---- arm state
        self.arm = None
        self.streamer = None
        self.gesture_filter = GestureFilter(config["filter"])
        self.teleop = None
        self.home_busy_until = 0.0
        self.parking = None
        self.park_aborted = False
        self.close_ready = False
        self.pressed_keys = {}
        self.pad_held = {}
        self.key_grip_pending = False
        self.last_motion_t = 0.0
        self.session_log = None

        self._build()
        self.bridge.frames.connect(self.on_batch)
        self.bridge.state.connect(self.on_link_state)

        self.control_timer = QTimer(self)
        self.control_timer.setTimerType(Qt.PreciseTimer)
        self.control_timer.setInterval(int(config["arm"]["control_period_ms"]))
        self.control_timer.timeout.connect(self.control_tick)
        self.control_timer.start()
        self.battery_timer = QTimer(self)
        self.battery_timer.setInterval(10000)
        self.battery_timer.timeout.connect(self.check_battery)
        self.battery_timer.start()
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(500)
        self.record_timer.timeout.connect(self._update_recording)

        self.go_to("model")
        last = settings.get("last_model")
        if last and Path(last).exists():
            QTimer.singleShot(200, lambda: self.load_model_path(last, quiet=True))
        self.refresh_status()

    # ================================================================== layout
    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        outer.addLayout(body, 1)

        # ---- sidebar
        side = QFrame()
        theme.set_role(side, "sidebar")
        side.setFixedWidth(230)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(16, 20, 16, 16)
        sl.setSpacing(4)
        sl.addWidget(label("BioWave", "h1"))
        sl.addWidget(label("EMG Robotic Arm", "muted"))
        sl.addSpacing(18)
        sl.addWidget(label("SETUP STEPS", "caption"))
        self.nav_group = QButtonGroup(self)
        self.nav_buttons, self.nav_dots = {}, {}
        for key, nav_text, _t, _l in PAGES:
            row = QHBoxLayout()
            row.setSpacing(0)
            b = QPushButton(nav_text)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            theme.set_role(b, "nav")
            b.clicked.connect(lambda _=False, k=key: self.go_to(k))
            self.nav_group.addButton(b)
            self.nav_buttons[key] = b
            dot = Dot()
            self.nav_dots[key] = dot
            row.addWidget(b, 1)
            row.addWidget(dot)
            row.addSpacing(6)
            sl.addLayout(row)
        sl.addStretch(1)
        sl.addWidget(label(f"Version {__version__}", "caption"))
        body.addWidget(side)

        # ---- content
        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(36, 26, 36, 8)
        hl.setSpacing(4)
        self.title = label("", "h1")
        self.lead = label("", "lead", wrap=True)
        hl.addWidget(self.title)
        hl.addWidget(self.lead)
        content.addWidget(head)

        self.stack = QStackedWidget()
        self.pages = {}
        simulate_default = not xarm_plugged_in()
        self.pages["model"] = ModelPage(self.settings)
        self.pages["armband"] = ArmbandPage(self.bridge, self.settings)
        self.pages["arm"] = ArmPage(simulate_default)
        self.pages["control"] = ControlPage(self.config, self.gesture_filter.min_conf)
        self.pages["data"] = DataPage(self.settings.get("theme"))
        for key, *_ in PAGES:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            host = QWidget()
            hbox = QHBoxLayout(host)
            hbox.setContentsMargins(36, 12, 36, 28)
            hbox.addStretch(1)
            self.pages[key].setMaximumWidth(960)
            hbox.addWidget(self.pages[key], 100)
            hbox.addStretch(1)
            scroll.setWidget(host)
            self.stack.addWidget(scroll)
        content.addWidget(self.stack, 1)
        body.addLayout(content, 1)

        # ---- bottom bar (always visible)
        bar = QFrame()
        theme.set_role(bar, "bottombar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(24, 10, 24, 10)
        bl.setSpacing(10)
        self.chip_band = Chip("Armband: offline")
        self.chip_arm = Chip("Arm: offline")
        self.chip_mode = Chip("DISARMED")
        for c in (self.chip_band, self.chip_arm, self.chip_mode):
            bl.addWidget(c)
        self.chip_note = label("", "muted")
        bl.addWidget(self.chip_note, 1)
        self.next_btn = button("Next step →", "primary", slot=self.go_next)
        bl.addWidget(self.next_btn)
        estop = button("STOP   (Space / Esc)", "estop", slot=lambda: self.emergency_stop("Stop button pressed"))
        estop.setMinimumHeight(46)
        bl.addWidget(estop)
        outer.addWidget(bar)

        # ---- wiring
        mp, ap, rp, cp, dp = (self.pages[k] for k in ("model", "armband", "arm", "control", "data"))
        mp.model_requested.connect(self.load_model_path)
        ap.link_changed.connect(lambda _s: self.refresh_status())
        rp.connect_toggled.connect(self.toggle_arm)
        rp.home_requested.connect(self.go_home)
        rp.gripper_requested.connect(self.manual_gripper)
        cp.arm_toggled.connect(self.toggle_armed)
        cp.pad_pressed.connect(lambda m: self.pad_held.setdefault(m, time.monotonic()))
        cp.pad_released.connect(lambda m: self.pad_held.pop(m, None))
        cp.grip_pressed.connect(lambda: setattr(self, "key_grip_pending", True))
        cp.vertical.currentIndexChanged.connect(self.on_vertical_changed)
        cp.speed.valueChanged.connect(lambda v: self.arm and self.arm.set_speed(v))
        cp.conf.valueChanged.connect(lambda v: setattr(self.gesture_filter, "min_conf", v / 100))
        dp.record_toggled.connect(self.toggle_recording)
        dp.theme_changed.connect(self.set_theme)
        self.on_vertical_changed()

    # ================================================================ navigation
    def go_to(self, key: str):
        keys = [p[0] for p in PAGES]
        idx = keys.index(key)
        self.stack.setCurrentIndex(idx)
        self.nav_buttons[key].setChecked(True)
        _k, _n, title, lead = PAGES[idx]
        self.title.setText(title)
        self.lead.setText(lead)
        self.current_key = key
        self.next_btn.setVisible(idx < len(keys) - 1)
        self.refresh_status()

    def go_next(self):
        keys = [p[0] for p in PAGES]
        self.go_to(keys[min(len(keys) - 1, keys.index(self.current_key) + 1)])

    def set_theme(self, mode: str):
        self.settings.set("theme", mode)
        theme.apply_theme(QApplication.instance(), mode)
        self.pages["armband"].restyle()
        theme.apply_title_bar(self)
        self.refresh_status()

    def refresh_status(self):
        s = self.link_state
        if s == LinkState.STREAMING:
            self.chip_band.set(f"Armband: streaming ({self.bridge.link.stats.transport if self.bridge.link else ''})", "good")
        elif s in (LinkState.CONNECTING, LinkState.RECONNECTING, LinkState.STALLED):
            self.chip_band.set("Armband: " + s.value + "…", "warn")
        elif s == LinkState.FAILED:
            self.chip_band.set("Armband: failed", "bad")
        else:
            self.chip_band.set("Armband: offline", "none")
        if self.arm:
            sim = isinstance(self.arm.backend, SimulatedBackend)
            self.chip_arm.set(f"Arm: {'simulated' if sim else 'real'}{', home' if self.arm.homed else ', not homed'}",
                              "good" if self.arm.homed else "warn")
        else:
            self.chip_arm.set("Arm: offline", "none")
        armed = bool(self.teleop and self.teleop.armed)
        if self.parking:
            self.chip_mode.set("PARKING", "warn")
        else:
            self.chip_mode.set("ARMED" if armed else "DISARMED", "bad" if armed else "none")

        tones = {"model": "good" if self.model_loaded else "none",
                 "armband": {LinkState.STREAMING: "good", LinkState.FAILED: "bad", LinkState.IDLE: "none"}.get(s, "warn"),
                 "arm": ("good" if self.arm and self.arm.homed else "warn") if self.arm else "none",
                 "control": "accent" if armed else "none", "data": "accent" if self.recorder else "none"}
        for key, tone in tones.items():
            self.nav_dots[key].set_tone(tone)

        cp = self.pages["control"]
        missing = []
        if not self.model_loaded:
            missing.append("load a model (step 1)")
        if s != LinkState.STREAMING:
            missing.append("connect the armband (step 2) - or use the keyboard only")
        if not self.arm:
            missing.append("connect the arm (step 3)")
        elif not self.arm.homed:
            missing.append("send the arm home (step 3)")
        if not self.arm and s == LinkState.STREAMING and self.model_loaded:
            cp.checklist.setText("Armband connected, no arm: PREVIEW mode. The Live panel shows what each gesture would do "
                                 "(nothing moves). Connect the arm (step 3) or tick “Simulate” to move it.")
        else:
            cp.checklist.setText("Before arming: " + "; ".join(missing) + "." if missing else "Everything is ready.")
        self.update_controls()

    # ================================================================== model
    def load_model_path(self, path: str, quiet: bool = False):
        mp = self.pages["model"]
        try:
            loaded = load_model(path)
        except ModelError as exc:
            if not quiet:
                mp.show_error(str(exc))
            return
        classes = loaded.class_names
        mapping = {c: canonical_gesture(c) for c in classes}
        problems = []
        unknown = [c for c, g in mapping.items() if g is None]
        missing = [g for g in ("left", "right", "up", "down", "grip", "rest") if g not in mapping.values()]
        if "rest" in missing:
            problems.append("⚠ This model has no Rest class, so it can never say “do nothing”. Use a model trained with Rest.")
        elif missing:
            problems.append("Not in this model: " + ", ".join(missing))
        if unknown:
            problems.append("Ignored classes: " + ", ".join(unknown))
        if self.bridge.active:
            self.bridge.disconnect()                       # buffer sizes change with the model
        self.disarm("Model changed")
        self.gesture_filter.switch_hold_s = 0.0 if "switch" in mapping.values() \
            else float(self.config["filter"].get("switch_hold_s", 0.0))
        self.window_samples, self.stride_samples = loaded.window_samples, loaded.stride_samples
        self.num_channels = loaded.input_channels
        self.data_buffer = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
        self.baseline = np.zeros(self.num_channels, dtype=np.float32)
        self.sample_count = 0
        self.inference.load_model(loaded.model, classes)
        self.gesture_filter.reset()
        self.loaded, self.model_loaded = loaded, True
        self.settings.set("last_model", path)
        mp.show_loaded(loaded, mapping, problems)
        self.refresh_status()

    # ================================================================ armband data
    def on_link_state(self, state_value: str, message: str):
        self.link_state = LinkState(state_value)
        if self.link_state in (LinkState.IDLE, LinkState.FAILED):
            if self.recorder:
                self.toggle_recording("")
            self.gesture_filter.reset()
        self.refresh_status()

    def on_batch(self, batch):
        if self.data_buffer is None:
            return
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 2 or not len(batch):
            return
        if batch.shape[1] < self.num_channels:
            batch = np.pad(batch, ((0, 0), (0, self.num_channels - batch.shape[1])))
        batch = batch[:, :self.num_channels]
        self.last_batch_t = time.monotonic()
        # Adaptive baseline centring, identical to the training pipeline.
        self.baseline *= 0.95
        self.baseline += 0.05 * np.mean(batch, axis=0, dtype=np.float32)
        incoming = (batch - self.baseline).T
        count = min(incoming.shape[1], self.window_samples)
        self.data_buffer[:, :-count] = self.data_buffer[:, count:]
        self.data_buffer[:, -count:] = incoming[:, -count:]
        self.sample_count += count
        if self.sample_count >= self.stride_samples:
            self.sample_count = 0
            self.inference.submit_window(self.data_buffer.copy())

    def on_prediction(self, label_text, confidence):
        now = time.monotonic()
        gesture = self.gesture_filter.add_prediction(label_text, confidence, now)
        self.pages["control"].raw_label.setText(
            f"Model output: {label_text}  ({confidence * 100:.0f}%)  →  {gesture or 'ignored'}")
        self.log("prediction", label=label_text, confidence=f"{confidence:.3f}", gesture=gesture or "")

    # ================================================================ recording
    def toggle_recording(self, tag: str):
        dp = self.pages["data"]
        if self.recorder:
            self.bridge.recorder = None
            rec, self.recorder = self.recorder, None
            rec.close()
            self.record_timer.stop()
            dp.set_recording(False, str(rec.path), rec.frames)
        else:
            if not self.bridge.active:
                dp.record_status.set("Connect the armband first (step 2).", "warn")
                return
            self.recorder = RawRecorder(label=tag)
            self.bridge.recorder = self.recorder
            self.record_timer.start()
            dp.set_recording(True, str(self.recorder.path), 0)
        self.refresh_status()

    def _update_recording(self):
        if self.recorder:
            self.pages["data"].set_recording(True, str(self.recorder.path), self.recorder.frames)

    # ===================================================================== arm
    def toggle_arm(self):
        if self.arm:
            self.disconnect_arm()
            return
        rp = self.pages["arm"]
        try:
            if rp.simulate.isChecked():
                start = dict(self.config["arm"]["home"])
                start[str(self.config["arm"]["gripper"]["servo"])] = self.config["arm"]["gripper"]["open"]
                backend = SimulatedBackend({int(k): int(v) for k, v in start.items()})
            else:
                backend = XArmBackend(self.config["arm"].get("connection", "USB"))
            arm = SafeArm(self.config, backend)
            streamer = MotionStreamer(arm, self.config["arm"])
        except ArmSafetyError as exc:
            QMessageBox.critical(self, "Config error", f"The arm configuration failed a safety check:\n{exc}")
            return
        except ImportError as exc:
            QMessageBox.critical(self, "Arm library missing", f"{exc}\n\nInstall it with:  pip install -r requirements.txt")
            return
        except Exception as exc:
            rp.banner.set(f"Could not open the xArm over USB: {exc}\nCheck the USB cable, that the arm's power switch "
                          "is ON, and that no other program is using the arm.", "bad")
            return
        positions = arm.read_positions()
        ok, voltage = arm.battery()
        arm.set_speed(self.pages["control"].speed.value())
        self.arm, self.streamer = arm, streamer
        streamer.start()
        self.teleop = Teleop(arm, self.gesture_filter, self.config, streamer)
        self.teleop.vertical = self.pages["control"].vertical.currentData()
        self.open_log()
        unknown = [str(s) for s, p in positions.items() if p is None]
        volt = f"{voltage:.2f} V" if voltage else "voltage n/a"
        rp.banner.set(f"Connected: {backend.name}  ·  {volt}.  Next: send it home." + (
            f"\nCould not read servo(s) {', '.join(unknown)}: their first move will be extra slow." if unknown else ""),
            "good" if ok else "bad")
        rp.connect_btn.setText("Disconnect arm")
        theme.set_role(rp.connect_btn, "danger")
        rp.card_connect.set_done(True)
        self.refresh_servo_view()
        self.log("arm_connected", note=f"{backend.name}; positions={json.dumps(positions)}; voltage={voltage}")
        if not ok:
            QMessageBox.warning(self, "Low voltage", f"Arm supply reads {voltage:.2f} V (minimum {arm.min_battery} V).\n"
                                "Use the 7.5 V adapter / charge the battery before moving the arm.")
        self.refresh_status()

    def disconnect_arm(self):
        self.disarm("Arm disconnected")
        if self.streamer:
            self.streamer.stop()
            self.streamer = None
        if self.arm:
            self.arm.close()        # torque stays on: the arm keeps holding its pose
        self.arm = None
        self.teleop = None
        rp = self.pages["arm"]
        rp.banner.set("Arm disconnected.", "none")
        rp.connect_btn.setText("Connect arm")
        theme.set_role(rp.connect_btn, "success")
        rp.card_connect.set_done(False)
        rp.card_home.set_done(False)
        self.close_log()
        self.refresh_status()

    def go_home(self):
        if not self.arm:
            self.pages["arm"].banner.set("Connect the arm first (step 1).", "warn")
            return
        if QMessageBox.question(self, "Go home", "The arm will move slowly to its home pose.\nMake sure nothing and nobody "
                                "is within reach of the arm.\n\nContinue?") != QMessageBox.Yes:
            return
        self.disarm("Homing")
        try:
            seconds = self.arm.go_home()
        except Exception as exc:
            self.handle_arm_failure(exc)
            return
        self.home_busy_until = time.monotonic() + seconds + 0.3
        self.pages["arm"].banner.set(f"Homing… ({seconds:.1f} s)", "accent")
        self.log("home", note=f"{seconds:.1f}s")
        self.refresh_servo_view()
        QTimer.singleShot(int((seconds + 0.3) * 1000), self.after_home)
        self.refresh_status()

    def after_home(self):
        if self.arm and self.arm.homed:
            self.pages["arm"].banner.set("At home. Continue to step 4 to arm the robot.", "good")
            self.pages["arm"].card_home.set_done(True)
        self.refresh_status()

    def manual_gripper(self, closed):
        if not self.arm:
            return
        try:
            self.arm.set_gripper(closed)
            self.refresh_servo_view()
            self.log("gripper", note="closed" if closed else "open", source="button")
        except Exception as exc:
            self.handle_arm_failure(exc)

    def check_battery(self):
        if not self.arm or isinstance(self.arm.backend, SimulatedBackend):
            return
        if time.monotonic() - self.last_motion_t < 1.0 or (self.streamer and self.streamer.is_moving()):
            return
        ok, voltage = self.arm.battery()
        if not ok:
            self.emergency_stop(f"Arm voltage low ({voltage:.2f} V)")

    # ================================================================ arming
    def can_arm(self):
        return bool(self.arm and self.arm.homed and time.monotonic() >= self.home_busy_until)

    def toggle_armed(self):
        cp = self.pages["control"]
        if self.teleop and self.teleop.armed:
            self.disarm("Disarmed by user")
            return
        if not self.can_arm():
            QMessageBox.information(self, "Not ready", "Connect the arm and press “Go home” first (step 3).")
            self.go_to("arm")
            return
        if cp.emg_check.isChecked() and not (self.link_state == LinkState.STREAMING and self.model_loaded):
            if QMessageBox.question(self, "Gestures not available",
                                    "The armband is not streaming or no model is loaded, so EMG gestures cannot be used.\n\n"
                                    "ARM in keyboard-only mode?") != QMessageBox.Yes:
                return
            cp.emg_check.setChecked(False)
        self.gesture_filter.reset()
        self.pressed_keys.clear()
        self.streamer.clear_error()
        self.teleop.armed = True
        self.log("armed")
        self.refresh_status()

    def not_ready_reason(self):
        if not self.arm:
            return "Keys only move the arm when ARMED: first connect the arm (step 3)."
        if not self.arm.homed:
            return "Keys only move the arm when ARMED: send the arm home first (step 3), then ARM."
        if time.monotonic() < self.home_busy_until:
            return "Wait for homing to finish, then ARM."
        return "Press ARM (step 4) to enable motion, then use the keys."

    def disarm(self, reason=""):
        if self.streamer:
            try:
                self.streamer.halt()
            except Exception as exc:
                console(f"Halt failed: {exc}")
        if self.teleop and self.teleop.armed:
            self.teleop.armed = False
            self.log("disarmed", note=reason)
        self.pressed_keys.clear()
        self.pad_held.clear()
        self.key_grip_pending = False
        cp = self.pages["control"]
        cp.key_label.setText("Keyboard: idle")
        if reason:
            cp.note_label.setText(reason)
        self.refresh_status()

    def emergency_stop(self, reason):
        homing = time.monotonic() < self.home_busy_until
        parking = self.parking is not None
        self.parking = None
        if parking:
            self.park_aborted = True
            reason += ". Parking stopped, the arm holds here. Close again to park or exit"
        if self.arm and (homing or parking):
            try:
                self.arm.freeze()
            except Exception as exc:
                console(f"Freeze failed: {exc}")
            self.home_busy_until = 0.0
            self.refresh_servo_view()
        self.disarm(f"STOPPED: {reason}")
        self.pages["control"].action_label.setText("STOPPED")

    def handle_arm_failure(self, exc):
        if isinstance(exc, ArmSafetyError):
            self.disarm(str(exc))
            QMessageBox.warning(self, "Safety", str(exc))
        else:
            self.emergency_stop(f"Arm communication error: {exc}")
            QMessageBox.critical(self, "Arm error", f"{exc}\n\nMotion stopped. Check the USB cable and arm power, then "
                                 "disconnect and connect the arm again.")

    def on_vertical_changed(self):
        cp = self.pages["control"]
        if self.teleop:
            self.teleop.vertical = cp.vertical.currentData()
        cp.joint_label.setText("Up / Down moves the " + cp.vertical.currentText().upper())

    def update_controls(self):
        armed = bool(self.teleop and self.teleop.armed)
        connected = self.arm is not None
        homing = time.monotonic() < self.home_busy_until or self.parking is not None
        rp, cp = self.pages["arm"], self.pages["control"]
        rp.simulate.setEnabled(not connected)
        rp.home_btn.setEnabled(connected and not armed and not homing)
        for b in (rp.open_btn, rp.close_btn):
            b.setEnabled(connected and self.arm.homed and not homing)
        cp.arm_btn.setEnabled((armed or self.can_arm()) and self.parking is None)
        cp.arm_btn.setText("DISARM" if armed else "ARM - enable motion")
        theme.set_role(cp.arm_btn, "ghost" if armed else "success")
        simulated = connected and isinstance(self.arm.backend, SimulatedBackend)
        if self.parking:
            cp.state_banner.set("PARKING the arm, then closing (Space/Esc stops it).", "warn")
        elif armed and simulated:
            cp.state_banner.set("ARMED - simulation only, the real arm will not move.", "accent")
        elif armed:
            cp.state_banner.set("ARMED - the arm WILL move. Keep clear of it.", "bad")
        else:
            cp.state_banner.set("DISARMED - the arm will not move.", "none")
        cp.card_arm.set_done(armed)
        cp.set_pad_enabled(armed)

    # ============================================================== keyboard
    def handle_key(self, event, pressed):
        key = event.key()
        cp = self.pages["control"]
        if key in STOP_KEYS:
            if pressed:
                self.emergency_stop("Space/Esc pressed")
            return True
        armed = bool(self.teleop and self.teleop.armed)
        if not armed:
            if key in KEY_MOTIONS or key == Qt.Key_G:
                if pressed and not event.isAutoRepeat():
                    cp.key_label.setText(f"Keyboard: {KEY_NAMES.get(key, 'G')} ignored (disarmed)")
                    cp.note_label.setText(self.not_ready_reason())
                return True
            return False
        if event.isAutoRepeat():
            return key in KEY_MOTIONS
        if key in KEY_MOTIONS:
            if pressed:
                self.pressed_keys[key] = time.monotonic()
            else:
                self.pressed_keys.pop(key, None)
            cp.highlight_key(KEY_MOTIONS[key], pressed)
            held = [KEY_NAMES[k] for k in self.pressed_keys]
            cp.key_label.setText(f"Keyboard: {' '.join(held)} held" if held else "Keyboard: idle")
            return True
        if pressed and key == Qt.Key_G:
            self.key_grip_pending = True
            return True
        if pressed and key in (Qt.Key_1, Qt.Key_2):
            index = 0 if key == Qt.Key_1 else 1
            if index < cp.vertical.count():
                cp.vertical.setCurrentIndex(index)
            return True
        return False

    # ============================================================ control loop
    def control_tick(self):
        now = time.monotonic()
        if not (self.teleop and self.teleop.armed):
            self.preview_tick(now)
            return
        cp = self.pages["control"]
        if self.streamer.error is not None:
            exc, self.streamer.error = self.streamer.error, None
            self.handle_arm_failure(exc)
            return
        key_motions = {KEY_MOTIONS[k]: now - since for k, since in self.pressed_keys.items()}
        key_motions.update({m: now - since for m, since in self.pad_held.items()})
        key_grip, self.key_grip_pending = self.key_grip_pending, False
        emg_enabled = cp.emg_check.isChecked()
        emg_ok = (emg_enabled and self.link_state == LinkState.STREAMING
                  and now - self.last_batch_t <= float(self.config["filter"]["emg_timeout_s"]))
        try:
            status = self.teleop.tick(now, key_motions, key_grip, emg_ok)
        except Exception as exc:
            self.handle_arm_failure(exc)
            return
        note = status["note"]
        if not emg_enabled and (not note or note.startswith("EMG stream lost")):
            note = "Keyboard only"
        cp.note_label.setText(note)
        cp.grip_hint.setText(self.gesture_filter.grip_hint(now) if emg_ok else "")
        cp.action_label.setText(f"{status['motion'].upper()}  ({status['source']})" if status["motion"] else "HOLD")
        if status["switched"]:
            cp.vertical.setCurrentIndex(cp.vertical.findData(status["switched"]))
            self.log("switch_joint", note=status["switched"], source=status["source"])
        if status["gripper"]:
            self.log("gripper", note=status["gripper"], source=status["source"])
        if status["motion"] or status["gripper"]:
            self.last_motion_t = now
            self.log("move", motion=status["motion"] or "", source=status["source"] or "", note=note)
        if status["motion"] or status["gripper"] or self.streamer.is_moving():
            self.refresh_servo_view()

    def preview_tick(self, now):
        """Not armed (or no arm at all): show what the gesture WOULD do, without moving anything."""
        cp = self.pages["control"]
        streaming = self.link_state == LinkState.STREAMING and self.model_loaded
        if not streaming:
            return
        try:
            motion = self.gesture_filter.motion(now) if self.gesture_filter.is_fresh(now) else None
        except Exception:
            motion = None
        tag = "PREVIEW" if self.arm else "PREVIEW (no arm connected)"
        cp.action_label.setText(f"{tag}: {motion.upper()}" if motion else f"{tag}: rest")

    def refresh_servo_view(self):
        if not self.arm:
            return
        cp = self.pages["control"]
        targets = dict(self.arm.targets)
        for sid, bar in cp.servo_bars.items():
            pos = targets.get(sid)
            if pos is not None:
                bar.setValue(max(bar.minimum(), min(bar.maximum(), pos)))
        if self.arm.gripper_closed is not None:
            cp.gripper_label.setText(f"Gripper: {'CLOSED' if self.arm.gripper_closed else 'OPEN'}")

    # ================================================================= logging
    def open_log(self):
        self.session_log = SessionLog(LOG_DIR)
        self.pages["data"].set_log(str(self.session_log.path))

    def log(self, event, **fields):
        if self.session_log:
            self.session_log.write(event, targets=dict(self.arm.targets) if self.arm else None, **fields)

    def close_log(self):
        if self.session_log:
            self.session_log.close()
        self.session_log = None

    # =================================================================== close
    def closeEvent(self, event):
        if self.close_ready:
            self.shutdown()
            event.accept()
            return
        event.ignore()
        if self.parking:
            self.pages["control"].note_label.setText("Parking the arm before exit… (Space/Esc stops it)")
            return
        if not (self.arm and self.config["arm"].get("park_on_exit", True)):
            self.finish_close()
            return
        if self.park_aborted:
            answer = QMessageBox.question(self, "Exit", "Parking was stopped.\n\nYes: park the arm at home now, then exit.\n"
                                          "No: exit and leave the arm holding where it is.",
                                          QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.No:
                self.finish_close()
                return
        self.start_parking()

    def start_parking(self):
        self.parking = "home"
        self.park_aborted = False
        self.disarm("Closing: parking the arm at home first (Space/Esc stops it)")
        try:
            seconds = self.arm.go_home()
        except Exception as exc:
            self.log("park", note=f"failed: {exc}")
            console(f"Parking failed: {exc}")
            self.parking = None
            QMessageBox.critical(self, "Parking failed", f"Could not move the arm home before exiting:\n{exc}\n\n"
                                 "The app will close and the arm keeps holding its pose.")
            self.finish_close()
            return
        real = not isinstance(self.arm.backend, SimulatedBackend)
        wait = seconds + 0.3 if real else 0.2
        self.home_busy_until = time.monotonic() + wait
        self.log("park", note=f"home {seconds:.1f}s")
        self.refresh_servo_view()
        self.refresh_status()
        QTimer.singleShot(int(wait * 1000), self.park_open_gripper)

    def park_open_gripper(self):
        if self.parking != "home":
            return
        self.home_busy_until = 0.0
        if not (self.config["arm"].get("open_gripper_on_exit", True) and self.arm.gripper_closed is not False):
            self.park_done()
            return
        self.parking = "gripper"
        try:
            self.arm.set_gripper(False)
        except Exception as exc:
            console(f"Opening the gripper failed: {exc}")
            self.park_done()
            return
        self.log("park", note="gripper opened")
        self.refresh_servo_view()
        real = not isinstance(self.arm.backend, SimulatedBackend)
        travel_ms = abs(self.arm.grip_close - self.arm.grip_open) / self.arm.hard_max_speed * 1000
        QTimer.singleShot(int(max(self.arm.grip_ms, travel_ms)) + 200 if real else 100, self.park_done)

    def park_done(self):
        if self.parking is None:
            return
        self.parking = None
        errors = {}
        if not isinstance(self.arm.backend, SimulatedBackend):
            try:
                errors = self.arm.home_errors()
            except Exception as exc:
                console(f"Could not check the home pose: {exc}")
        if errors:
            detail = "\n".join(f"  {self.arm.names[sid]} (servo {sid}): at {pos if pos is not None else '?'}, home is {home}"
                               for sid, (pos, home) in sorted(errors.items()))
            self.log("park", note=f"not at home: {errors}")
            QMessageBox.warning(self, "Arm not fully parked", "The arm did not fully reach its home pose:\n" + detail +
                                "\n\nSomething may be blocking it. The app will now close.")
        else:
            self.log("park", note="at home")
        self.finish_close()

    def finish_close(self):
        self.close_ready = True
        QTimer.singleShot(0, self.close)     # not directly: Qt ignores close() while inside closeEvent

    def shutdown(self):
        self.control_timer.stop()
        self.battery_timer.stop()
        self.record_timer.stop()
        if self.recorder:
            self.recorder.close()
            self.recorder = None
        if self.streamer:
            self.streamer.stop()
        self.bridge.shutdown()
        if self.arm:
            self.disconnect_arm()
        self.inference.stop()
