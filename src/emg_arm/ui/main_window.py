"""Main window: top status bar with the always-visible STOP button, sidebar navigation, and the control logic that ties
armband -> model -> gesture filter -> motion streamer -> arm together."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QMainWindow, QMessageBox,
                             QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout,
                             QWidget)

from .. import __version__
from ..arm.driver import ArmSafetyError, SafeArm, SimulatedBackend, XArmBackend
from ..arm.motion import MotionStreamer
from ..arm.teleop import GestureFilter, Teleop
from ..app_log import record
from ..armband.link import LinkState
from ..config_bundle import effective_config, export_bundle, import_bundle
from ..control.commands import CommandGenerator
from ..control.gestures import BY_ID
from ..control.mapping import MappingError, MappingStore
from ..ml.inference import InferenceWorker, ModelError, load_model
from ..paths import LOG_DIR
from ..session import RawRecorder, SessionLog
from . import icons, theme
from .bridge import ArmbandBridge
from .keyboard import KeyboardController, KeyMap, build_actions
from .pages.armband_page import ArmbandPage
from .pages.control_page import ControlPage
from .pages.data_page import DataPage
from .pages.mapping_page import MappingPage
from .pages.model_page import ModelPage
from .pages.monitor_page import MonitorPage
from .widgets import Chip, Dot, button, label, section

SAMPLE_RATE = 500

# (key, sidebar text, page title, lead text, sidebar group)
PAGES = [
    ("control", "Arm control", "Arm control",
     "Connect, arm, then move with gestures, buttons or keys. Space stops at any time.", "ROBOTIC ARM"),
    ("model", "Gesture model", "Gesture model", "Pick the trained model that recognises your gestures.", "EMG & MODEL"),
    ("armband", "Armband", "Armband connection", "Set up Wi-Fi once, find it, enter the key, connect.", "EMG & MODEL"),
    ("monitor", "Live monitor", "Live monitor", "EMG signal, recognised gesture, confidence and transitions.", "EMG & MODEL"),
    ("mapping", "Gesture mapping", "Gesture mapping", "Choose what the arm does for each of the ten gestures.", "EMG & MODEL"),
    ("data", "Settings && data", "Settings & data", "Appearance, configuration files and arm calibration.", "SETTINGS"),
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


class MainWindow(QMainWindow):
    def __init__(self, config: dict, settings, settings_dir=None):
        super().__init__()
        self.base_config = config
        self.config = effective_config(config, settings)       # user calibration, never wider than shipped limits
        self.settings = settings
        folder = Path(settings_dir) if settings_dir else Path(settings.path).parent
        self.setWindowTitle("EMG Robotic Arm - BioWave research console")
        self.resize(1280, 880)
        self.setMinimumSize(1100, 700)   # widest page needs ~1100 px; 1366x768 laptops and MacBooks are wider

        # ---- shared services
        self.mapping = MappingStore(self.config, folder / "gesture_mapping.json")
        self.keymap = KeyMap(build_actions(self.config), folder / "keymap.json")
        for warning in (self.mapping.load_warning, self.keymap.load_warning):
            if warning:
                record("warning", "app", warning)

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
        self.last_link_message = ""
        self.bridge = ArmbandBridge(self)
        self.recorder = None
        self.inference = InferenceWorker(SAMPLE_RATE)
        self.inference.prediction_ready.connect(self.on_prediction)
        self.inference.error.connect(self._inference_error)
        self.inference.start()
        self.last_prediction = ("", 0.0, None)
        self.model_gestures = None
        self._stale_logged = False

        # ---- arm state
        self.arm = None
        self.streamer = None
        self.gesture_filter = GestureFilter(self.config["filter"], resolver=self.mapping.resolve,
                                            is_continuous=self.mapping.is_continuous,
                                            conf_for=self.mapping.min_confidence)
        saved_conf = float(settings.get("min_confidence") or 0.0)
        if 0.4 <= saved_conf <= 0.99:
            self.gesture_filter.min_conf = saved_conf
        self.teleop = None
        self.home_busy_until = 0.0
        self.parking = None
        self.park_aborted = False
        self.close_ready = False
        self.pad_held = {}
        self.key_grip_pending = False
        self.last_motion_t = 0.0
        self.last_status = {"motion": None, "gesture": None}
        self.session_log = None

        self._build()
        self.keyboard = KeyboardController(self, self.keymap, self.is_armed, self.emergency_stop)
        self.keyboard.held_changed.connect(self._keys_held_changed)
        self.keyboard.ignored.connect(self._key_ignored)
        self.pages["keyboard"].attach(self.keyboard)
        self.bridge.frames.connect(self.on_batch)
        self.bridge.state.connect(self.on_link_state)

        self.control_timer = QTimer(self)
        self.control_timer.setTimerType(Qt.PreciseTimer)
        self.control_timer.setInterval(int(self.config["arm"]["control_period_ms"]))
        self.control_timer.timeout.connect(self.control_tick)
        self.control_timer.start()
        self.battery_timer = QTimer(self)
        self.battery_timer.setInterval(10000)
        self.battery_timer.timeout.connect(self.check_battery)
        self.battery_timer.start()
        self.record_timer = QTimer(self)
        self.record_timer.setInterval(500)
        self.record_timer.timeout.connect(self._update_recording)
        self.ui_timer = QTimer(self)                        # status chips + monitor refresh (4 Hz, cheap)
        self.ui_timer.setInterval(250)
        self.ui_timer.timeout.connect(self.refresh_live)
        self.ui_timer.start()

        record("info", "app", f"EMG Robotic Arm {__version__} started on {sys.platform}")
        self.go_to("control")
        last = settings.get("last_model")
        if last and Path(last).exists():
            QTimer.singleShot(200, lambda: self.load_model_path(last, quiet=True))
        self.refresh_status()

    def is_armed(self) -> bool:
        return bool(self.teleop and self.teleop.armed)

    # ================================================================== layout
    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- top bar (always visible): brand, live status, STOP
        bar = QFrame()
        theme.set_role(bar, "topbar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(18, 9, 12, 9)
        bl.setSpacing(6)
        brand = QWidget()
        brand.setMinimumWidth(124)
        names_box = QVBoxLayout(brand)
        names_box.setContentsMargins(0, 0, 0, 0)
        names_box.setSpacing(0)
        names_box.addWidget(label("BIOWAVE", "brand"))
        names_box.addWidget(label("EMG Robotic Arm", "caption"))
        bl.addWidget(brand)
        self.chip_band = Chip("Armband: offline")
        self.chip_arm = Chip("Arm: offline")
        self.chip_mode = Chip("DISARMED")
        self.chip_gesture = Chip("Gesture: -")
        for c in (self.chip_band, self.chip_arm, self.chip_mode, self.chip_gesture):
            bl.addWidget(c)
        self.chip_note = label("", "muted")
        bl.addWidget(self.chip_note, 1)
        estop = button("STOP   (Space / Esc)", "estop", slot=lambda: self.emergency_stop("Stop button pressed"))
        estop.setMinimumHeight(44)
        estop.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)      # always shown in full
        bl.addWidget(estop)
        outer.addWidget(bar)
        glow = QFrame()
        theme.set_role(glow, "glow")
        outer.addWidget(glow)

        body = QHBoxLayout()
        body.setSpacing(0)
        outer.addLayout(body, 1)

        # ---- sidebar
        side = QFrame()
        theme.set_role(side, "sidebar")
        side.setFixedWidth(224)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(10, 6, 10, 10)
        sl.setSpacing(4)
        self.nav_group = QButtonGroup(self)
        self.nav_buttons, self.nav_dots = {}, {}
        group = None
        for key, nav_text, _t, _l, grp in PAGES:
            if grp != group:
                group = grp
                sl.addSpacing(10)
                heading = section(grp)
                heading.setContentsMargins(8, 0, 0, 2)
                sl.addWidget(heading)
            b = QPushButton(" " + nav_text)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setIconSize(QSize(icons.SIZE, icons.SIZE))
            theme.set_role(b, "nav")
            b.clicked.connect(lambda _=False, k=key: self.go_to(k))
            self.nav_group.addButton(b)
            self.nav_buttons[key] = b
            dot = Dot()                                      # status light inside the button, at the right edge
            dot.setAttribute(Qt.WA_TransparentForMouseEvents)
            inside = QHBoxLayout(b)
            inside.setContentsMargins(0, 0, 8, 0)
            inside.addStretch(1)
            inside.addWidget(dot)
            self.nav_dots[key] = dot
            sl.addWidget(b)
        self._paint_nav_icons()
        sl.addStretch(1)
        sl.addWidget(label(f"Version {__version__}", "caption"))
        body.addWidget(side)

        # ---- content
        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(20, 12, 20, 4)
        hl.setSpacing(14)
        self.title = label("", "h1")
        self.lead = label("", "lead", wrap=True)
        hl.addWidget(self.title, 0, Qt.AlignVCenter)
        hl.addWidget(self.lead, 1, Qt.AlignVCenter)
        content.addWidget(head)

        self.stack = QStackedWidget()
        self.pages = {}
        simulate_default = not xarm_plugged_in()
        names = {int(k): v.get("name", f"servo {k}") for k, v in self.config["arm"]["servos"].items()}
        self.pages["model"] = ModelPage(self.settings, self.mapping, names)
        self.pages["armband"] = ArmbandPage(self.bridge, self.settings)
        self.pages["control"] = ControlPage(self.config, self.gesture_filter.min_conf, self.keymap, simulate_default)
        self.pages["arm"] = self.pages["control"].arm_page          # the arm card and the key table live on the control page
        self.pages["keyboard"] = self.pages["control"].drive
        self.pages["monitor"] = MonitorPage()
        self.pages["mapping"] = MappingPage(self.mapping, self.config)
        self.pages["data"] = DataPage(self.settings.get("theme"), self.base_config, self.settings)
        for key, *_ in PAGES:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            host = QWidget()
            hbox = QHBoxLayout(host)
            hbox.setContentsMargins(20, 8, 20, 14)
            hbox.addStretch(1)
            self.pages[key].setMaximumWidth(1500)              # keeps lines readable on very wide screens
            hbox.addWidget(self.pages[key], 100)
            hbox.addStretch(1)
            scroll.setWidget(host)
            self.stack.addWidget(scroll)
        content.addWidget(self.stack, 1)
        body.addLayout(content, 1)

        # ---- wiring
        p = self.pages
        p["model"].model_requested.connect(self.load_model_path)
        p["armband"].link_changed.connect(lambda _s: self.refresh_status())
        p["arm"].connect_toggled.connect(self.toggle_arm)
        p["arm"].home_requested.connect(self.go_home)
        p["arm"].gripper_requested.connect(self.manual_gripper)
        cp = p["control"]
        cp.arm_toggled.connect(self.toggle_armed)
        cp.pad_pressed.connect(lambda m: self.pad_held.setdefault(m, time.monotonic()))
        cp.pad_released.connect(lambda m: self.pad_held.pop(m, None))
        cp.grip_pressed.connect(lambda: setattr(self, "key_grip_pending", True))
        cp.vertical.currentIndexChanged.connect(self.on_vertical_changed)
        cp.speed.valueChanged.connect(lambda v: self.arm and self.arm.set_speed(v))
        cp.conf.valueChanged.connect(self.on_confidence_changed)
        cp.mode.currentIndexChanged.connect(self.on_mode_changed)
        saved_mode = self.settings.get("control_mode")
        cp.mode.setCurrentIndex(max(0, cp.mode.findData(saved_mode)))
        dp = p["data"]
        dp.theme_changed.connect(self.set_theme)
        dp.tuning_changed.connect(self.apply_tuning)
        dp.export_requested.connect(self.export_configuration)
        dp.import_requested.connect(self.import_configuration)
        p["mapping"].mapping_saved.connect(self.on_mapping_saved)
        p["mapping"].disarm_requested.connect(lambda: self.disarm("Disarmed to change the gesture mapping"))
        self.on_vertical_changed()
        p["mapping"].set_vertical(cp.vertical.currentData() or "shoulder")

    # ================================================================ navigation
    def _paint_nav_icons(self):
        for key, b in self.nav_buttons.items():
            b.setIcon(icons.nav_icon(key))

    def go_to(self, key: str):
        keys = [p[0] for p in PAGES]
        idx = keys.index(key)
        self.stack.setCurrentIndex(idx)
        self.nav_buttons[key].setChecked(True)
        _k, _n, title, lead, _g = PAGES[idx]
        self.title.setText(title)
        self.lead.setText(lead)
        self.current_key = key
        self.refresh_status()

    def set_theme(self, mode: str):
        self.settings.set("theme", mode)
        theme.apply_theme(QApplication.instance(), mode)
        for key in ("armband", "monitor", "mapping"):
            self.pages[key].restyle()
        theme.apply_title_bar(self)
        self._paint_nav_icons()
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
        armed = self.is_armed()
        if self.parking:
            self.chip_mode.set("PARKING", "warn")
        else:
            self.chip_mode.set("ARMED" if armed else "DISARMED", "bad" if armed else "none")

        arm_tone = "accent" if armed else ("good" if self.arm.homed else "warn") if self.arm else "none"
        tones = {"model": "good" if self.model_loaded else "none",
                 "armband": {LinkState.STREAMING: "good", LinkState.FAILED: "bad", LinkState.IDLE: "none"}.get(s, "warn"),
                 "control": arm_tone, "data": "accent" if self.recorder else "none"}
        for key, tone in tones.items():
            self.nav_dots[key].set_tone(tone)

        cp = self.pages["control"]
        missing = []
        if not self.model_loaded:
            missing.append("load a model (Gesture model)")
        if s != LinkState.STREAMING:
            missing.append("connect the armband - or use the keyboard only")
        if not self.arm:
            missing.append("connect the arm (step 1)")
        elif not self.arm.homed:
            missing.append("send the arm home (step 1)")
        if not self.arm and s == LinkState.STREAMING and self.model_loaded:
            cp.checklist.setText("Armband connected, no arm: PREVIEW mode. The Live panel shows what each gesture would do "
                                 "(nothing moves). Connect the arm or tick “Simulate” to move it.")
        else:
            cp.checklist.setText("Before arming: " + "; ".join(missing) + "." if missing else "Everything is ready.")
        self.pages["mapping"].set_armed(armed)
        self.pages["keyboard"].set_armed(armed)
        self.update_controls()
        self.refresh_live()

    # ================================================================== model
    def load_model_path(self, path: str, quiet: bool = False):
        mp = self.pages["model"]
        try:
            loaded = load_model(path)
        except ModelError as exc:
            record("error", "gesture", str(exc))
            if not quiet:
                mp.show_error(str(exc))
            return
        classes = loaded.class_names
        if self.bridge.active:
            self.bridge.disconnect()                       # buffer sizes change with the model
        self.disarm("Model changed")
        self.window_samples, self.stride_samples = loaded.window_samples, loaded.stride_samples
        self.num_channels = loaded.input_channels
        self.data_buffer = np.zeros((self.num_channels, self.window_samples), dtype=np.float32)
        self.baseline = np.zeros(self.num_channels, dtype=np.float32)
        self.sample_count = 0
        self.inference.load_model(loaded.model, classes)
        self.gesture_filter.reset()
        self.loaded, self.model_loaded = loaded, True
        self.settings.set("last_model", path)
        problems = self._show_model(loaded)
        record("info", "gesture", f"Model loaded: {Path(path).parent.name} ({len(classes)} classes: {', '.join(classes)})")
        for pmsg in problems:
            record("warning", "gesture", pmsg.lstrip("⚠ "))
        self.refresh_status()

    def _show_model(self, loaded):
        """(Re)build the class -> gesture table. Called on load and whenever the mapping changes."""
        classes = loaded.class_names
        mapping = {c: self.mapping.resolve(c) for c in classes}
        recognised = set(mapping.values()) - {None}
        problems = []
        unknown = [c for c, g in mapping.items() if g is None]
        missing = [BY_ID[g].name for g in BY_ID if g not in recognised]
        if "rest" not in recognised:
            problems.append("⚠ This model has no Rest class, so it can never say “do nothing”. Use a model trained with Rest.")
        if missing:
            problems.append("Gestures this model cannot produce (their mappings stay inactive): " + ", ".join(missing))
        if unknown:
            problems.append("Ignored classes (not one of the ten gestures; set a “Model class” in Gesture mapping): " + ", ".join(unknown))
        self.model_gestures = recognised
        if self.teleop:
            self.teleop.commands.available = recognised
        self.pages["model"].show_loaded(loaded, mapping, problems)
        self.pages["mapping"].set_model_classes(classes)
        return problems

    # ================================================================ armband data
    def on_link_state(self, state_value: str, message: str):
        previous = self.link_state
        self.link_state = LinkState(state_value)
        self.last_link_message = message
        if self.link_state != previous:
            level = "error" if self.link_state == LinkState.FAILED else \
                "warning" if self.link_state in (LinkState.STALLED, LinkState.RECONNECTING) else "info"
            record(level, "link", f"{previous.value} → {self.link_state.value}" + (f": {message}" if message else ""))
        if self.link_state != LinkState.STREAMING:
            # Predictions made before a link problem must never keep driving the arm.
            self.gesture_filter.reset()
            self.last_prediction = ("", 0.0, None)
        if self.link_state in (LinkState.IDLE, LinkState.FAILED):
            if self.recorder:
                self.toggle_recording("")
        if self.is_armed() and self.link_state != LinkState.STREAMING and self.teleop.mode != "keyboard":
            self.pages["control"].note_label.setText("Armband link lost - gestures are ignored until it recovers")
        self.refresh_status()

    def on_batch(self, batch):
        if self.data_buffer is None:
            return
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim != 2 or not len(batch):
            return
        if self.pages["monitor"].isVisible():
            self.pages["monitor"].scope.push(batch)
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
        latency = getattr(self.inference, "last_latency_s", -1.0)
        self.last_prediction = (label_text, confidence, gesture)
        self.pages["control"].raw_label.setText(
            f"Model output: {label_text}  ({confidence * 100:.0f}%)  →  {BY_ID[gesture].name if gesture else 'ignored'}")
        self.pages["monitor"].on_prediction(label_text, confidence, gesture, latency)
        stable = self.gesture_filter.stable
        m = self.mapping.get(stable)
        self.pages["monitor"].on_stable(BY_ID[stable].name if stable else None,
                                        "→ " + CommandGenerator.describe(m, self.arm_names(), self.vertical_name()) if m else "",
                                        confidence)
        self.pages["mapping"].set_live(stable, confidence)
        self.log("prediction", label=label_text, confidence=f"{confidence:.3f}", gesture=gesture or "")

    def _inference_error(self, message: str):
        record("error", "gesture", message)
        self.pages["control"].note_label.setText(message)

    def arm_names(self):
        return {int(k): v.get("name", f"servo {k}") for k, v in self.config["arm"]["servos"].items()}

    def vertical_name(self):
        return self.pages["control"].vertical.currentData() or ""

    # ================================================================ recording
    def toggle_recording(self, tag: str):
        """Raw-signal recording UI was removed; this only stops a recorder if one exists."""
        if self.recorder:
            self.bridge.recorder = None
            rec, self.recorder = self.recorder, None
            rec.close()
            self.record_timer.stop()
        self.refresh_status()

    def _update_recording(self):
        pass

    # ===================================================================== arm
    def toggle_arm(self):
        if self.arm:
            self.disconnect_arm()
            return
        rp = self.pages["arm"]
        self.config = effective_config(self.base_config, self.settings)   # pick up calibration changes
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
            record("error", "arm", f"Could not open the xArm over USB: {exc}")
            return
        positions = arm.read_positions()
        ok, voltage = arm.battery()
        arm.set_speed(self.pages["control"].speed.value())
        self.arm, self.streamer = arm, streamer
        self.arm_note = ""
        self.apply_tuning(self.pages["data"].current_tuning(), save=False)
        streamer.start()
        self.teleop = Teleop(arm, self.gesture_filter, self.config, streamer, self.mapping)
        self.teleop.vertical = self.pages["control"].vertical.currentData()
        self.teleop.commands.available = self.model_gestures
        self.on_mode_changed()
        self.open_log()
        unknown = [str(s) for s, p in positions.items() if p is None]
        volt = f"{voltage:.2f} V" if voltage else "voltage n/a"
        self.arm_note = f"{backend.name}  ·  {volt}" + (
            f"\nCould not read servo(s) {', '.join(unknown)}: their first move will be extra slow." if unknown else "")
        rp.banner.set(f"Connected: {self.arm_note}", "good" if ok else "bad")
        rp.connect_btn.setText("Disconnect arm")
        theme.set_role(rp.connect_btn, "danger")
        self.refresh_servo_view()
        self.log("arm_connected", note=f"{backend.name}; positions={json.dumps(positions)}; voltage={voltage}")
        record("info", "arm", f"Connected: {backend.name}, {volt}")
        if not ok:
            QMessageBox.warning(self, "Low voltage", f"Arm supply reads {voltage:.2f} V (minimum {arm.min_battery} V).\n"
                                "Use the 7.5 V adapter / charge the battery before moving the arm.")
        else:
            self.go_home(confirm=False)               # connecting is the go-ahead: the arm homes by itself (slowly)
        self.refresh_status()

    def disconnect_arm(self):
        record("info", "arm", "Arm disconnected")
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
        self.close_log()
        self.refresh_status()

    def go_home(self, confirm=True):
        if not self.arm:
            self.pages["arm"].banner.set("Connect the arm first.", "warn")
            return
        if confirm and QMessageBox.question(self, "Go home", "The arm will move slowly to its home pose.\nMake sure nothing and nobody "
                                            "is within reach of the arm.\n\nContinue?") != QMessageBox.Yes:
            return
        self.disarm("Homing")
        try:
            seconds = self.arm.go_home()
        except Exception as exc:
            self.handle_arm_failure(exc)
            return
        wait = seconds + 0.3 if not isinstance(self.arm.backend, SimulatedBackend) else 0.2   # a simulated arm needs no real time
        self.home_busy_until = time.monotonic() + wait
        self.pages["arm"].banner.set(f"Moving to the home pose… ({seconds:.1f} s). Please keep clear of the arm.", "accent")
        self.log("home", note=f"{seconds:.1f}s")
        self.refresh_servo_view()
        QTimer.singleShot(int(wait * 1000), self.after_home)
        self.refresh_status()

    def after_home(self):
        if self.arm and self.arm.homed:
            self.pages["arm"].banner.set(f"Ready: the arm is at home ({getattr(self, 'arm_note', '').splitlines()[0] if getattr(self, 'arm_note', '') else 'connected'}). "
                                         "Press ARM below to enable motion.", "good")
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
            QMessageBox.information(self, "Not ready", "Connect the arm and press “Go home” first (step 1 on this page).")
            return
        if cp.mode.currentData() != "keyboard" and not (self.link_state == LinkState.STREAMING and self.model_loaded):
            if QMessageBox.question(self, "Gestures not available",
                                    "The armband is not streaming or no model is loaded, so EMG gestures cannot be used.\n\n"
                                    "ARM in keyboard-only mode?") != QMessageBox.Yes:
                return
            cp.mode.setCurrentIndex(cp.mode.findData("keyboard"))
        self.gesture_filter.reset()
        self.keyboard.release_all()
        self.streamer.clear_error()
        self.teleop.armed = True
        self.log("armed")
        record("info", "arm", f"ARMED ({cp.mode.currentText()})")
        self.refresh_status()

    def not_ready_reason(self):
        if not self.arm:
            return "Keys only move the arm when ARMED: first connect the arm (step 1)."
        if not self.arm.homed:
            return "Keys only move the arm when ARMED: send the arm home first (step 1), then ARM."
        if time.monotonic() < self.home_busy_until:
            return "Wait for homing to finish, then ARM."
        return "Press ARM to enable motion, then use the keys."

    def disarm(self, reason=""):
        if self.streamer:
            try:
                self.streamer.halt()
            except Exception as exc:
                console(f"Halt failed: {exc}")
        if self.teleop and self.teleop.armed:
            self.teleop.armed = False
            self.log("disarmed", note=reason)
            record("info", "arm", f"Disarmed: {reason or 'user'}")
        self.keyboard.release_all() if hasattr(self, "keyboard") else None
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
        record("warning", "arm", f"STOP: {reason}")
        self.disarm(f"STOPPED: {reason}")
        self.pages["control"].action_label.setText("STOPPED")

    def handle_arm_failure(self, exc):
        record("error", "arm", f"{type(exc).__name__}: {exc}")
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
        if "mapping" in self.pages:
            self.pages["mapping"].set_vertical(cp.vertical.currentData() or "shoulder")

    def update_controls(self):
        armed = bool(self.teleop and self.teleop.armed)
        connected = self.arm is not None
        homing = time.monotonic() < self.home_busy_until or self.parking is not None
        rp, cp = self.pages["arm"], self.pages["control"]
        rp.simulate.setEnabled(not connected)
        rp.card.set_done(connected and self.arm.homed)
        rp.home_btn.setEnabled(connected and not armed and not homing)
        rp.grip_btn.setEnabled(connected and self.arm.homed and not homing)
        rp.set_gripper_state(self.arm.gripper_closed if connected else None)
        cp.arm_btn.setEnabled((armed or self.can_arm()) and self.parking is None)
        cp.arm_btn.setText("DISARM" if armed else "ARM - enable motion")
        theme.set_role(cp.arm_btn, "outline" if armed else "success")
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
    def _keys_held_changed(self, names):
        cp = self.pages["control"]
        cp.key_label.setText(f"Keyboard: {' '.join(names)} held" if names else "Keyboard: idle")
        held = self.keyboard.held_motions(time.monotonic())
        for motion in cp.pad_buttons:
            cp.highlight_key(motion, motion in held)

    def _key_ignored(self, text):
        if self.current_key == "control":
            cp = self.pages["control"]
            cp.key_label.setText(f"Keyboard: {text}")
            cp.note_label.setText(self.not_ready_reason())

    # ============================================================ control loop
    def control_tick(self):
        now = time.monotonic()
        if not self.is_armed():
            self.preview_tick(now)
            return
        cp = self.pages["control"]
        if self.streamer.error is not None:
            exc, self.streamer.error = self.streamer.error, None
            self.handle_arm_failure(exc)
            return
        select = self.keyboard.take_select()
        if select and cp.vertical.findData(select) >= 0:
            cp.vertical.setCurrentIndex(cp.vertical.findData(select))
        key_motions = self.keyboard.held_motions(now)
        key_motions.update({m: now - since for m, since in self.pad_held.items()})
        key_grip = self.keyboard.take_grip() or self.key_grip_pending
        self.key_grip_pending = False
        mode = cp.mode.currentData()
        emg_ok = (mode != "keyboard" and self.link_state == LinkState.STREAMING and self.model_loaded
                  and now - self.last_batch_t <= float(self.config["filter"]["emg_timeout_s"]))
        try:
            status = self.teleop.tick(now, key_motions, key_grip, emg_ok)
        except Exception as exc:
            self.handle_arm_failure(exc)
            return
        note = status["note"]
        if mode == "keyboard" and (not note or note.startswith("EMG stream lost")):
            note = "Keyboard only"
        stale = mode != "keyboard" and note.startswith("EMG stream lost")
        if stale and not self._stale_logged:                      # log the transition, not every tick
            record("warning", "gesture", "EMG data stale - the arm holds its position")
        self._stale_logged = stale
        cp.note_label.setText(note)
        cp.grip_hint.setText(status["hint"] if emg_ok else "")
        cp.action_label.setText(f"{status['motion'].upper()}  ({status['source']})" if status["motion"] else "HOLD")
        self.last_status = status
        if status["switched"]:
            cp.vertical.setCurrentIndex(cp.vertical.findData(status["switched"]))
            self.log("switch_joint", note=status["switched"], source=status["source"])
        if status["gripper"]:
            self.log("gripper", note=status["gripper"], source=status["source"])
        if status["goto"]:
            self.log("goto", note=f"servo {status['goto'][0]} -> {status['goto'][1]}", source=status["source"])
        if status["home"]:
            record("info", "gesture", "Gesture requested: go home")
            self.go_home(confirm=False)
            return
        if status["motion"] or status["gripper"]:
            self.last_motion_t = now
            self.log("move", motion=status["motion"] or "", source=status["source"] or "", note=note)
        if status["motion"] or status["gripper"] or status["goto"] or self.streamer.is_moving():
            self.refresh_servo_view()

    def preview_tick(self, now):
        """Not armed (or no arm at all): show what the gesture WOULD do, without moving anything."""
        cp = self.pages["control"]
        streaming = self.link_state == LinkState.STREAMING and self.model_loaded
        if not streaming:
            return
        gid = self.gesture_filter.stable if self.gesture_filter.is_fresh(now) else None
        m = self.mapping.get(gid)
        tag = "PREVIEW" if self.arm else "PREVIEW (no arm connected)"
        what = CommandGenerator.describe(m, self.arm_names(), self.vertical_name()) if m else ""
        cp.action_label.setText(f"{tag}: {BY_ID[gid].name}  →  {what}" if gid else f"{tag}: rest")

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
        self.pages["arm"].set_gripper_state(self.arm.gripper_closed)

    # ================================================================ settings hooks
    def apply_tuning(self, values, save=False):
        """Push the smoothing parameters to the running motion thread and the gesture guard."""
        if self.streamer:
            try:
                self.streamer.apply_tuning(lookahead=values["lookahead"] / 1000.0, accel=values["accel"],
                                           decel=values["decel"], jerk=values["jerk"],
                                           min_speed=values["min_speed"], min_lead=values["min_lead"])
            except ArmSafetyError as exc:
                record("error", "arm", f"Smoothing not applied: {exc}")
                QMessageBox.warning(self, "Smoothing not applied", str(exc))
                return
        if self.teleop:
            self.teleop.reversal_guard_s = float(values["reversal_guard"])
        record("info", "arm", "Motion smoothing: " + ", ".join(f"{k}={v:g}" for k, v in values.items()))

    def on_mode_changed(self, *_):
        mode = self.pages["control"].mode.currentData()
        self.settings.set("control_mode", mode)
        if self.teleop:
            self.teleop.mode = mode
        self.refresh_live()

    def on_confidence_changed(self, value):
        self.gesture_filter.min_conf = value / 100.0
        self.settings.set("min_confidence", value / 100.0)
        self.pages["monitor"].set_threshold(f"Global confidence threshold: {value:.0f} %  (change it on Arm control; "
                                            "individual gestures can override it in Gesture mapping)")

    def on_mapping_saved(self):
        self.gesture_filter.reset()
        if self.teleop:
            self.teleop.commands.reset()
        if self.loaded:
            self._show_model(self.loaded)
        record("info", "gesture", "Gesture mapping saved and activated")
        self.refresh_live()

    def export_configuration(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export configuration", str(Path.home() / "emg_arm_config.json"), "JSON (*.json)")
        if not path:
            return
        try:
            export_bundle(Path(path), self.mapping, self.keymap, self.settings)
        except OSError as exc:
            self.pages["data"].config_note.setText(f"Export failed: {exc}")
            return
        self.pages["data"].config_note.setText(f"Exported to {path}")
        record("info", "app", f"Configuration exported to {path}")

    def import_configuration(self):
        if self.is_armed():
            QMessageBox.information(self, "Disarm first", "Disarm the arm before importing a configuration.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import configuration", str(Path.home()), "JSON (*.json)")
        if not path:
            return
        try:
            import_bundle(Path(path), self.mapping, self.keymap, self.settings)
        except (ValueError, MappingError) as exc:
            self.pages["data"].config_note.setText(f"Import failed, nothing was changed: {exc}")
            record("error", "app", f"Configuration import failed: {exc}")
            return
        self.pages["keyboard"].refresh()
        self.pages["data"].load_tuning()
        self.pages["data"].load_calibration()
        self.pages["mapping"].work = __import__("copy").deepcopy(self.mapping.mappings)
        self.pages["mapping"]._rebuild_list()
        self.pages["mapping"]._load_editor()
        cp = self.pages["control"]
        cp.mode.setCurrentIndex(max(0, cp.mode.findData(self.settings.get("control_mode"))))
        self.on_mapping_saved()
        self.pages["data"].config_note.setText(f"Imported {path}. Calibration applies when the arm is reconnected.")
        record("info", "app", f"Configuration imported from {path}")

    # ============================================================ live status
    def refresh_live(self):
        """Cheap 4 Hz refresh of the bottom-bar gesture chip and the monitor's movement banner."""
        stable = self.gesture_filter.stable if self.gesture_filter.is_fresh(time.monotonic()) else None
        self.chip_gesture.set(f"Gesture: {BY_ID[stable].name}" if stable else "Gesture: -", "accent" if stable else "none")
        move = self.last_status.get("motion")
        self.pages["monitor"].set_movement(f"Arm: {move.upper()}" if move else "Arm: no movement", bool(move))

    # ================================================================= logging
    def open_log(self):
        self.session_log = SessionLog(LOG_DIR)

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
        self.keyboard.detach()
        self.ui_timer.stop()
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
