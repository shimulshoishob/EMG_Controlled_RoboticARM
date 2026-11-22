# EMG Robotic Arm - analysis, changes and validation

Scope of this revision: gesture-to-arm mapping, keyboard control, base-servo jitter, UI organisation, diagnostics and
safety around connection loss. The armband protocol/link layer was **not** modified (see 1.4).

## 1. Analysis of the existing application

### 1.1 Architecture before (what the code actually did)

```
BioWave ESP32-S3 (8x EMG @500 Hz + IMU, BWIM v2, UDP/USB)
   │  armband/link.py (WifiLink | UsbLink threads: keepalive, watchdog, re-START, back-off)
   │  armband/protocol.py (parse + validate, seq-gap/late/drop counters)
   ▼
ui/bridge.py  (Qt signals: frames / state / stats)
   ▼
MainWindow.on_batch: baseline centring, sliding window (100 samples, stride 25)
   ▼
ml/inference.py  InferenceWorker (QThread): features -> RandomForest -> (label, confidence)
   ▼
MainWindow.on_prediction -> arm/teleop.GestureFilter   (vote 4/5, hysteresis, stale timeout, lock-out,
   │                                                     + HARD-CODED fist logic: tap=gripper, long=switch joint)
   ▼
Teleop.tick  (config["gestures"] = fixed left/right/up/down -> servo)   <- keyboard keys merged here
   ▼
arm/motion.MotionStreamer (own thread, 40 ms packets, S-curve velocity, soft-limit braking)
   ▼
arm/driver.SafeArm (limits, speed cap, one packet per cycle, int conversion) -> xarm HID -> servos
```
GUI: PyQt5, five "step" pages (Model, Armband, Arm, Control, Data), fixed 5-step flow, a global STOP button.
Cross-platform plumbing (user-settings folder, fonts, title bar, exclusive UDP bind, adapter enumeration) was already in place.

### 1.2 Findings, by priority

| # | Sev | Finding | Status |
|---|---|---|---|
| 1 | High | Only 4 directions + a fist were representable. Gesture meaning was hard-coded in `GestureFilter`/`Teleop`/`config["gestures"]`; none of the other 5 study gestures (Double Fist, Victory, Thumbs Up, Pinch, + configurable Rest) could do anything, no UI to change it. | Fixed: `control/` package + Gesture Mapping page |
| 2 | High | Keyboard handler swallowed arrows/W/S/E/D whenever the window was active and **disarmed**, so they never reached lists/spin boxes (focus conflict) and the only feedback was one muted label ("keys do nothing"). Cmd/Ctrl combos counted as keys (macOS never delivers the key-up of a letter pressed with Cmd → stuck key until the 6 s limit). A and Q were unused and nothing was re-assignable. | Fixed: `ui/keyboard.py` |
| 3 | High | Base jitter: see section 3. | Mitigated in software; **hardware confirmation pending** |
| 4 | Med | Stale predictions: already blocked by `prediction_timeout_s`, `emg_timeout_s` and link-state gating, but the filter was only reset on IDLE/FAILED, not on STALLED/RECONNECTING. | Fixed (reset on any non-streaming state), tested |
| 5 | Med | No place to see packet statistics, event history or errors; failures were only toasts/labels. | Fixed: Logs & diagnostics, `logs/app.log` |
| 6 | Med | Calibration limited to a confidence spin box that was not remembered; no export/import of settings. | Fixed (narrow-only calibration, bundle export/import) |
| 7 | Low | Fist logic documented only in code; "dow" typo class in one trained model was ignored. | Mapping + resolver; `dow` recognised |
| 8 | Info | `QPushButton` nav text "&" was eaten as a mnemonic on new pages (found while reviewing screenshots). | Fixed |

### 1.3 Things that were already fine and were kept
Connection state machine, keepalive/auto-reconnect, exclusive port bind, packet validation, USB COBS/CRC framing, SafeArm limits,
speed cap, homing rules, parking on exit, the 75 existing tests. They were not rewritten.

### 1.4 Packet format
The firmware (`EMG_IMU_WIFI_SETUP_V2.ino`) and `armband/protocol.py` agree on magic `BWIM`, version 2 and 5 frames per packet
(spot-checked; the parser was **not** changed, and its existing emulator-based tests still pass). I did not re-audit every field offset.

## 2. Architecture after

```
model label ──resolve_gesture──▶ GestureFilter ──stable gesture id──▶ CommandGenerator ◀── MappingStore (JSON, validated)
(model.classes)  (+per-gesture      (vote / hysteresis /               (moves, one-shots,         ▲ Gesture Mapping page
                  model_label)       stale / lock-out,                  hold timers, once-per-     │ import / export / reset
                                     per-gesture confidence)            activation, cooldown)
                                                                              │
keyboard (KeyboardController + KeyMap) ───────────────────────────────────────┤
                                                                              ▼
                                                   Teleop (mode, reversal guard, limits) ─▶ MotionStreamer ─▶ SafeArm ─▶ xArm
```
`control/*` and `arm/teleop.py` contain no Qt and no hardware access, so the whole decision chain is unit-tested without hardware.
Hardware access stays in `arm/driver.py`; GUI code never talks to USB/UDP directly.

Navigation: Dashboard · Gesture model · Connection · Robotic arm · Arm control · Live monitor · Gesture mapping · Keyboard ·
Logs & diagnostics · Settings & data. Visual identity (theme, cards, banners, chips) is unchanged.

## 3. Base-servo jitter: investigation and result

Evidence gathered (software side only; no arm was available in this session):

* **Prediction noise is not the cause.** In your recorded session `logs/session_20261110_101541.csv` there are 5 406 predictions and
  **0** direct Left↔Right flips; only 90 are below the 0.65 threshold. Keyboard sessions show 0-2 base direction reversals each (deliberate presses).
* **Commanded targets are smooth** (steps of 3-4 units per 40 ms packet at 100 u/s).
* **What is noisy is the commanded *speed* at low speed.** The streamer sends integer targets "lead units ahead, reach it in `lookahead` ms".
  At 10-20 u/s the lead is 1-2 units, so ±0.5-unit rounding of the target changes the commanded servo speed by 13 % (`tools/jitter_report.py`,
  ideal-servo model). A servo re-commanded every 40 ms with a speed that wobbles that much buzzes; at ≥50 u/s the effect is negligible.

Software changes (all configurable; Settings & data → Motion smoothing, or `config/arm_config.json`):

| Technique | Where | Default |
|---|---|---|
| Duration matching: packet duration derived from the *rounded* lead so commanded speed is exactly v | `MotionStreamer._match_duration`, `match_duration` | on |
| Minimum speed floor (no creeping) | `min_speed_units_per_s` | 8 u/s |
| Adaptive lookahead: lead ≥ `min_lead_units`, lookahead stretched at low speed (≤ 500 ms) | `min_lead_units` | 3 |
| Direction-reversal guard for gestures (no instant left↔right flip) | `filter.reversal_guard_s` | 0.2 s |
| Rate limiting / duplicate suppression | `stream_period_ms` (≥ 30 ms enforced), identical targets never re-sent | existing |
| Smoothing (jerk-limited S-curve), acceleration limits, soft-limit braking | `accel/decel/jerk` | existing, now live-tunable |
| Prediction stability (vote window, hysteresis, per-gesture confidence) | `filter.*` | existing + per-gesture threshold |

Measured (simulated ideal servo, std of commanded speed / target speed): 10 u/s **0.136 → 0.001**, 20 u/s **0.123 → 0.001**,
50 and 100 u/s unchanged (0.001). Full range is preserved (test: the base still reaches its configured limit). 

**Not established:** whether the vibration you see is caused by this effect. Other candidates that software cannot rule out: servo
backlash/load on the base, supply sag (battery voltage logged at connect was 7.7 V, fine), and the xArm firmware re-profiling on each
packet. Hardware validation steps are in section 7. If it persists, raise *Minimum speed* / *Minimum lead*, or raise
`stream_period_ms` to 50-60 with `lookahead_ms` 150-200.

## 4. Keyboard control on macOS

Root causes (from code reading, then reproduced with Qt key events): see finding #2. New rules are documented at the top of
`ui/keyboard.py`. Verified with Qt test events (press/release, auto-repeat pairs, Cmd/Ctrl, text-field focus, focus loss, remapping, Space/Esc).
**Not verified:** the real macOS keyboard path. `tools/native_key_check.py` injects OS-level key events into the real Cocoa window; it
could not run in this session's sandbox (Cocoa window did not start), so run it yourself (needs Accessibility permission for your terminal).

## 5. Files

New: `src/emg_arm/control/{__init__,gestures,mapping,commands}.py`, `src/emg_arm/{config_bundle,diagnostics_log}.py`,
`src/emg_arm/ui/keyboard.py`, `src/emg_arm/ui/pages/{dashboard,monitor,mapping,keyboard,diagnostics}_page.py`,
`tests/test_{mapping,keyboard,motion_jitter,integration}.py`, `tools/{jitter_report,native_key_check}.py`, this document.

Modified: `src/emg_arm/arm/{teleop,motion,driver}.py` (generic filter, mapping execution, anti-jitter, `move_to`),
`src/emg_arm/ml/inference.py` (latency measurement), `src/emg_arm/ui/main_window.py` (navigation, wiring, safety, diagnostics),
`src/emg_arm/ui/pages/{control,model,data}_page.py`, `src/emg_arm/{settings,app}.py`, `config/arm_config.json`,
`tests/{test_safety,test_ui_smoke}.py` (API change), `README.md`.

Removed: `KeyboardFilter` / `MainWindow.handle_key` (replaced by `KeyboardController`); `GestureFilter.take_grip_toggle/take_switch/motion`
(behaviour moved to `CommandGenerator`, same scenarios still tested); the "Use EMG gestures" checkbox (now the Control mode selector);
the 5-step wizard "Next" button.

Dependencies: none added.

## 6. Tests

`python -m unittest discover tests` → 149 tests, no hardware (75 before). Simulated: firmware emulator, simulated arm, Qt offscreen
events. New coverage: ten-gesture resolution; mapping defaults/validation/conflicts/persistence/import/export/corrupt files;
once-per-activation + cooldown + legacy fist behaviour; per-gesture speed/confidence; modes; link-loss and arm-failure safety;
keyboard (arrows, WASD, release, repeat, combos, focus, remap, STOP); jitter metric; navigation, dashboard, diagnostics; config bundle.

## 7. Hardware validation checklist (to do on the real arm, arm powered from its own supply, workspace clear)

1. `python tools/check_arm.py` (read-only) passes; voltage ≥ 7.0 V.
2. Connection → Robotic arm → Go home; verify every servo reaches home.
3. Keyboard only: A/← and D/→ rotate the base the right way, release brings the servo to rest within ~0.3 s (jerk-limited stop), holding 5 s keeps moving smoothly; W/S, E/Q, G, 1/2.
4. **Base jitter:** jog the base at 10, 20, 50, 100 u/s with *Settings → smoothing* defaults, then with `match_duration:false`, `min_speed 0`, `min_lead 0`; compare by ear/eye and note any speed where it still buzzes (try raising min speed/lead).
5. Gestures (armband on, model with the 10 classes): each of the ten does what the mapping says; hold Closed Fist for 10 s → gripper toggles once.
6. Safety: switch the armband off while a gesture is active (arm must stop within ~0.5 s: 0.25 s command timeout + ~0.3 s jerk-limited stop); unplug the arm USB while moving (app must disarm and report); Space/Esc during homing.
7. Wi-Fi: power-cycle the armband, move out of range, change the router channel; link must recover on its own (Logs & diagnostics shows the transitions).
8. macOS: run `tools/native_key_check.py`; Windows: repeat steps 3 and 5 once.

## 8. Limitations and open items

* The trained models in the repo have 6 classes (Left, Right, Up, Down, fist, Rest), none has the 10 study gestures. Mappings for
  Double Fist, Victory, Thumbs Up and Pinch are therefore inactive until a 10-class model is trained; the UI marks them "not in model".
* "EMG-based automatic control" (as a mode separate from gesture control) was not implemented: it is undefined for this hardware.
  Modes are: keys + gestures, gestures only, keyboard only, plus emergency stop.
* Servo *current* position is not read back during motion (the xArm HID read is slow and shares the USB link with the motion stream); the monitor shows the
  commanded targets. Command-transmission status is visible as arm errors in the log; per-packet acknowledgements do not exist on this hardware.
* End-to-end latency from muscle to servo was not measured; only signal-window → prediction (shown live) is.
* Arm USB loss disarms and reports; automatic arm reconnection is not implemented (re-connect on the Robotic arm page) because silently re-enabling a physical arm is a safety risk.
* Keys are matched by Qt key code (layout-dependent). On non-QWERTY layouts use the Keyboard page to re-assign.
* The macOS native key path and all real-arm behaviour listed in section 7 are unverified.
