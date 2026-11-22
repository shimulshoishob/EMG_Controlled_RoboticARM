# EMG Robotic Arm

Research application: control a Hiwonder **xArm 1S** with gestures from the **BioWave ESP32-S3 EMG armband**
(8-channel EMG @ 500 Hz + BNO080 IMU, firmware `EMG_IMU_WIFI_SETUP_V2`, protocol BWIM v2). Runs on **macOS and Windows** (Linux too), light and dark theme.

```
Armband ──Wi-Fi (UDP) or USB──▶ laptop: Random Forest ─▶ vote filter ─▶ motion thread ─▶ safety layer ─▶ USB ─▶ xArm 1S
```

## Start

```bash
python3 run.py        # macOS        |   py run.py     # Windows
```
The first run creates `.venv` and installs `requirements.txt` (a few minutes, needs internet). Python 3.11+.

## Sections (left sidebar)

| Group | Section | What it is for |
|---|---|---|
| Overview | **Dashboard** | Armband / ESP32-S3 link, incoming data, model, detected gesture, arm, control mode, current movement, error count, recent warnings |
| Setup | **Gesture model** | Pick the trained `.joblib`; shows how each model class is recognised as one of the ten gestures |
| Setup | **Connection** | Armband over Wi-Fi, USB or the virtual armband (four cards, *Connection doctor*) |
| Setup | **Robotic arm** | Connect the xArm 1S (or simulate it), *Go home*, gripper test |
| Operate | **Arm control** | *ARM*, control mode (keys + gestures / gestures only / keyboard only), speed, global confidence, on-screen pad, key reference |
| Operate | **Live monitor** | 8-channel EMG, model output, confidence, stable gesture, transition history, signal→prediction latency |
| Operate | **Gesture mapping** | What each of the ten gestures does (see below) |
| Operate | **Keyboard** | Key tester, re-assign keys, reset |
| System | **Logs & diagnostics** | Packet rate / loss / age, firmware counters, event log, error history, save/copy log |
| System | **Settings & data** | Motion smoothing, calibration (servo range, rest pose), export/import configuration, raw recording, session logs, theme |

The red **STOP** button (Space / Esc) is on every page and always works.

### Gesture mapping

Ten gestures: Left, Right, Up, Down, Closed Fist, Double Closed Fist, Victory Sign, Thumbs Up, Pinch, Rest.
Each has: enabled, robot action, joint/servo, direction, speed, target position + duration (go-to-position), hold time, optional
long-hold action, confidence threshold, and an optional *model class* (when your model names a gesture differently).

Actions: move a joint while held · go to a fixed position · toggle / open / close gripper · switch Up/Down joint (shoulder ↔ elbow) ·
go home · hold still · nothing. The editor rejects invalid values (unknown servo, speed over the cap, target outside the servo's safe range,
two gestures driving the same joint the same way) and shows live what the gesture will do; the highlighted row is the gesture detected right now.
Mappings are saved per user (`gesture_mapping.json` in the settings folder) and survive restarts; *Reset to defaults*, *Export…* and *Import…* are on the page.
Mappings cannot be saved while the arm is ARMED. **Double Closed Fist → Go home is shipped disabled**; enable it deliberately.

One-shot actions fire **once per gesture**: keep the gesture detected and nothing repeats; relax / change gesture to arm it again. A per-action cooldown (1.5 s) stops flicker re-triggering.

### Keyboard (defaults, all re-assignable on the Keyboard page)

| Keys | Action | Keys | Action |
|---|---|---|---|
| ← / A | base left | → / D | base right |
| ↑ | selected joint up | ↓ | selected joint down |
| W / S | shoulder up / down | E / Q | elbow up / down |
| G | toggle gripper | 1 / 2 | Up/Down controls shoulder / elbow |
| **Space / Esc** | **STOP (fixed)** | | |

Hold to move, release to stop, hold several to combine. Keys only move the arm while ARMED and when no text box has focus; Ctrl/Cmd/Alt combinations are ignored.
*Changed from before:* **D** was elbow-down and is now base-right (WASD-style); elbow-down moved to **Q**. Re-assign on the Keyboard page if you prefer the old layout.

## What was fixed for the new firmware

| Problem | Fix |
|---|---|
| App only understood BWIM v1 (252 B); firmware sends **v2** (308 B, 28-byte header, 64-bit timestamps, health counters) | `armband/protocol.py` parses v2 (and still v1) and reports sequence gaps, late samples and the firmware's own drop/overrun counters |
| `START` always failed with `AUTH_FAILED`: the firmware ties a challenge to the sender's **IP + source port**, the old app used a new socket per message | `ControlClient` keeps one socket for CHALLENGE + command |
| `HELLO` (9 fields) / `INFO` (7 fields) rejected | tolerant parsers |
| Lost UDP replies | every transaction retried with a fresh challenge, stale datagrams dropped |
| Discovery guessed broadcast addresses (wrong on macOS/Windows, one adapter only) | enumerates real adapters with netmasks, directed broadcast per adapter, 3 rounds, unicast /24 sweep fallback |
| Data port 5000 busy / stolen by another copy (`SO_REUSEADDR`) | exclusive bind, falls back to 5001-5009, tells the armband the port |
| Armband reboot, Wi-Fi drop, DHCP address change, firmware keepalive (15 s) timeout | keepalive every 3 s, watchdog, automatic re-`START`, re-discovery by device ID, exponential back-off |
| Silent "connected but no data" (firewall/VPN) | detected and explained (Windows Defender / macOS firewall / client isolation) |
| USB: opening the port reset the ESP32; fixed 3.5 s sleep; long passwords overflowed the firmware's 255-byte line | DTR/RTS held low, probe retries until the board answers, minimal encoding + length checks |
| USB streaming (CSV text) | COBS + CRC-16 binary framing, resynchronises after corruption |
| Placeholder access key accepted | key validated (≥16 chars, not the placeholder), stored per user outside the repo |

## Layout

```
run.py                      launcher (sets up .venv)
config/arm_config.json      limits, home pose, gesture map, filter settings
firmware/                   armband firmware (+ secrets.example.h; never commit secrets.h)
src/emg_arm/
  armband/  protocol · control · netinfo · link · serial_setup · diagnostics · fake (firmware emulator)
  arm/      driver · motion · teleop       (safety layer, smooth streaming, gesture filter)
  control/  gestures · mapping · commands  (ten gestures, central mapping store, gesture -> command generator; no Qt)
  ml/       features · inference
  ui/       theme · widgets · bridge · dialogs · keyboard · main_window · pages/
  session.py settings.py config_bundle.py diagnostics_log.py paths.py bootstrap.py app.py
tools/check_arm.py          read-only hardware check / nudge / glide
tools/jitter_report.py      software-side base-servo jitter measurement (old vs new streaming)
tools/native_key_check.py   macOS: injects real OS key events into the real window (see docs; unverified in CI)
tests/                      149 tests, no hardware needed
```

## Tests

```bash
python -m unittest discover tests
```
Includes a software copy of the firmware (`armband/fake.py`) that can lose replies and data, reboot and go silent, a USB test over a pseudo-terminal, and a headless run of the real window. Run the emulator on its own with `python -m emg_arm.armband.fake --key <key>`.

## Troubleshooting (or press *Connection doctor*)

| Symptom | Fix |
|---|---|
| Scan finds nothing | same 2.4 GHz Wi-Fi as the laptop; VPN off; router "client/AP isolation" off; or type the IP |
| "access key does not match" | *Import from secrets.h* with the file you flashed |
| Connected, no data | allow Python in the firewall (Windows: Private networks; macOS: Firewall options) |
| USB port busy | close the Arduino Serial Monitor |
| Port not listed | use the armband's USB port; try another cable (some are charge-only) |

## Hardware safety (unchanged)

Power the arm from its own supply; keep its reach clear; wear the armband on Wi-Fi, or run the laptop on battery when using USB; one program controls the arm; know the stop: Space/Esc, then the power switch (support the arm by hand). The arm never loses torque in software, on exit it parks at home.

## More documentation

`docs/CHANGES_AND_ARCHITECTURE.md` - codebase analysis, architecture before/after, issue list, what changed (and what was *not* verified), hardware validation checklist, limitations.
