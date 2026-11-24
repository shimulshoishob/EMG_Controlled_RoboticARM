# EMG Robotic Arm

Research application: control a Hiwonder **xArm 1S** with gestures from the **BioWave ESP32-S3 EMG armband**
(8-channel EMG @ 500 Hz + BNO080 IMU, firmware `EMG_IMU_WIFI_SETUP_V2`, protocol BWIM v2). Runs on **macOS and Windows** (Linux too), light and dark theme.

```
Armband ──Wi-Fi (UDP) or USB──▶ laptop: gesture model ─▶ vote filter ─▶ motion thread ─▶ safety layer ─▶ USB ─▶ xArm 1S
```

## Start

```bash
python3 run.py        # macOS        |   py run.py     # Windows
```
The first run creates `.venv` and installs `requirements.txt` (a few minutes, needs internet). Python 3.11+.

## Sections (left sidebar)

| Group | Section | What it is for |
|---|---|---|
| Robotic arm | **Arm control** (opens first) | One page: connect the xArm 1S (or simulate it): it **goes home by itself** after connecting, and one gripper button opens or closes it · *ARM* · control mode (keys + gestures / gestures only / keyboard only), speed, global confidence · live servo positions · the table of on-screen buttons and keyboard shortcuts is tucked under **Show advanced keyboard controls** (the keys work either way; click a key there to re-assign it) |
| EMG & model | **Gesture model** | Pick the trained `.joblib`; shows how each model class is recognised as one of the ten gestures |
| EMG & model | **Armband** | Connect the real armband over Wi-Fi or USB (four cards, *Connection setup*; the live signal is hidden until you press *Show live signal*) |
| EMG & model | **Live monitor** | 8-channel EMG, model output, confidence, stable gesture, transition history |
| EMG & model | **Gesture mapping** | What each of the ten gestures does (see below) |
| Settings | **Settings & data** | Theme, export/import configuration, raw-signal recording, servo calibration (range, rest pose), advanced motion smoothing (hidden until you ask) |

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
  armband/  protocol · control · netinfo · link · serial_setup · diagnostics (connection setup check)
  arm/      driver · motion · teleop       (safety layer, smooth streaming, gesture filter)
  control/  gestures · mapping · commands  (ten gestures, central mapping store, gesture -> command generator; no Qt)
  ml/       features · inference
  ui/       theme · widgets · bridge · dialogs · keyboard · main_window · pages/
  session.py settings.py config_bundle.py app_log.py paths.py bootstrap.py app.py
tools/check_arm.py          read-only hardware check / nudge / glide
tools/jitter_report.py      software-side base-servo jitter measurement (old vs new streaming)
tools/native_key_check.py   macOS: injects real OS key events into the real window (see docs; unverified in CI)
tests/                      no hardware needed (tests/fake_armband.py is a software copy of the armband firmware)
```

## Tests

```bash
python -m unittest discover tests
```
Includes a software copy of the firmware (`tests/fake_armband.py`, test-only: the app has no demo mode) that can lose replies and data, reboot and go silent, a USB test over a pseudo-terminal, and a headless run of the real window. Run the emulator on its own with `python tests/fake_armband.py --key <key>`.

## Troubleshooting (or press *Connection setup*)

| Symptom | Fix |
|---|---|
| Scan finds nothing | same 2.4 GHz Wi-Fi as the laptop; VPN off; router "client/AP isolation" off; or type the IP |
| "access key does not match" | *Import from secrets.h* with the file you flashed |
| Connected, no data | allow Python in the firewall (Windows: Private networks; macOS: Firewall options) |
| USB port busy | close the Arduino Serial Monitor |
| Port not listed | use the armband's USB port; try another cable (some are charge-only) |
| *SIMULATION ONLY* in the banner | **Simulate the arm** was ticked. Disconnect, untick it, connect again (it is unticked automatically when the xArm is plugged in at start-up) |
| `open failed` when connecting the arm | arm switched off, USB cable is power-only, or another program has the arm open |
| Arm never moves with EMG (the right gesture shows) | confidence is below *Min. confidence*; look at the % in "Model output". If correct gestures sit at 50-60 %, lower it a little or retrain |
| Arm moves when you are relaxed | raise *Min. confidence*, re-record Rest, retrain, keep the electrodes where they were when you recorded |
| "Same gesture too long - relax" | stuck-gesture lockout after 6 s; relax briefly |
| Motion still a bit jerky on the real arm | raise *Lookahead* (Settings & data → advanced motion smoothing) to 160-200 ms; check with `tools/check_arm.py --glide 6` |
| "Model needs catboost/xgboost/lightgbm" | delete the `.venv` folder and start again, or `pip install` the named library into `.venv` |

## Hardware safety checklist

1. **Power the arm from its own 7.5 V adapter or charged battery.** USB is only for data.
2. **Keep the arm's reach clear.** Stable table, nothing and no hands within about 40 cm. The person wearing the armband sits **outside** that reach.
3. **Electrical safety for the person wearing the electrodes.** Use the **Wi-Fi** connection, or if you use USB, run the laptop **on battery (charger unplugged)**.
4. **Only one program may control the arm.** Close any other script that uses it (it may move the arm home as soon as it starts).
5. **Know your physical stop.** Space/Esc first. The arm's power switch cuts everything, but the arm then falls: support it with one hand.
6. **Before closing the app, put down anything the gripper holds.** On exit the arm parks at home and opens the gripper.
7. **Don't leave the gripper squeezing an object for long.** A blocked servo heats up; if needed lower `gripper.close` in `config/arm_config.json`.

The arm never loses torque in software (no `servoOff()`): on stop, disarm, disconnect or exit it holds its pose. Voltage is checked every 10 s; below 6.6 V motion stops.

## Bring-up, one stage at a time (do not skip stages)

**Stage 1 - no arm, no armband.** Open **Arm control**, tick **Simulate the arm**, *Connect arm* (the simulated arm homes by itself), switch the control mode to *Keyboard only*, press **ARM** and drive the simulated arm with the keys or the on-screen buttons. Practise Space/Esc.

**Stage 2 - real armband, simulated arm.** On **Gesture model** pick a model trained on the person who wears the armband. On **Armband** connect over Wi-Fi (or USB). Keep **Simulate** ticked and ARM. Do each gesture and check on **Arm control** (Live) / **Live monitor** that the right action appears and that Rest reliably shows HOLD. Raise *Min. confidence* if you see false moves. Models trained on someone else are much weaker.

**Stage 3 - real arm, keyboard only.** Plug in the arm and switch it on, then run the read-only check:

```bash
python3 tools/check_arm.py            # prints voltage + positions, does not move
python3 tools/check_arm.py --nudge 6  # base moves +30 units and back (asks first); also 5 = shoulder, 4 = elbow
python3 tools/check_arm.py --glide 6  # smooth streamed sweep +80 and back: should look like one continuous motion
```

Use the nudges to confirm which way "+" goes; if Left/Right or Up/Down are reversed, flip the sign in `config/arm_config.json` (`gestures.*.direction` or `vertical_joints.*.up_direction`). Then untick **Simulate**, *Connect arm* (it moves slowly to home by itself, so stand clear), mode *Keyboard only*, ARM, and drive each joint to both ends of its range at speed 50. If a limit would let the arm hit the table or itself, tighten it in `config/arm_config.json` and restart.

**Stage 4 - armband and real arm.** Mode *Keys + gestures* (or *Gestures only*), speed 50-100, ARM, small movements first, a finger near the Space bar.

## More documentation

`docs/CHANGES_AND_ARCHITECTURE.md` - codebase analysis, architecture before/after, issue list, what changed (and what was *not* verified), hardware validation checklist, limitations.
