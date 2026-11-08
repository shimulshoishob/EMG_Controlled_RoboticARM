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

## The five steps (left sidebar)

1. **Model** – pick the trained `.joblib`; the table shows what each gesture will do.
2. **Armband** – choose *Wi-Fi*, *USB cable* or *Virtual armband (demo, no hardware)*. Wi-Fi is four cards:
   1. **Give the armband your Wi-Fi** (first time only): plug it in by USB → *Check armband* → enter 2.4 GHz Wi-Fi name + password → *Send*.
   2. **Find your armband**: *Scan network* (or type its IP).
   3. **Access key**: *Import from secrets.h* (the key flashed into the firmware) or paste it.
   4. **Connect**: live 8-channel scope, frames/s, packet loss, firmware warning counters. *Connection doctor* explains any failure step by step.
3. **Robotic arm** – connect (or simulate), *Go home*.
4. **Control** – *ARM*, then gestures/keyboard. The red **STOP** button (Space / Esc) is on every page.
5. **Data & settings** – record raw EMG+IMU to CSV, open logs, theme.

Gestures, keyboard map, motion smoothing and safety rules are unchanged from before (see `config/arm_config.json`; the app refuses unsafe values at start-up).

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
  ml/       features · inference
  ui/       theme · widgets · bridge · dialogs · main_window · pages/
  session.py settings.py paths.py bootstrap.py app.py
tools/check_arm.py          read-only hardware check / nudge / glide
tests/                      73 tests, no hardware needed
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
