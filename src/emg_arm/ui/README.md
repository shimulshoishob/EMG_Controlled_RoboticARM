<div align="center">

# 🖥️ `ui/`

**The desktop window: pages, keyboard, theme, dialogs**

[↑ emg_arm](../README.md) · [Pages](pages/README.md)

</div>

PyQt5, Fusion style on every platform, light / dark / pink themes, always-visible **STOP** button and status chips.

| Module | Role |
|---|---|
| [`main_window.py`](main_window.py) | the window, navigation, and the **control logic** that ties armband → model → gesture filter → motion streamer → arm together (arming, homing, parking on exit, emergency stop, battery check, recording) |
| [`bridge.py`](bridge.py) | `ArmbandBridge`: runs connect/disconnect off the GUI thread and re-emits link callbacks as Qt signals (`frames`, `state`, `stats`, `closed`) |
| [`keyboard.py`](keyboard.py) | `KeyMap` (configurable keys, saved to `keymap.json`) and `KeyboardController` (application-wide handler: hold to move, release to stop, never swallows keys meant for text boxes) |
| [`theme.py`](theme.py) | colour tokens and stylesheet, macOS/Windows fonts, light/dark/pink, title-bar colour |
| [`widgets.py`](widgets.py) | `Card`, `Banner`, `Chip`, `Segmented`, `Stat`, `SignalScope`… colours come from roles, never from the widgets |
| [`dialogs.py`](dialogs.py) | first-time Wi-Fi setup over USB, and the *Connection setup* check |
| [`tasks.py`](tasks.py) | run a blocking function off the GUI thread and get the result back as a signal |
| [`pages/`](pages/README.md) | one module per screen |

## Navigation

| Group | Page |
|---|---|
| **Robotic arm** | Arm control |
| **EMG & model** | Gesture model · Armband · Live monitor · Gesture mapping |
| **Settings** | Settings & data |

## Timers

| Timer | Period | Purpose |
|---|---|---|
| `control_timer` | 25 ms | `control_tick`: gestures/keys → motion |
| `ui_timer` | 250 ms | status chips, monitor banner |
| `battery_timer` | 10 s | arm voltage |
| `record_timer` | 500 ms | recording counter |
