<div align="center">

# 📄 `ui/pages/`

**One module per screen**

[↑ ui](../README.md)

</div>

| Page (sidebar) | Module | What you do there |
|---|---|---|
| **Arm control** | [`control_page.py`](control_page.py) | the main screen: connect the arm, ARM / DISARM, live status and servo positions, control options (mode, speed, confidence), advanced keyboard controls |
| *(card on Arm control)* | [`arm_page.py`](arm_page.py) | connect/disconnect the xArm (it goes home by itself), simulate it, one gripper open/close button |
| *(advanced, on Arm control)* | [`keyboard_page.py`](keyboard_page.py) | `DrivePanel`: every movement with its on-screen **Hold** button and its two re-assignable keys |
| **Gesture model** | [`model_page.py`](model_page.py) | upload several `.joblib` models, clear the list, pick one; table of classes with training windows and the robot action |
| **Armband** | [`armband_page.py`](armband_page.py) | Wi-Fi or USB: set up Wi-Fi once, scan, access key, connect, *Connection setup* check, live signal (hidden until asked) |
| **Live monitor** | [`monitor_page.py`](monitor_page.py) | 8-channel EMG, model output, confidence, stable gesture, transition history |
| **Gesture mapping** | [`mapping_page.py`](mapping_page.py) | choose what each of the ten gestures does, with live preview and validation |
| **Settings & data** | [`data_page.py`](data_page.py) | theme, configuration export/import, raw-signal recording, servo calibration, advanced motion smoothing |

Pages hold widgets and emit signals; the logic that reacts to them lives in [`main_window.py`](../main_window.py).
