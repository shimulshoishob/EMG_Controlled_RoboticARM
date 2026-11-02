# EMG_Controlled_RoboticARM
Real-Time Multimodal EMG–IMU Control of a 6-DOF Robotic Arm: Performance, Responsiveness, and Reliability Evaluation


## Run the EMG controller

`emg_arm_control.py` is a single-file, stdlib-only app with a browser UI (http://127.0.0.1:8765).

```bash
./run.sh                       # picks a Python that has xarm + hidapi, then open the page and press "Connect both"
./run.sh --arm usb --band bridge
```

**Armband links** (same ones BioWave `main.py` offers; use only one at a time, they share the device):

| Link | When to use | Notes |
|---|---|---|
| BioWave app (gestures) | easiest, uses your trained model | start BioWave `main.py` -> load model -> Realtime prediction; this app listens on UDP 127.0.0.1:5005 |
| BioWave Wi-Fi (direct) | no BioWave app | close BioWave app, press Scan, pick device, enter the device access key (or `BIOWAVE_KEY` / `--key`) |
| BioWave USB cable (direct) | wired | close BioWave app, 921600 baud, port auto-detected |
| Myo / Simulated | | Myo untested on hardware |

**Arm:** xArm over USB-HID (auto-detected) or COM port. Failed links show the reason and retry every 4 s; the last choices are restored on the next start (`settings.json`, git-ignored). "More -> Run checks" lists the Python in use, installed packages and what the USB bus sees.

**Gesture mode:** each BioWave gesture maps to an arm action (editable in the UI): Left/Right = Base, Up/Down = Elbow, fist = close gripper, pinch = open, Victory/Thumbs-up = Reach, Double fist (hold) = Home, Rest = stop. The arm stops if predictions stop arriving or confidence is below the slider.
**Raw EMG mode** (Wi-Fi / USB / Myo): flex = active joint `+`, extend = `-`, hold both ~0.5 s = next joint; calibrate Rest / Flex / Extend first. *EMG + Tilt* maps arm roll/pitch to Base/Reach.
**Always available:** E-STOP (Space; freezes commands, arm holds position, click again to release), Home, gripper, per-joint sliders and jog, 3 saved poses, speed, CSV logging (`logs/`), Manual mode.

Tested against simulated devices (a fake BioWave Wi-Fi device with signed START, a pty serial stream, UDP gestures). The real xArm and Myo paths are untested on hardware, so start slowly with the arm clear of obstacles.
