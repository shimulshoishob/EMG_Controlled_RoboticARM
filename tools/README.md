<div align="center">

# 🛠️ `tools/`

**Stand-alone helpers (not part of the app)**

[↑ Project](../README.md)

</div>

Run from the project root. `check_arm.py` sets up the project's `.venv` automatically; the others use the Python you start them with.

| Tool | Needs hardware? | Purpose |
|---|---|---|
| [`check_arm.py`](check_arm.py) | the xArm | read-only check (voltage, servo positions); `--nudge N` moves servo *N* by +30 and back to confirm direction; `--glide N` streams a smooth sweep to judge smoothness |
| [`jitter_report.py`](jitter_report.py) | no | streams a constant-speed base jog through the real `MotionStreamer` into a simulated arm and reports how steady the *commanded* speed is (software contribution only) |
| [`native_key_check.py`](native_key_check.py) | no (macOS) | injects real OS key events into the real window to check the keyboard path |

```bash
python3 tools/check_arm.py              # prints voltage + positions, does not move
python3 tools/check_arm.py --nudge 6    # base +30 units and back (asks first); 5 = shoulder, 4 = elbow
python3 tools/check_arm.py --glide 6    # smooth sweep +80 and back
python3 tools/jitter_report.py          # software-side jitter evidence
```

> Close the desktop app before using `check_arm.py`: only one program may control the arm.
