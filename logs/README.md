<div align="center">

# 📝 `logs/`

**Control-session logs and the background log**

[↑ Project](../README.md)

</div>

| File | Written | Contents |
|---|---|---|
| `session_<date>_<time>.csv` | each time the arm is connected | one row per event for analysis |
| `app.log` | always (rotating, 1 MB × 3) | quiet background log for support; **not shown in the app** |

## Session CSV columns

| Column | Meaning |
|---|---|
| `time_s` | seconds since the session started |
| `event` | `arm_connected`, `armed`, `disarmed`, `prediction`, `move`, `gripper`, `switch_joint`, `goto`, `home`, `park`, `error` … |
| `label`, `confidence` | what the model predicted |
| `gesture` | the stable gesture after the vote filter |
| `motion`, `source` | the movement and where it came from (gesture, keyboard, button) |
| `note` | free text (for example the reason for a stop) |
| `targets` | JSON of the servo targets at that moment |

The contents of this folder are **not committed** (see `.gitignore`); this README is.
