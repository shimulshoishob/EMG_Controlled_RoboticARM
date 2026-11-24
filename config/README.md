<div align="center">

# ⚙️ `config/`

**Safety limits and tuning for the arm and the gesture filter**

[↑ Project](../README.md) · [`arm/` package](../src/emg_arm/arm/README.md)

</div>

One file: [`arm_config.json`](arm_config.json). It is loaded and **validated at start-up**; unsafe values (for example a home pose outside the limits) stop the app with a clear message. User calibration (narrower servo ranges, a different rest pose) is stored in the per-user settings and can **never widen** these limits.

## `arm` section

| Key | Meaning |
|---|---|
| `servos.N.min / max` | soft limits per servo (xArm units 0-1000, ≈ 0.24° per unit): 1 gripper, 2 wrist rotate, 4 elbow, 5 shoulder, 6 base |
| `home`, `home_min_duration_ms` | home pose (inside the limits) and the minimum (slow) homing time, 3000 ms |
| `gripper.open / close / move_ms` | gripper positions and move time |
| `hard_max_speed_units_per_s`, `default_speed_units_per_s` | absolute speed cap (300) and start speed (100) |
| `control_period_ms` | control tick (25 ms) |
| `stream_period_ms`, `lookahead_ms` | motion thread period (40 ms) and how far ahead each joint is aimed (120 ms) |
| `accel / decel / jerk` | softness of starts and stops (500 / 1000 units/s² and 6000 units/s³) |
| `min_speed_units_per_s`, `min_lead_units`, `match_duration` | anti-jitter settings |
| `command_timeout_s` | the arm slows to a stop if commands stop for this long (0.25 s) |
| `min_battery_voltage` | below this the arm is stopped (6.6 V) |
| `park_on_exit`, `open_gripper_on_exit` | on exit: move home, then open the gripper |

## `gestures` and `vertical_joints`

Which servo each keyboard/gesture direction moves, and its sign: `left`/`right` → base (servo 6); `up`/`down` → the *selected* vertical joint, shoulder (servo 5, up = +) or elbow (servo 4, up = −). **If a direction turns out reversed on the real arm, flip the sign here** (find out with `tools/check_arm.py --nudge`).

## `filter` section

| Key | Meaning |
|---|---|
| `min_confidence` | global confidence gate (0.65) |
| `vote_window`, `stable_votes`, `hold_votes` | 5 votes, 4 to start, 3 to continue |
| `dropout_grace_s` | a brief miss inside a held motion is ignored (0.3 s) |
| `prediction_timeout_s`, `emg_timeout_s` | stale guards (0.3 s) |
| `grip_hold_s`, `switch_hold_s`, `grip_cooldown_s` | fist hold times and cooldown between one-shot actions |
| `max_continuous_move_s` | stuck-gesture lockout (6 s) |
| `reversal_guard_s` | pause before a joint is driven the other way (0.2 s) |
| `ramp_start`, `ramp_time_s` | speed at the start of a hold (50 %) and time to full speed (0.6 s) |

> Tighten a limit, never loosen it, unless you have verified the arm cannot hit the table or itself. Restart the app after editing.
