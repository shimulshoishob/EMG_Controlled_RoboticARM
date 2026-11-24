<div align="center">

# 🧪 `tests/`

**Automated tests: no hardware, no screen needed**

[↑ Project](../README.md)

</div>

```bash
python -m unittest discover tests        # or:  python -m pytest tests -q
```

The UI tests run Qt **offscreen** (`QT_QPA_PLATFORM=offscreen` is set for you). A full run takes a couple of minutes, mostly the network-recovery tests.

| File | What it covers |
|---|---|
| [`test_armband.py`](test_armband.py) | protocol parsing, HMAC and framing, discovery, Wi-Fi and USB links, loss/reboot/silence recovery, wrong key, provisioning, against the software armband |
| [`test_safety.py`](test_safety.py) | driver limits and speed cap, homing, motion streamer, gesture filter (votes, confidence, lockout, stale guards), one-shot actions, teleop modes |
| [`test_motion_jitter.py`](test_motion_jitter.py) | smoothness of the commanded motion, anti-jitter behaviour |
| [`test_mapping.py`](test_mapping.py) | gesture resolution, mapping validation and persistence, command generation |
| [`test_keyboard.py`](test_keyboard.py) | hold/release/combine keys, focus rules, re-assignment, STOP, modes |
| [`test_integration.py`](test_integration.py) | the real window: navigation, mapping editor, safety on link/arm failure, configuration files, model list |
| [`test_ui_smoke.py`](test_ui_smoke.py) | the real window end to end: streaming through the UI, wrong key message, auto-home and gripper toggle, hidden sections |
| [`test_model_loading.py`](test_model_loading.py) | loading gesture bundles: decision rate, sidecar info, friendly errors |
| [`fake_armband.py`](fake_armband.py) | **helper, not a test**: a software copy of the armband firmware that can lose replies and data, reboot and go silent. Run alone: `python tests/fake_armband.py --key <key>` |

Training-side tests live in [`Model_Training/tests`](../Model_Training/tests/README.md).
