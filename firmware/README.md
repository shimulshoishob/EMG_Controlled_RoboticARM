<div align="center">

# 🔌 `firmware/`

**Armband firmware (ESP32-S3)**

[↑ Project](../README.md) · [Network & protocol](../docs/NETWORK.md)

</div>

| Folder | Contents |
|---|---|
| [`EMG_IMU_WIFI_SETUP_V2/`](EMG_IMU_WIFI_SETUP_V2/README.md) | the Arduino sketch `emg-imu-2.0.0` and the example secrets file |

The app speaks the protocol this firmware defines; the matching code is in [`src/emg_arm/armband/`](../src/emg_arm/armband/README.md), and [`tests/fake_armband.py`](../tests/fake_armband.py) is a software copy of it for tests.

> **Never commit `secrets.h`.** It holds the device access key.
