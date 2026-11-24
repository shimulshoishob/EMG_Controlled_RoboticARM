<div align="center">

# 📟 `EMG_IMU_WIFI_SETUP_V2`

**8-channel EMG + BNO080 IMU streamer for the ESP32-S3** · firmware `emg-imu-2.0.0` · protocol `BWIM` v2

[↑ firmware](../README.md) · [Network & protocol](../../docs/NETWORK.md)

</div>

## What it does

- samples **8 EMG channels at 500 Hz** (one frame every 2000 µs) on ADC pins `1, 2, 4, 5, 6, 7, 9, 10`;
- reads a **BNO080** IMU over SPI (rotation vector, requested every 5 ms ≈ 200 Hz) and attaches roll/pitch/yaw to each frame;
- batches **5 frames per packet** (308 bytes, 100 packets/s) to avoid hundreds of tiny UDP packets;
- joins your Wi-Fi as a station (Wi-Fi sleep off) and answers **discovery and authenticated control** on UDP 5001, then streams to the laptop;
- offers the same data over **USB serial** (921 600 baud, COBS + CRC-16) and accepts first-time Wi-Fi setup (`PROVISION`) over USB;
- keeps health counters (queue drops, late samples, send failures, Wi-Fi reconnects) in every packet header.

It uses both cores: one for ADC sampling, one for IMU polling, USB, Wi-Fi control and streaming.

## Wiring (BNO080 over SPI)

| Signal | GPIO |
|---|---|
| RST | 11 |
| INT | 12 |
| CS | 13 |
| MOSI | 14 |
| MISO | 15 |
| SCK | 16 |

## Build and flash

1. Install the Arduino IDE with **ESP32 board support** and select your **ESP32-S3** board.
2. Install the library **SparkFun BNO08x Arduino Library**.
3. Copy `secrets.example.h` to **`secrets.h`** and set a long random key:

   ```bash
   python3 -c "import secrets; print(secrets.token_hex(24))"
   ```

4. Upload `EMG_IMU_WIFI_SETUP_V2.ino`.
5. In the desktop app (*Armband*): set up Wi-Fi over USB once, then enter the same key (or *Import from secrets.h*).

## Security model

- Control messages are authenticated with **HMAC-SHA256** over `COMMAND|challenge[|payload]`, keyed with `DEVICE_ACCESS_KEY_VALUE`; the key is never sent.
- A challenge belongs to one remote *(IP, port)*, is single-use, and the firmware keeps 4 challenge slots.
- The stream stops after 15 s without an authorised command (the app sends a keepalive every 3 s).
- Wi-Fi credentials are stored on the device (stored form prefixed `enc1`), not in the repository.

## Files

| File | Purpose |
|---|---|
| `EMG_IMU_WIFI_SETUP_V2.ino` | the firmware |
| `secrets.example.h` | template for `secrets.h` (gitignored) |
