<div align="center">

# 📡 `armband/`

**Everything between the ESP32 armband and a stream of EMG samples**

[↑ emg_arm](../README.md) · [Network & connectivity](../../../docs/NETWORK.md)

</div>

Pure Python (no Qt). Links report through callbacks from background threads; the UI wraps them in Qt signals ([`ui/bridge.py`](../ui/bridge.py)).

| Module | What it does |
|---|---|
| [`protocol.py`](protocol.py) | `BWIM` v2 (and v1) packet parsing, COBS + CRC-16 USB framing, `HELLO`/`INFO` parsing, HMAC-SHA256 signing, access-key checks |
| [`control.py`](control.py) | `ControlClient` (one socket for `CHALLENGE` + signed command, retries with fresh challenges) and `discover()` (directed broadcast per adapter, unicast /24 sweep fallback) |
| [`link.py`](link.py) | `WifiLink` and `UsbLink`: start/stop, keepalive, stall detection, automatic recovery, loss and rate statistics, `LinkState` |
| [`netinfo.py`](netinfo.py) | list real network interfaces with netmasks, same-subnet test, VPN detection, current Wi-Fi name |
| [`serial_setup.py`](serial_setup.py) | list USB ports, probe the armband (`INFO`), send Wi-Fi credentials (`PROVISION`) |
| [`diagnostics.py`](diagnostics.py) | the *Connection setup* checks: network, address, subnet, key, reachability, firewall hints |

## Key numbers

| Item | Value |
|---|---|
| Sample rate | 500 Hz, 8 EMG channels (12-bit) + roll/pitch/yaw |
| Packet | 5 frames, 308 bytes, 100 packets per second |
| Ports | UDP 5000 (data), UDP 5001 (control), USB serial 921 600 baud |
| Keepalive | every 3 s (the firmware stops streaming after 15 s without one) |
| Stall detection | no data for 1.5 s |

## Link contract

```python
link = WifiLink(ip, key, on_frames, on_state, on_stats)   # or UsbLink(port, ...)
link.start()        # returns at once
# on_frames(ndarray[n, 11])   8 raw EMG counts + roll, pitch, yaw
# on_state(LinkState, text)   idle · connecting · streaming · stalled · reconnecting · failed
# on_stats(LinkStats)         once per second
link.stop()
```

The links never give up after a network problem (they back off and retry), except for errors retrying cannot fix, such as a wrong access key.
