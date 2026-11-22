"""Connection doctor: runs the same steps a person would, one at a time, and says what to fix.  No Qt."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from . import netinfo
from .control import (AuthFailed, ControlClient, ControlError, DeviceRejected, DeviceTimeout, NoRoute, discover)
from .protocol import STREAM_PORT, CONTROL_PORT, check_key

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"


@dataclass
class Check:
    title: str
    status: str
    detail: str
    fix: str = ""


def run_checks(device_ip: str, key: str, progress: Optional[Callable[[Check], None]] = None,
               stream_port: int = STREAM_PORT, control_port: int = CONTROL_PORT) -> List[Check]:
    """Run every check. ``progress`` is called with each result as soon as it is known."""
    results: List[Check] = []

    def add(title, status, detail, fix=""):
        c = Check(title, status, detail, fix)
        results.append(c)
        if progress:
            progress(c)
        return c

    # 1. adapters ------------------------------------------------------------------------
    interfaces = netinfo.list_interfaces(include_loopback=False)
    real = [i for i in interfaces if not i.is_virtual]
    if not interfaces:
        add("This computer's network", FAIL, "No active network connection was found.",
            "Join the same Wi-Fi network as the armband (2.4 GHz), then run the check again.")
    else:
        desc = ", ".join(f"{i.name} {i.ip}" for i in (real or interfaces))
        add("This computer's network", OK if real else WARN, f"Connected: {desc}.",
            "" if real else "Only virtual adapters are active. Connect to Wi-Fi or Ethernet.")

    # 2. VPN -------------------------------------------------------------------------------
    vpns = netinfo.vpn_interfaces(interfaces)
    if vpns:
        add("VPN / tunnel", WARN, "A VPN-like adapter is active: " + ", ".join(i.name for i in vpns) + ".",
            "Turn the VPN off while using the armband; it can hide the local network.")
    else:
        add("VPN / tunnel", OK, "No VPN detected.")

    if not device_ip:
        add("Armband address", FAIL, "No armband address yet.", "Press Scan on the Armband page, or type its IP address.")
        return results
    if not netinfo.is_valid_ipv4(device_ip):
        add("Armband address", FAIL, f"'{device_ip}' is not a valid IPv4 address.",
            "It looks like 192.168.1.42 (see the armband's Serial Monitor output or your router's device list).")
        return results

    # 3. same network ----------------------------------------------------------------------
    loopback = device_ip.startswith("127.")
    if loopback:
        add("Same network", OK, "Virtual armband on this computer.")
    elif netinfo.same_subnet(device_ip, interfaces):
        iface = netinfo.interface_for(device_ip, interfaces)
        add("Same network", OK, f"{device_ip} is on {iface.name} ({iface.cidr}).")
    else:
        add("Same network", FAIL, f"{device_ip} is not on any network this computer is connected to "
            f"({', '.join(i.cidr for i in real) or 'none'}).",
            "Put this computer on the same Wi-Fi as the armband, or re-send the Wi-Fi details with USB setup.")

    # 4. reachable ---------------------------------------------------------------------------
    client = ControlClient(device_ip, key or "x" * 16, port=control_port, timeout=1.0)
    try:
        try:
            info = client.hello(attempts=4)
            add("Armband answers", OK, f"{info.name} (firmware {info.firmware}, IMU {'ready' if info.imu_ready else 'NOT detected'}).",
                "" if info.imu_ready else "The IMU did not start: check the BNO080 wiring. EMG still works.")
        except (DeviceTimeout, NoRoute) as exc:
            add("Armband answers", FAIL, str(exc),
                "Check that the armband is powered and has joined the Wi-Fi (rescan for it), that the Wi-Fi is "
                "2.4 GHz, and that the router has no 'client isolation' / 'AP isolation' option enabled.")
            return results

        # 5. access key --------------------------------------------------------------------------
        problem = check_key(key)
        if problem:
            add("Access key", FAIL, problem, "Use 'Import secrets.h' to load the key automatically.")
            return results
        try:
            client.ping()
            add("Access key", OK, "The armband accepted the key.")
        except AuthFailed as exc:
            add("Access key", FAIL, str(exc), "Use 'Import secrets.h' and choose the secrets.h you flashed. If you "
                "changed it since, flash the firmware again.")
            return results
        except ControlError as exc:
            add("Access key", WARN, str(exc), "Try again; Wi-Fi may be dropping packets.")

        # 6. data port -------------------------------------------------------------------------
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port = None
        try:
            for candidate in range(stream_port, stream_port + 10):
                try:
                    sock.bind(("0.0.0.0", candidate))
                    port = candidate
                    break
                except OSError:
                    continue
            if port is None:
                add("Data port", FAIL, f"UDP ports {stream_port}-{stream_port + 9} are all in use.",
                    "Close other copies of this app.")
                return results
            add("Data port", OK if port == stream_port else WARN,
                f"Listening on UDP port {port}." + ("" if port == stream_port else f" (port {stream_port} was busy)"))

            # 7. real data test ---------------------------------------------------------------------
            sock.settimeout(0.3)
            client_ip = netinfo.local_ip_for(device_ip)
            try:
                client.start_stream(client_ip, port)
            except ControlError as exc:
                add("Start streaming", FAIL, str(exc), "")
                return results
            got = 0
            end = time.monotonic() + 3.0
            while time.monotonic() < end and got < 5:
                try:
                    data, addr = sock.recvfrom(2048)
                    if addr[0] == device_ip and data[:4] == b"BWIM":
                        got += 1
                except socket.timeout:
                    pass
                except OSError:
                    break
            if got:
                add("Data arrives", OK, f"Received {got}+ packets from {device_ip} (sent to {client_ip}:{port}).")
            else:
                add("Data arrives", FAIL,
                    f"The armband accepted the stream (to {client_ip}:{port}) but no data reached this computer.",
                    "A firewall is the usual cause. Windows: allow Python in 'Windows Defender Firewall' for Private "
                    "networks. macOS: System Settings > Network > Firewall > allow incoming connections for Python. "
                    "Also check the VPN, and that the network is not set to 'Public'.")
            try:
                client.stop_stream()
            except ControlError:
                pass
        finally:
            sock.close()
    finally:
        client.close()
    return results


def summarize(results: List[Check]) -> str:
    bad = [c for c in results if c.status == FAIL]
    if not bad:
        return "Everything checks out - you can connect."
    return f"Problem found: {bad[0].title}. {bad[0].fix or bad[0].detail}"
