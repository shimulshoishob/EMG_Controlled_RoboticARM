"""Which network interfaces does this computer have, and which one reaches the armband?

The old app guessed broadcast addresses from ``gethostname()`` (wrong on macOS and Windows,
where that often returns a loopback or virtual-adapter address) and sent a single
``255.255.255.255`` datagram, which the OS sends out of only one adapter. Here every real
interface is enumerated with its netmask so discovery can use the right directed broadcast.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

try:
    import psutil
except ImportError:  # optional: a socket-only fallback keeps the app working
    psutil = None

# Interface names that are tunnels/virtual adapters, not the Wi-Fi/Ethernet the armband is on.
_VIRTUAL_NAME = re.compile(
    r"^(utun|ppp|tun|tap|wg|ipsec|gif|stf|awdl|llw|bridge|vmnet|vboxnet|veth|docker|br-|lo)"
    r"|vpn|vethernet|virtualbox|vmware|hyper-v|tailscale|zerotier|wireguard|loopback|bluetooth|npcap",
    re.IGNORECASE)
_VPN_NAME = re.compile(r"^(utun|ppp|tun|wg|ipsec)|vpn|tailscale|zerotier|wireguard|anyconnect|globalprotect",
                       re.IGNORECASE)


@dataclass
class Interface:
    name: str
    ip: str
    netmask: str
    broadcast: str
    is_up: bool = True
    is_virtual: bool = False
    is_vpn: bool = False

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(f"{self.ip}/{self.netmask}", strict=False)

    @property
    def cidr(self) -> str:
        return str(self.network)

    def contains(self, ip: str) -> bool:
        try:
            return ipaddress.IPv4Address(ip) in self.network
        except ValueError:
            return False


def list_interfaces(include_loopback: bool = False) -> List[Interface]:
    """IPv4 interfaces that are up and have a usable address (link-local 169.254 excluded)."""
    found: List[Interface] = []
    if psutil is not None:
        try:
            stats = psutil.net_if_stats()
            for name, addrs in psutil.net_if_addrs().items():
                st = stats.get(name)
                if st is not None and not st.isup:
                    continue
                for addr in addrs:
                    if addr.family != socket.AF_INET or not addr.address or not addr.netmask:
                        continue
                    if addr.address.startswith("169.254.") or addr.address == "0.0.0.0":
                        continue
                    loop = addr.address.startswith("127.")
                    if loop and not include_loopback:
                        continue
                    try:
                        net = ipaddress.IPv4Network(f"{addr.address}/{addr.netmask}", strict=False)
                    except ValueError:
                        continue
                    bcast = addr.broadcast or str(net.broadcast_address)
                    found.append(Interface(name, addr.address, addr.netmask, bcast, True,
                                           bool(_VIRTUAL_NAME.search(name)) or loop,
                                           bool(_VPN_NAME.search(name))))
        except Exception:
            found = []
    if not found:
        ip = primary_local_ip()
        if ip and (include_loopback or not ip.startswith("127.")):
            net = ipaddress.IPv4Network(f"{ip}/24", strict=False)   # best guess without psutil
            found.append(Interface("default", ip, "255.255.255.0", str(net.broadcast_address)))
    # Real adapters first, so discovery tries Wi-Fi/Ethernet before virtual ones.
    found.sort(key=lambda i: (i.is_virtual, i.name))
    return found


def primary_local_ip() -> Optional[str]:
    """The address this computer uses for its default route (no packet is sent)."""
    for target in ("192.0.2.1", "10.255.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target, 9))
                ip = s.getsockname()[0]
            if ip and ip != "0.0.0.0":
                return ip
        except OSError:
            continue
    return None


def local_ip_for(target_ip: str) -> str:
    """The local address the OS would use to reach ``target_ip`` (what the armband must send to)."""
    if target_ip.startswith("127."):
        return "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((target_ip, CONTROL_PORT_FOR_ROUTE))
            ip = s.getsockname()[0]
        if ip and ip != "0.0.0.0":
            return ip
    except OSError:
        pass
    return primary_local_ip() or "127.0.0.1"


CONTROL_PORT_FOR_ROUTE = 5001


def interface_for(target_ip: str, interfaces: Optional[List[Interface]] = None) -> Optional[Interface]:
    interfaces = interfaces if interfaces is not None else list_interfaces(include_loopback=True)
    for iface in interfaces:
        if iface.contains(target_ip):
            return iface
    return None


def same_subnet(target_ip: str, interfaces: Optional[List[Interface]] = None) -> bool:
    return interface_for(target_ip, interfaces) is not None


def vpn_interfaces(interfaces: Optional[List[Interface]] = None) -> List[Interface]:
    interfaces = interfaces if interfaces is not None else list_interfaces()
    return [i for i in interfaces if i.is_vpn]


def is_valid_ipv4(text: str) -> bool:
    try:
        ipaddress.IPv4Address(text.strip())
        return True
    except ValueError:
        return False


def current_wifi_ssid() -> Optional[str]:
    """Best-effort name of the Wi-Fi network this computer is on (to pre-fill the setup form).

    Returns None when it cannot be read; macOS hides it without Location permission."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True,
                                 timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            match = re.search(r"^\s*SSID\s*:\s*(.+?)\s*$", out, re.MULTILINE)
            return match.group(1) if match else None
        if sys.platform == "darwin":
            for dev in ("en0", "en1"):
                out = subprocess.run(["networksetup", "-getairportnetwork", dev], capture_output=True,
                                     text=True, timeout=3).stdout
                match = re.search(r"Current Wi-Fi Network:\s*(.+)", out)
                if match:
                    return match.group(1).strip()
            return None
        out = subprocess.run(["iwgetid", "-r"], capture_output=True, text=True, timeout=3).stdout.strip()
        return out or None
    except Exception:
        return None
