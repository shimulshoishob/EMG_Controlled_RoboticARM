"""Armband protocol + networking tests. No hardware: a software copy of the firmware stands in.

    python -m unittest tests.test_armband -v
"""

import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emg_arm.armband import control, protocol as P  # noqa: E402
from emg_arm.armband.fake import FakeArmband, FakeUsbDevice, SignalSource, build_packet  # noqa: E402
from emg_arm.armband.link import LinkState, UsbLink, WifiLink  # noqa: E402

KEY = "b6807509df72ab2088ba4945138391657aa2d6c61b5f7397"
CONTROL_PORT = 25001      # not the real 5001: tests must not fight a real armband / app
STREAM_PORT = 25100


def make_packet(seq=0, frames=5):
    return build_packet(seq, SignalSource(1).frames(frames))


class ProtocolTests(unittest.TestCase):
    def test_sizes_match_firmware(self):
        self.assertEqual(P.V2_PACKET_SIZE, 308)
        self.assertEqual(P.V1_PACKET_SIZE, 252)

    def test_parse_v2_packet(self):
        pkt = P.parse_packet(make_packet(7))
        self.assertEqual((pkt.version, pkt.sequence, pkt.n_frames), (2, 7, 5))
        self.assertEqual(pkt.frames.shape, (5, 11))
        self.assertTrue(np.all(pkt.frames[:, :8] <= 4095))
        self.assertEqual(list(pkt.frame_ids), [0, 1, 2, 3, 4])

    def test_parse_v1_packet_still_works(self):
        header = P.V1_HEADER.pack(b"BWIM", 1, 5, 48, 9)
        frame = np.zeros(5, dtype=P.FRAME_V1_DTYPE)
        frame["emg"][:] = 2000
        pkt = P.parse_packet(header + frame.tobytes())
        self.assertEqual((pkt.version, pkt.sequence), (1, 9))
        self.assertEqual(float(pkt.frames[0, 0]), 2000.0)

    def test_rejects_bad_packets(self):
        good = make_packet()
        for bad in (b"", b"XXXX" + good[4:], good[:-1], good + b"\0", good[:4] + b"\x09" + good[5:]):
            with self.assertRaises(P.ProtocolError):
                P.parse_packet(bad)

    def test_firmware_counters_are_read(self):
        frames = SignalSource(1).frames(5)
        pkt = P.parse_packet(build_packet(1, frames, counters=(3, 4, 5, 6)))
        self.assertEqual((pkt.queue_drops, pkt.acq_overruns, pkt.send_failures, pkt.wifi_reconnects), (3, 4, 5, 6))

    def test_cobs_round_trip(self):
        rng = np.random.default_rng(0)
        for n in (0, 1, 2, 253, 254, 255, 256, 600):
            data = bytes(rng.integers(0, 256, n, dtype=np.uint8))
            data2 = data.replace(b"\x00", b"\x01") if n % 2 else data      # with and without zeros
            for d in (data, data2, b"\0" * n):
                enc = P.cobs_encode(d)
                self.assertNotIn(0, enc)
                self.assertEqual(P.cobs_decode(enc), d)

    def test_crc_known_value(self):
        self.assertEqual(P.crc16_ccitt(b"123456789"), 0x29B1)       # CRC-16/CCITT-FALSE check value

    def test_usb_frame_reader_handles_noise_and_text(self):
        frames = [P.encode_usb_frame(make_packet(i)) for i in range(4)]
        stream = b"boot log line\n" + frames[0] + b"INFO|X|Y|1|1|fw|500\n" + b"\x00" + frames[1][:100]
        reader = P.UsbFrameReader()
        got = reader.feed(stream[:57]) + reader.feed(stream[57:])      # split mid-frame
        got += reader.feed(frames[1][100:] + b"\xde\xad\xbe\xef\x00" + frames[2] + frames[3])
        self.assertEqual([p.sequence for p in got], [0, 1, 2, 3])
        self.assertTrue(any(t.startswith("INFO|") for t in reader.take_text()))
        self.assertGreaterEqual(reader.bad_frames, 1)

    def test_usb_frame_crc_corruption_is_dropped(self):
        frame = bytearray(P.encode_usb_frame(make_packet(1)))
        frame[40] ^= 0x10
        reader = P.UsbFrameReader()
        self.assertEqual(reader.feed(bytes(frame)), [])

    def test_parse_hello_and_info(self):
        info = P.parse_hello("HELLO|AA|BioWave|STA|192.168.1.9|1|0|emg-imu-2.0.0|500", "192.168.1.9")
        self.assertEqual((info.name, info.sample_rate, info.imu_ready, info.streaming), ("BioWave", 500, True, False))
        old = P.parse_hello("HELLO|AA|BioWave|STA|192.168.1.9|0|1|fw1", "1.2.3.4")        # 8 fields (v1 firmware)
        self.assertEqual((old.sample_rate, old.imu_ready, old.streaming), (500, False, True))
        serial_info = P.parse_info("INFO|AA|BioWave|1|0|emg-imu-2.0.0|500", "/dev/x")
        self.assertEqual((serial_info.wifi_saved, serial_info.port), (False, "/dev/x"))
        with self.assertRaises(P.ProtocolError):
            P.parse_hello("garbage", "1.1.1.1")

    def test_provision_line_stays_within_firmware_limit(self):
        line = P.build_provision_line("My Café WiFi|2", "p@ss+word%1" * 5)
        self.assertLessEqual(len(line), P.MAX_SERIAL_LINE)
        self.assertEqual(line.count("|"), 2 + 0)       # '|' in the name is escaped
        from urllib.parse import unquote_plus
        self.assertEqual(unquote_plus(line.split("|")[1]), "My Café WiFi|2")
        # worst case the old quote(safe='') encoding would overflow: 32-byte SSID + 63-byte password of symbols
        worst = P.build_provision_line("é" * 16, "!@#$%^&*()" * 6)
        self.assertLessEqual(len(worst), P.MAX_SERIAL_LINE)
        for ssid, pw in (("", "x" * 10), ("a" * 33, "x" * 10), ("ok", "short"), ("ok", "x" * 64)):
            with self.assertRaises(P.ProtocolError):
                P.build_provision_line(ssid, pw)
        P.build_provision_line("open-network", "")        # open networks are allowed

    def test_secrets_h_import_and_key_checks(self):
        text = '#pragma once\n#define DEVICE_ACCESS_KEY_VALUE "%s"\n' % KEY
        self.assertEqual(P.read_key_from_secrets_h(text), KEY)
        with self.assertRaises(P.ProtocolError):
            P.read_key_from_secrets_h("nothing here")
        self.assertIsNone(P.check_key(KEY))
        for bad in ("", "short", P.PLACEHOLDER_KEY, "has space in it but is long enough"):
            self.assertIsNotNone(P.check_key(bad))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.dev = FakeArmband(KEY, control_port=CONTROL_PORT, seed=3).start()
        self.addCleanup(self.dev.stop)

    def client(self, key=KEY, **kw):
        c = control.ControlClient("127.0.0.1", key, port=CONTROL_PORT, timeout=0.4, **kw)
        self.addCleanup(c.close)
        return c

    def test_start_ping_stop(self):
        c = self.client()
        self.assertEqual(c.start_stream("127.0.0.1", STREAM_PORT), ["STARTED"])
        self.assertEqual(self.dev.target, ("127.0.0.1", STREAM_PORT))
        self.assertEqual(c.ping(), ["PONG"])
        self.assertEqual(c.stop_stream(), ["STOPPED"])
        self.assertIsNone(self.dev.target)

    def test_old_one_socket_per_message_behaviour_is_rejected_by_firmware(self):
        """Regression: CHALLENGE and START from different source ports must fail - that is what the
        old app did on every connect attempt."""
        s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(s1.close)
        self.addCleanup(s2.close)
        s1.settimeout(1)
        s2.settimeout(1)
        s1.sendto(b"CHALLENGE", ("127.0.0.1", CONTROL_PORT))
        challenge = s1.recvfrom(100)[0].decode().split("|")[1]
        s2.sendto(P.build_authed("PING", challenge, KEY).encode(), ("127.0.0.1", CONTROL_PORT))
        self.assertEqual(s2.recvfrom(100)[0], b"ERR|AUTH_FAILED")

    def test_wrong_key_is_reported_as_auth_failure(self):
        with self.assertRaises(control.AuthFailed):
            self.client(key="x" * 32).start_stream("127.0.0.1", STREAM_PORT)

    def test_survives_lost_replies(self):
        self.dev.reply_drop_probability = 0.3
        c = self.client(attempts=14)
        for _ in range(10):
            self.assertEqual(c.authed("PING"), ["PONG"])

    def test_silent_device_times_out_with_plain_message(self):
        self.dev.silent = True
        with self.assertRaises(control.DeviceTimeout) as ctx:
            self.client(attempts=2).ping()
        self.assertIn("No answer", str(ctx.exception))

    def test_two_clients_do_not_invalidate_each_other(self):
        a, b = self.client(), self.client()
        self.assertEqual(a.ping(), ["PONG"])
        self.assertEqual(b.ping(), ["PONG"])

    def test_bad_ip_and_port_are_rejected(self):
        c = self.client()
        with self.assertRaises(control.DeviceRejected) as ctx:
            c.start_stream("999.1.1.1", STREAM_PORT)
        self.assertEqual(ctx.exception.code, "BAD_IP")
        with self.assertRaises(control.DeviceRejected) as ctx:
            c.start_stream("127.0.0.1", 70000)
        self.assertEqual(ctx.exception.code, "BAD_PORT")

    def test_discovery_finds_device(self):
        report = control.discover(timeout=1.5, port=CONTROL_PORT, sweep=False)
        self.assertEqual([d.device_id for d in report.found], ["AABBCCDDEEFF"])
        self.assertEqual(report.found[0].firmware, "emg-imu-2.0.0")

    def test_discovery_survives_lost_replies(self):
        self.dev.reply_drop_probability = 0.5
        found = sum(bool(control.discover(timeout=1.5, port=CONTROL_PORT, sweep=False).found) for _ in range(5))
        self.assertGreaterEqual(found, 4)

    def test_discovery_with_nothing_there(self):
        self.dev.silent = True
        t0 = time.monotonic()
        report = control.discover(timeout=0.9, port=CONTROL_PORT, sweep=False)
        self.assertEqual(report.found, [])
        self.assertLess(time.monotonic() - t0, 3)


class Recorder:
    def __init__(self):
        self.frames = 0
        self.states = []
        self.lock = threading.Lock()

    def on_frames(self, f):
        with self.lock:
            self.frames += len(f)

    def on_state(self, s, m):
        self.states.append((s, m))

    def on_stats(self, s):
        pass


def wait_until(cond, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


class WifiLinkTests(unittest.TestCase):
    def setUp(self):
        self.dev = FakeArmband(KEY, control_port=CONTROL_PORT, seed=5).start()
        self.addCleanup(self.dev.stop)
        self.rec = Recorder()

    def link(self, key=KEY, **kw):
        link = WifiLink("127.0.0.1", key, self.rec.on_frames, self.rec.on_state, self.rec.on_stats,
                        device_id="AABBCCDDEEFF", stream_port=STREAM_PORT, control_port=CONTROL_PORT,
                        keepalive_interval_s=0.3, **kw)
        self.addCleanup(link.stop)
        return link

    def test_streams_at_full_rate_without_loss(self):
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8), link.message)
        n0, t0 = self.rec.frames, time.monotonic()
        time.sleep(2.0)
        rate = (self.rec.frames - n0) / (time.monotonic() - t0)
        self.assertGreater(rate, 400)                # 500 Hz target
        self.assertEqual(link.stats.lost_packets, 0)
        self.assertEqual(link.stats.bad_packets, 0)

    def test_wrong_key_fails_with_clear_message(self):
        link = self.link(key="z" * 32)
        link.start()
        self.assertTrue(link.wait_for(LinkState.FAILED, 8))
        self.assertIn("access key", link.message.lower())
        self.assertEqual(link.last_error_kind, "auth")

    def test_connects_through_lossy_control_replies(self):
        self.dev.reply_drop_probability = 0.35
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 15), link.message)

    def test_counts_lost_data_packets(self):
        self.dev.data_drop_probability = 0.2
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8))
        time.sleep(2.0)
        self.assertGreater(link.stats.lost_packets, 5)
        self.assertEqual(link.state, LinkState.STREAMING)         # 20 % loss must not look like a disconnect
        self.assertGreater(link.stats.loss_percent, 5)

    def test_recovers_after_armband_reboot(self):
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8))
        before = self.rec.frames
        self.dev.reboot()                                           # forgets the target, like a power cycle
        self.assertTrue(wait_until(lambda: self.rec.frames > before + 1000, 12),
                        f"no recovery, state={link.state} {link.message}")
        self.assertEqual(link.state, LinkState.STREAMING)

    def test_recovers_when_armband_disappears_and_returns(self):
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8))
        self.dev.silent = True
        self.dev.target = None
        self.assertTrue(wait_until(lambda: link.state in (LinkState.STALLED, LinkState.RECONNECTING), 6))
        self.dev.silent = False
        self.assertTrue(link.wait_for(LinkState.STREAMING, 20), f"{link.state} {link.message}")

    def test_keepalive_prevents_firmware_timeout(self):
        self.dev.keepalive_timeout_s = 1.0          # firmware stops streaming if not pinged within 1 s
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8))
        before = self.rec.frames
        time.sleep(3.0)
        self.assertGreater(self.rec.frames - before, 1000)
        self.assertNotIn("keepalive timeout", self.dev.log)

    def test_stop_tells_armband_to_stop_and_frees_port(self):
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8))
        link.stop()
        self.assertIsNone(self.dev.target)
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(probe.close)
        probe.bind(("0.0.0.0", STREAM_PORT))               # port is free again

    def test_uses_next_port_when_data_port_is_taken(self):
        blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        blocker.bind(("0.0.0.0", STREAM_PORT))
        self.addCleanup(blocker.close)
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8), link.message)
        self.assertEqual(link.stats.stream_port, STREAM_PORT + 1)

    def test_blocked_data_path_is_diagnosed_as_firewall(self):
        self.dev.data_drop_probability = 1.0              # armband accepts START but nothing arrives
        link = self.link()
        link.start()
        self.assertTrue(wait_until(lambda: link.last_error_kind == "firewall", 10))
        self.assertIn("firewall", link.diagnosis.lower())
        self.assertEqual(link.state, LinkState.STALLED)

    def test_ignores_packets_from_other_hosts(self):
        link = self.link()
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 8))
        # a packet from a different source address must not be delivered: emulate with a bad packet instead
        junk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(junk.close)
        junk.sendto(b"not a packet", ("127.0.0.1", link.stats.stream_port))
        junk.sendto(make_packet(1)[:-3], ("127.0.0.1", link.stats.stream_port))
        time.sleep(0.3)
        self.assertGreaterEqual(link.stats.bad_packets, 2)


@unittest.skipUnless(hasattr(os, "openpty") and sys.platform != "win32", "needs a pty")
class UsbLinkTests(unittest.TestCase):
    def test_streams_binary_over_a_serial_port(self):
        master, slave = os.openpty()
        import tty
        tty.setraw(master)
        tty.setraw(slave)
        dev = FakeUsbDevice(master, seed=2, boot_delay_s=0.8).start()
        self.addCleanup(dev.stop)
        rec = Recorder()
        link = UsbLink(os.ttyname(slave), rec.on_frames, rec.on_state, rec.on_stats, baud=115200)
        self.addCleanup(link.stop)
        link.start()
        self.assertTrue(link.wait_for(LinkState.STREAMING, 10), f"{link.state} {link.message}")
        n0 = rec.frames
        time.sleep(1.5)
        self.assertGreater(rec.frames - n0, 500)
        self.assertEqual(link.stats.lost_packets, 0)
        link.stop()
        self.assertFalse(dev.streaming)

    def test_probe_and_provision(self):
        from emg_arm.armband import serial_setup
        master, slave = os.openpty()
        import tty
        tty.setraw(master)
        tty.setraw(slave)
        dev = FakeUsbDevice(master, boot_delay_s=0.5).start()
        self.addCleanup(dev.stop)
        info = serial_setup.probe(os.ttyname(slave), timeout=6, baud=115200)
        self.assertEqual((info.name, info.firmware), ("BioWave-EMG-IMU-2026", "emg-imu-2.0.0"))
        serial_setup.provision(os.ttyname(slave), "Lab WiFi+5GHz", "p|ss word 123", baud=115200)
        self.assertEqual(dev.provisioned, ("Lab WiFi+5GHz", "p|ss word 123"))


if __name__ == "__main__":
    unittest.main()
