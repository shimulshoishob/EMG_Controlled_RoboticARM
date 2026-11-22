#!/usr/bin/env python3
"""Real-keyboard check on macOS: opens the real window (Cocoa), arms the *simulated* arm and injects
operating-system key events (Quartz CGEvent) for A, Left, W, then reports what the app saw.

Needs: macOS, and Accessibility permission for the terminal/IDE running this script
(System Settings -> Privacy & Security -> Accessibility). Nothing moves a real arm.

    python tools/native_key_check.py
"""

import ctypes
import ctypes.util
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from emg_arm.arm.driver import load_config  # noqa: E402
from emg_arm.settings import Settings  # noqa: E402
from emg_arm.ui import theme  # noqa: E402
from emg_arm.ui.main_window import MainWindow  # noqa: E402

KEYCODES = {"a": 0, "s": 1, "d": 2, "w": 13, "left": 123, "right": 124, "down": 125, "up": 126, "space": 49}


def load_quartz():
    cg = ctypes.CDLL(ctypes.util.find_library("ApplicationServices"))
    cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
    cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
    cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    return cg


def post(cg, code, down):
    ev = cg.CGEventCreateKeyboardEvent(None, code, down)
    cg.CGEventPost(0, ev)                                # kCGHIDEventTap


def main():
    if sys.platform != "darwin":
        print("macOS only")
        return 2
    app = QApplication(sys.argv)
    theme.apply_theme(app, "light")
    w = MainWindow(load_config(), Settings(Path(tempfile.mkdtemp()) / "s.json"))
    w.show()
    w.raise_()
    w.activateWindow()
    seen = []
    w.keyboard.key_seen.connect(seen.append)

    def pump(s):
        end = time.time() + s
        while time.time() < end:
            app.processEvents()
            time.sleep(0.005)

    pump(1.0)
    print("window active:", w.isActiveWindow())
    cp = w.pages["control"]
    cp.mode.setCurrentIndex(cp.mode.findData("keyboard"))
    w.toggle_arm()
    w.arm.homed = True
    w.toggle_armed()
    w.go_to("control")
    pump(0.3)
    cg = load_quartz()
    results = {}
    for name, servo in (("a", 6), ("left", 6), ("w", 5), ("d", 6), ("down", 5)):
        before = w.arm.targets[servo]
        post(cg, KEYCODES[name], True)
        pump(0.6)                                        # held: the OS starts auto-repeat after ~0.4 s
        held = bool(w.keyboard.pressed)
        post(cg, KEYCODES[name], False)
        pump(0.4)
        after = w.arm.targets[servo]
        pump(0.5)
        results[name] = (held, after != before, w.arm.targets[servo] == after)
        print(f"{name:>6}: held-detected={held}  moved={after != before}  stopped-after-release={w.arm.targets[servo] == after}")
    print("key tester saw:", seen[:6])
    w.emergency_stop("done")
    w.close_ready = True
    w.close()
    ok = all(h and m and s for h, m, s in results.values())
    print("RESULT:", "PASS" if ok else "FAIL (or the OS blocked synthetic key events: check Accessibility permission)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
