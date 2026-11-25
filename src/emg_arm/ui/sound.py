"""The "function switched" beep (Up/Down joint: shoulder <-> elbow).

It plays on this computer. The Hiwonder xArm 1S has a buzzer on its controller board, but the USB protocol and the
``xarm`` library offer no documented command to sound it, and sending guessed commands to the arm is not safe.
"""

from __future__ import annotations

import sys
import threading

from PyQt5.QtWidgets import QApplication

FREQUENCY_HZ = 1500
DURATION_MS = 180


def beep():
    """Returns at once; never raises (a missing sound device must not affect control of the arm)."""
    if sys.platform == "win32":
        def work():
            try:
                import winsound
                winsound.Beep(FREQUENCY_HZ, DURATION_MS)
            except Exception:
                pass

        threading.Thread(target=work, name="beep", daemon=True).start()
    elif QApplication.instance() is not None:
        QApplication.beep()
