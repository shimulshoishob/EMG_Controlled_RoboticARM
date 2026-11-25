"""The app logo: window title bar and taskbar on Windows, Dock on macOS."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtGui import QIcon

ASSETS = Path(__file__).parent / "assets"
APP_ID = "BioWave.EMGRoboticArm"


def icon_path() -> Path:
    """Each system gets the file made for it (Windows: edge to edge; macOS: with the Dock's margin)."""
    return ASSETS / {"win32": "app.ico", "darwin": "app.icns"}.get(sys.platform, "logo.png")


def app_icon() -> QIcon:
    return QIcon(str(icon_path()))


def set_windows_fallback_icon(window):
    """The taskbar asks the window for its icon. While the app is busy (loading the gesture model at start-up) it
    cannot answer, and Windows then keeps the icon of the window *class*: a blank one taken from Python. Give the
    class our logo too. Call before the window is shown. No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32")                 # private copy: these signatures affect nobody else
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int,
                                      ctypes.c_int, wintypes.UINT]
        set_class = getattr(user32, "SetClassLongPtrW", None) or user32.SetClassLongW     # 32-bit Windows
        set_class.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        set_class.restype = ctypes.c_void_p
        hwnd = int(window.winId())
        image_icon, from_file = 1, 0x10
        for slot, metric in ((-14, 11), (-34, 49)):      # (large icon slot, its size), (small icon slot, its size)
            size = user32.GetSystemMetrics(metric)
            icon = user32.LoadImageW(None, str(ASSETS / "app.ico"), image_icon, size, size, from_file)
            if icon:                                     # kept for the life of the app
                set_class(hwnd, slot, icon)
    except Exception:
        pass


def set_windows_app_id():
    """Without its own id Windows groups the app under Python and shows Python's icon in the taskbar.
    Call before the first window is created. No-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass
