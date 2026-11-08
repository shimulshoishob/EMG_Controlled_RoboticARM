"""Application entry point."""

from __future__ import annotations

import signal
import sys
import traceback

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication

from .arm.driver import load_config
from .settings import Settings
from .ui import theme
from .ui.main_window import KeyboardFilter, MainWindow, console


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    config_path = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None
    config = load_config(config_path)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(argv)
    app.setApplicationName("EMG Robotic Arm")
    app.setOrganizationName("BioWave")
    settings = Settings()
    theme.apply_theme(app, settings.get("theme"))
    window = MainWindow(config, settings)

    # Ctrl+C, `kill` or closing the terminal go through closeEvent too, so the arm is parked first.
    # (Python only runs signal handlers while it executes Python code, hence the idle timer.)
    def request_close(*_):
        QTimer.singleShot(0, window.close)

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_close)
    idle = QTimer()
    idle.timeout.connect(lambda: None)
    idle.start(200)

    # An unexpected error must not kill the app (PyQt would abort, skipping the parking):
    # stop motion, report it, and keep the window open.
    def on_error(etype, value, tb):
        console("".join(traceback.format_exception(etype, value, tb)))
        try:
            window.log("error", note=f"{etype.__name__}: {value}")
            window.emergency_stop(f"Internal error ({etype.__name__}: {value})")
        except Exception:
            pass

    sys.excepthook = on_error
    app.installEventFilter(KeyboardFilter(window))
    app.applicationStateChanged.connect(lambda state: state != Qt.ApplicationActive and window.pressed_keys.clear())
    window.show()
    theme.apply_title_bar(window)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
