"""Look and feel: one design for macOS and Windows, light and dark.

* Qt's Fusion style everywhere, so controls look identical on both systems.
* The platform's own UI font (SF Pro on macOS, Segoe UI Variable on Windows 11), sized in points so
  it scales with the display (Retina / 125 % / 150 % Windows scaling).
* Colour tokens for a clean, slightly futuristic console look: thin outlines, soft gradients and one
  bright accent per theme; ``Auto`` follows the operating-system setting.
* Widgets never carry colours of their own: they set a ``role`` / ``tone`` property and this
  stylesheet does the rest, so switching theme repaints everything.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt5.QtWidgets import QApplication, QWidget

CHECK_ICON = (Path(__file__).parent / "assets" / "check.svg").as_posix()
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"
MOD = "⌘" if IS_MAC else "Ctrl+"


@dataclass(frozen=True)
class Tokens:
    name: str
    bg: str
    surface: str
    surface2: str
    border: str
    text: str
    muted: str
    accent: str
    accent_text: str
    accent_soft: str
    good: str
    good_soft: str
    warn: str
    warn_soft: str
    bad: str
    bad_soft: str
    plot_bg: str
    plot_grid: str
    accent2: str              # second accent, only used in gradients


LIGHT = Tokens("light", bg="#EEF2FA", surface="#FFFFFF", surface2="#E9EFFA", border="#CFDAEE",
               text="#101A33", muted="#55648A", accent="#3A53E0", accent_text="#FFFFFF",
               accent_soft="#E2E7FD", good="#067A5F", good_soft="#DAF4EC", warn="#9C5200",
               warn_soft="#FCEED6", bad="#C61F37", bad_soft="#FCE2E6", plot_bg="#FBFCFF", plot_grid="#DFE6F5",
               accent2="#0A7EA8")
DARK = Tokens("dark", bg="#070B14", surface="#0E1524", surface2="#151F33", border="#223252",
              text="#E4EEFF", muted="#8296BC", accent="#35D6F5", accent_text="#031019",
              accent_soft="#0E2C42", good="#37E3A8", good_soft="#0C2D27", warn="#FFC15C",
              warn_soft="#33270F", bad="#FF6B81", bad_soft="#38141F", plot_bg="#090F1B", plot_grid="#1B2943",
              accent2="#8B7CFF")

PINK = Tokens("pink", bg="#FDF0F6", surface="#FFFFFF", surface2="#FBE6F0", border="#F0C2D9",
              text="#34132A", muted="#8A4F70", accent="#C9246E", accent_text="#FFFFFF",
              accent_soft="#FADAE9", good="#067A5F", good_soft="#DAF4EC", warn="#9C5200",
              warn_soft="#FCEED6", bad="#C61F37", bad_soft="#FCE2E6", plot_bg="#FFFBFD", plot_grid="#F6D9E7",
              accent2="#7B3FD0")

# Series colours for the 8 EMG channels (readable on both backgrounds).
CHANNEL_COLORS = ["#2563EB", "#0E9F8E", "#D97706", "#DB2777", "#7C3AED", "#65A30D", "#0891B2", "#E11D48"]

_current = LIGHT
_mode = "auto"


def tokens() -> Tokens:
    return _current


def system_is_dark(app: QApplication) -> bool:
    return app.palette().color(QPalette.Window).lightness() < 128


def _base_font() -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    if IS_WIN:
        for family in ("Segoe UI Variable Text", "Segoe UI"):
            if family in QFontDatabase().families():
                font.setFamily(family)
                break
        font.setPointSizeF(10.0)
    elif IS_MAC:
        font.setPointSizeF(13.0)
    else:
        font.setPointSizeF(10.5)
    return font


def apply_theme(app: QApplication, mode: str = "auto") -> Tokens:
    """mode: 'auto' | 'light' | 'dark' | 'pink'."""
    global _current, _mode
    _mode = mode
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    if not hasattr(app, "_emg_base_dark"):
        app._emg_base_dark = system_is_dark(app)          # remember the OS choice before we repaint
    if mode == "pink":
        t = PINK
    else:
        dark = app._emg_base_dark if mode == "auto" else mode == "dark"
        t = DARK if dark else LIGHT
    _current = t
    app.setFont(_base_font())

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(t.bg))
    pal.setColor(QPalette.WindowText, QColor(t.text))
    pal.setColor(QPalette.Base, QColor(t.surface))
    pal.setColor(QPalette.AlternateBase, QColor(t.surface2))
    pal.setColor(QPalette.Text, QColor(t.text))
    pal.setColor(QPalette.Button, QColor(t.surface))
    pal.setColor(QPalette.ButtonText, QColor(t.text))
    pal.setColor(QPalette.Highlight, QColor(t.accent))
    pal.setColor(QPalette.HighlightedText, QColor(t.accent_text))
    pal.setColor(QPalette.ToolTipBase, QColor(t.surface2))
    pal.setColor(QPalette.ToolTipText, QColor(t.text))
    pal.setColor(QPalette.PlaceholderText, QColor(t.muted))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(t.muted))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(t.muted))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(t))
    for widget in app.topLevelWidgets():
        apply_title_bar(widget)
    return t


def current_mode() -> str:
    return _mode


def stylesheet(t: Tokens) -> str:
    base = _base_font().pointSizeF()

    def pt(scale: float) -> str:                       # Qt style sheets need absolute sizes, not "em"
        return f"{base * scale:.1f}pt"

    mono = '"Cascadia Mono", Menlo, Consolas, "Courier New", monospace'
    on_good = t.accent_text if t.name == "dark" else "#FFFFFF"

    def rgba(colour: str, alpha: float) -> str:
        c = QColor(colour)
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {int(alpha * 255)})"

    grad = f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {t.accent}, stop:1 {t.accent2})"
    return f"""
* {{ outline: 0; }}
QMainWindow, QDialog, QWidget#root {{ background: {t.bg}; }}
QWidget {{ color: {t.text}; }}
QToolTip {{ background: {t.surface2}; color: {t.text}; border: 1px solid {t.accent}; padding: 6px 8px; }}

/* ---- text roles ------------------------------------------------------------------ */
QLabel {{ background: transparent; }}
QLabel[role="brand"] {{ font-size: {pt(1.45)}; font-weight: 800; color: {t.accent}; }}
QLabel[role="h1"] {{ font-size: {pt(1.6)}; font-weight: 700; }}
QLabel[role="h2"] {{ font-size: {pt(1.12)}; font-weight: 700; }}
QLabel[role="display"] {{ font-size: {pt(1.7)}; font-weight: 800; }}
QLabel[role="lead"] {{ color: {t.muted}; }}
QLabel[role="muted"] {{ color: {t.muted}; }}
QLabel[role="caption"] {{ color: {t.muted}; font-size: {pt(0.9)}; }}
QLabel[role="section"] {{ color: {t.accent}; font-size: {pt(0.82)}; font-weight: 700; }}
QLabel[role="mono"] {{ font-family: {mono}; color: {t.muted}; }}
QLabel[role="stat"] {{ font-family: {mono}; font-size: {pt(1.55)}; font-weight: 700; }}
QLabel[tone="good"] {{ color: {t.good}; }}
QLabel[tone="warn"] {{ color: {t.warn}; }}
QLabel[tone="bad"] {{ color: {t.bad}; }}
QLabel[tone="accent"] {{ color: {t.accent}; }}

/* ---- panels ---------------------------------------------------------------------- */
QFrame[role="card"] {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 12px; }}
QFrame[role="card"][tone="good"] {{ border-color: {t.good}; }}
QFrame[role="card"][tone="warn"] {{ border-color: {t.warn}; }}
QFrame[role="card"][tone="bad"] {{ border-color: {t.bad}; }}
QFrame[role="tile"] {{ background: {t.surface2}; border: 1px solid {t.border}; border-radius: 10px; }}
QFrame[role="hero"] {{ background: {t.surface2}; border: 1px solid {t.border}; border-left: 3px solid {t.accent};
                      border-radius: 10px; }}
QFrame[role="topbar"] {{ background: {t.surface}; border: none; border-bottom: 1px solid {t.border}; }}
QFrame[role="sidebar"] {{ background: {t.surface}; border: none; border-right: 1px solid {t.border}; }}
QFrame[role="glow"] {{ border: none; min-height: 2px; max-height: 2px; background: {grad}; }}
QFrame[role="banner"] {{ border-radius: 9px; border: 1px solid {t.border}; background: {t.surface2}; }}
QFrame[role="banner"][tone="good"] {{ background: {t.good_soft}; border-color: {t.good}; }}
QFrame[role="banner"][tone="warn"] {{ background: {t.warn_soft}; border-color: {t.warn}; }}
QFrame[role="banner"][tone="bad"] {{ background: {t.bad_soft}; border-color: {t.bad}; }}
QFrame[role="banner"][tone="accent"] {{ background: {t.accent_soft}; border-color: {t.accent}; }}

/* ---- step badge (1, 2, 3 ...) and status dots --------------------------------------- */
QLabel[role="badge"] {{ background: {t.accent_soft}; color: {t.accent}; border: 1px solid {t.accent}; border-radius: 8px;
                        font-family: {mono}; font-weight: 700;
                        min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; }}
QLabel[role="badge"][tone="good"] {{ background: {t.good_soft}; color: {t.good}; border-color: {t.good}; }}
QLabel[role="badge"][tone="muted"] {{ background: {t.surface2}; color: {t.muted}; border-color: {t.border}; }}
QLabel[role="dot"] {{ border-radius: 4px; min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;
                      background: {t.border}; }}
QLabel[role="dot"][tone="good"] {{ background: {t.good}; }}
QLabel[role="dot"][tone="warn"] {{ background: {t.warn}; }}
QLabel[role="dot"][tone="bad"] {{ background: {t.bad}; }}
QLabel[role="dot"][tone="accent"] {{ background: {t.accent}; }}
QLabel[role="chip"] {{ background: {t.surface2}; border: 1px solid {t.border}; border-radius: 8px; padding: 6px 5px;
                       color: {t.muted}; font-weight: 600; font-size: {pt(0.84)}; }}
QLabel[role="chip"][tone="good"] {{ background: {t.good_soft}; border-color: {t.good}; color: {t.good}; }}
QLabel[role="chip"][tone="warn"] {{ background: {t.warn_soft}; border-color: {t.warn}; color: {t.warn}; }}
QLabel[role="chip"][tone="bad"] {{ background: {t.bad_soft}; border-color: {t.bad}; color: {t.bad}; }}
QLabel[role="chip"][tone="accent"] {{ background: {t.accent_soft}; border-color: {t.accent}; color: {t.accent}; }}

/* ---- buttons ------------------------------------------------------------------------- */
QPushButton {{ background: {t.surface2}; border: 1px solid {t.border}; border-radius: 8px; padding: 7px 14px;
               font-weight: 600; }}
QPushButton:hover:!disabled {{ border-color: {t.accent}; color: {t.accent}; }}
QPushButton:pressed:!disabled {{ background: {t.accent_soft}; }}
QPushButton:disabled {{ color: {t.muted}; background: transparent; border-color: {t.border}; }}
QPushButton:focus {{ border: 2px solid {t.accent}; padding: 6px 13px; }}
QPushButton[role="primary"] {{ background: {grad}; color: {t.accent_text}; border: 1px solid {t.accent}; }}
QPushButton[role="primary"]:hover:!disabled {{ border-color: {t.text}; color: {t.accent_text}; }}
QPushButton[role="primary"]:disabled {{ background: transparent; color: {t.muted}; border-color: {t.border}; }}
QPushButton[role="success"] {{ background: {t.good}; color: {on_good}; border-color: {t.good}; }}
QPushButton[role="success"]:hover:!disabled {{ border-color: {t.text}; color: {on_good}; background: {t.good}; }}
QPushButton[role="success"]:disabled {{ background: transparent; color: {t.muted}; border-color: {t.border}; }}
QPushButton[role="danger"] {{ background: {t.bad_soft}; color: {t.bad}; border-color: {t.bad}; }}
QPushButton[role="danger"]:hover:!disabled {{ background: #C0262D; color: #FFFFFF; border-color: #C0262D; }}
QPushButton[role="outline"] {{ background: {t.accent_soft}; color: {t.accent}; border: 1px solid {t.accent}; }}
QPushButton[role="outline"]:hover:!disabled {{ border-color: {t.text}; }}
QPushButton[role="ghost"] {{ background: transparent; border-color: transparent; color: {t.accent}; }}
QPushButton[role="ghost"]:hover:!disabled {{ background: {t.accent_soft}; border-color: transparent; }}
QPushButton[compact="true"] {{ padding: 6px 4px; }}
QPushButton[compact="true"]:focus {{ padding: 5px 3px; }}
QPushButton[big="true"] {{ font-size: {pt(1.12)}; font-weight: 800; }}
QPushButton[role="estop"] {{ background: #D12A3F; color: #FFFFFF; border: 2px solid #FF6B81; border-radius: 10px;
                             font-size: {pt(1.15)}; font-weight: 800; padding: 6px 16px; }}
QPushButton[role="estop"]:hover {{ background: #E5364C; color: #FFFFFF; border-color: {t.text}; }}
QPushButton[role="nav"] {{ text-align: left; background: transparent; border: 1px solid transparent; border-radius: 9px;
                           padding: 10px 10px; font-weight: 600; color: {t.muted}; }}
QPushButton[role="nav"]:hover {{ background: {t.surface2}; color: {t.text}; }}
QPushButton[role="nav"]:checked {{ background: {t.accent_soft}; border-color: {t.accent}; color: {t.accent}; }}
QPushButton[role="seg"] {{ border-radius: 0; margin: 0; padding: 7px 18px; background: {t.surface};
                           border: 1px solid {t.border}; color: {t.muted}; }}
QPushButton[role="seg"]:checked {{ background: {grad}; color: {t.accent_text}; border-color: {t.accent}; }}
QPushButton[role="seg"][pos="first"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
QPushButton[role="seg"][pos="last"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}

/* ---- inputs ---------------------------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {t.bg}; border: 1px solid {t.border}; border-radius: 8px; padding: 6px 10px;
    selection-background-color: {t.accent}; selection-color: {t.accent_text}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 2px solid {t.accent}; padding: 5px 9px; }}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ background: {t.surface2}; color: {t.muted}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{ background: {t.surface}; border: 1px solid {t.border}; selection-background-color: {t.accent_soft};
                               selection-color: {t.text}; padding: 4px; outline: 0; }}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 17px; height: 17px; border: 1px solid {t.muted}; border-radius: 5px; background: {t.bg}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; image: url({CHECK_ICON}); }}
QListWidget, QTableWidget, QTreeWidget {{ background: {t.bg}; border: 1px solid {t.border}; border-radius: 8px; padding: 4px;
                                         alternate-background-color: {t.surface2}; gridline-color: {t.border}; }}
QListWidget::item {{ padding: 6px 10px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {t.accent_soft}; color: {t.text}; border-left: 3px solid {t.accent}; }}
QHeaderView::section {{ background: {t.surface2}; color: {t.muted}; border: none; border-bottom: 1px solid {t.accent};
                        padding: 6px 8px; font-weight: 700; }}
QProgressBar {{ background: {t.surface2}; border: none; border-radius: 5px; text-align: center; min-height: 16px; max-height: 16px;
                font-family: {mono}; font-size: {pt(0.82)}; }}
QProgressBar::chunk {{ background: {rgba(t.accent, 0.38)}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 6px; background: {t.surface2}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {t.accent}; width: 16px; margin: -6px 0; border-radius: 8px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {t.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t.border}; border-radius: 4px; min-width: 30px; }}
QMessageBox {{ background: {t.surface}; }}
QMenu {{ background: {t.surface}; border: 1px solid {t.border}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t.accent_soft}; }}
"""


def set_role(widget: QWidget, role: str = None, tone: str = None):
    """Set the ``role`` / ``tone`` dynamic properties and repaint with the new style."""
    changed = False
    if role is not None and widget.property("role") != role:
        widget.setProperty("role", role)
        changed = True
    if tone is not None and widget.property("tone") != tone:
        widget.setProperty("tone", tone)
        changed = True
    if changed:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()


def apply_title_bar(window: QWidget):
    """Windows 10/11: dark/light title bar matching the theme. No-op elsewhere."""
    if not IS_WIN:
        return
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = int(window.winId())
        flag = ctypes.c_int(1 if _current.name == "dark" else 0)
        for attr in (20, 19):                       # DWMWA_USE_IMMERSIVE_DARK_MODE (new, old id)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(attr),
                                                          ctypes.byref(flag), ctypes.sizeof(flag)) == 0:
                break
    except Exception:
        pass
