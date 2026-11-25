"""Small line icons for the sidebar, drawn from inline SVG so they take the colour of the current theme."""

from __future__ import annotations

from PyQt5.QtCore import QByteArray, QRectF, Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer

from . import theme

SIZE = 20

# 24 x 24 drawings, stroke only
_PATHS = {
    "control": '<path d="M4 21h9M8 21v-5l-3-6 6-5 6 3"/><circle cx="5" cy="10" r="1.6"/><circle cx="11" cy="5" r="1.6"/>'
               '<path d="M17 8l4-1M17 8l2 4"/>',
    "model": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="10" y="10" width="4" height="4"/>'
             '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "armband": '<path d="M4.5 9a10.5 10.5 0 0 1 15 0M7.8 12.600a6 6 0 0 1 8.4 0"/><circle cx="12" cy="17.5" r="1.6"/>',
    "monitor": '<path d="M2 12h4l3-8 5 16 3-8h5"/>',
    "log": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l3 2M9 2h6M12 2v3M19 6l1.500-1.500"/>',
    "mapping": '<path d="M3 7h13M13 3.500L16.500 7 13 10.500M21 17H8M11 13.500L7.500 17l3.500 3.500"/>',
    "data": '<path d="M4 7h9M18 7h2M4 17h2M11 17h9"/><circle cx="15.500" cy="7" r="2.500"/><circle cx="8.500" cy="17" r="2.500"/>',
}


def _pixmap(name: str, color: str, ratio: int = 2) -> QPixmap:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_PATHS[name]}</svg>')
    pix = QPixmap(SIZE * ratio, SIZE * ratio)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter, QRectF(pix.rect()))
    painter.end()
    pix.setDevicePixelRatio(ratio)
    return pix


def nav_icon(name: str) -> QIcon:
    """Muted when the page is closed, accent colour when it is the open one."""
    t = theme.tokens()
    icon = QIcon()
    icon.addPixmap(_pixmap(name, t.muted), QIcon.Normal, QIcon.Off)
    icon.addPixmap(_pixmap(name, t.accent), QIcon.Normal, QIcon.On)
    return icon
