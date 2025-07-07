"""PyQt overlay (click‑through, frameless) for highlighting a bounding box.

Usage:
    from box import show_box
    show_box({"y_min": 100, "x_min": 100, "y_max": 300, "x_max": 400})

Key fix (v1.1):
• Replaced Qt.WindowTransparentForInput (Qt6‑only on macOS) with
  `setAttribute(Qt.WA_TransparentForMouseEvents)` so the window becomes
  click‑through *and* visible in PyQt5.
• Added semi‑transparent fill so you can actually see it.
"""
import sys
from typing import Dict
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import QApplication, QWidget


class Overlay(QWidget):
    def __init__(self, bbox: Dict[str, int], timeout_ms: int = 5000):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # Make window background transparent and pass mouse events through
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Normalise bbox → set geometry
        x_min, x_max = sorted((bbox["x_min"], bbox["x_max"]))
        y_min, y_max = sorted((bbox["y_min"], bbox["y_max"]))
        self.setGeometry(x_min, y_min, x_max - x_min, y_max - y_min)

        # Style config
        self._border_color = QColor("#007AFF")        # iOS blue
        self._border_width = 4
        self._border_radius = 14
        self._fill_color = QColor(0, 122, 255, 60)     # translucent fill

        if timeout_ms > 0:
            QTimer.singleShot(timeout_ms, self.close)

    # Painting happens each time Qt needs to redraw
    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill
        painter.setBrush(QBrush(self._fill_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), self._border_radius, self._border_radius)

        # Border
        pen = QPen(self._border_color, self._border_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        # inset so border is fully visible
        inset = self._border_width / 2
        border_rect = QRectF(inset, inset, self.width() - self._border_width, self.height() - self._border_width)
        painter.drawRoundedRect(border_rect, self._border_radius, self._border_radius)


# Public helper
def show_box(bbox: Dict[str, int], timeout_ms: int = 5000):
    app = QApplication.instance() or QApplication(sys.argv)
    ov = Overlay(bbox, timeout_ms)
    ov.show()
    if not QApplication.instance().closingDown():
        app.exec_()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5:
        y_min, x_min, y_max, x_max = map(int, sys.argv[1:5])
        show_box({"y_min": y_min, "x_min": x_min,
                  "y_max": y_max, "x_max": x_max}, timeout_ms=0)
    else:
        print("Usage: python box.py y_min x_min y_max x_max")
