"""Tema visual moderno e opaco usando apenas recursos nativos do PyQt5."""

import os
import sys

from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication

APP_STYLESHEET = r"""
QMainWindow {
    background: #07101d;
}

QWidget#appRoot {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #061522,
        stop: 0.38 #0b2b3d,
        stop: 0.72 #172b48,
        stop: 1 #281b43
    );
}

QWidget {
    color: #e8f1ff;
    selection-background-color: #28b8d8;
    selection-color: #06111d;
}

QScrollArea#panelScroll, QScrollArea#manualScroll {
    background: transparent;
    border: none;
}

QScrollArea#panelScroll > QWidget > QWidget {
    background: transparent;
}

QWidget#manualContent {
    background-color: #091624;
}

QSplitter#mainSplitter {
    background: transparent;
}

QSplitter#mainSplitter::handle:horizontal {
    background-color: #10283a;
    border: 1px solid #24465e;
    border-radius: 3px;
    margin: 22px 1px;
}

QSplitter#mainSplitter::handle:horizontal:hover {
    background-color: #1a536b;
    border-color: #3a829d;
}

QGroupBox#sideCard, QGroupBox#viewerCard {
    background-color: #0d1b2a;
    border: 1px solid #243b53;
    border-radius: 16px;
    margin-top: 18px;
    padding: 14px 10px 10px 10px;
    font-size: 10pt;
    font-weight: 600;
}

QGroupBox#viewerCard {
    background-color: #050a12;
    border-color: #1d4860;
}

QGroupBox#dynamicsPanel {
    background-color: #0a2232;
    border: 1px solid #28627c;
    border-radius: 11px;
    margin-top: 13px;
    padding: 8px 6px 5px 6px;
}

QGroupBox#dynamicsPanel::title {
    color: #bcefff;
    background-color: #12384c;
    border-color: #32728b;
    left: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    padding: 2px 8px;
    color: #ccecff;
    background-color: #10263a;
    border: 1px solid #28465e;
    border-radius: 7px;
}

QLabel {
    color: #dbe9f7;
    background: transparent;
}

QLabel#orbitalTitle {
    color: #f4fbff;
    font-size: 20pt;
    font-weight: 700;
    padding: 4px;
}

QLabel#helpBanner {
    color: #caecff;
    background-color: rgba(38, 128, 167, 48);
    border: 1px solid rgba(89, 198, 235, 72);
    border-radius: 10px;
    padding: 8px;
}

QLabel#sliceStatus {
    color: #c9eaff;
    background-color: #10283a;
    border: 1px solid #28526d;
    border-radius: 9px;
    padding: 8px 12px;
}

QLabel#phaseLegend {
    color: #d9eaf7;
    background-color: #0d2030;
    border: 1px solid #284a61;
    border-radius: 8px;
    padding: 7px 10px;
    font-size: 9pt;
}

QLabel#dynamicsStatus {
    color: #d7f4ff;
    background-color: rgba(22, 87, 111, 70);
    border-left: 3px solid #51d7ef;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 8.5pt;
}

QLabel#transitionResult {
    color: #dceeff;
    background-color: #10263a;
    border: 1px solid #31556f;
    border-radius: 9px;
    padding: 9px 12px;
}

QLabel#radialSummary {
    color: #dceeff;
    background-color: #10263a;
    border: 1px solid #31556f;
    border-radius: 9px;
    padding: 8px 11px;
}

QLabel#quantumStateCard {
    color: #dff3ff;
    background-color: #0d2638;
    border-left: 4px solid #49cce8;
    border-radius: 8px;
    padding: 8px 11px;
    font-family: "Cascadia Mono";
    font-size: 9pt;
}

QLabel#emptySlice {
    color: #7891a8;
    background-color: #07111f;
    border: 1px dashed #29445b;
    border-radius: 10px;
    padding: 24px;
    font-size: 11pt;
}

QWidget#sliceWorkspace {
    background-color: #07111f;
}

QLabel#manualStatus {
    border-radius: 10px;
    padding: 8px;
    border: 1px solid rgba(150, 205, 240, 40);
}

QLabel#manualStatus[state="ground"] {
    background-color: rgba(30, 139, 111, 55);
    border-color: rgba(71, 220, 164, 100);
    color: #c9ffe9;
}

QLabel#manualStatus[state="excited"] {
    background-color: rgba(185, 132, 35, 55);
    border-color: rgba(255, 198, 83, 105);
    color: #fff0c2;
}

QLabel#manualStatus[state="building"] {
    background-color: rgba(45, 115, 181, 48);
    border-color: rgba(102, 185, 255, 90);
    color: #d8efff;
}

QLabel#manualStatus[state="inactive"] {
    background-color: rgba(255, 255, 255, 18);
    color: #9fb3c8;
}

QWidget#manualContent QPushButton {
    min-height: 24px;
    padding: 3px 8px;
}

QTextEdit, QTextBrowser, QTableWidget {
    background-color: #07111f;
    border: 1px solid #263e54;
    border-radius: 10px;
    padding: 6px;
    color: #dcecff;
    gridline-color: rgba(154, 207, 245, 28);
}

QTableWidget::item {
    border-bottom: 1px solid rgba(150, 205, 240, 22);
    padding: 5px;
}

QTableWidget::item:selected {
    background-color: rgba(42, 177, 216, 105);
    color: #ffffff;
}

QHeaderView::section {
    background-color: #19344e;
    color: #ccecff;
    border: none;
    border-right: 1px solid rgba(170, 220, 255, 35);
    padding: 7px;
    font-weight: 600;
}

QTabWidget::pane {
    background-color: #091624;
    border: 1px solid #263e54;
    border-radius: 11px;
    top: -1px;
}

QTabBar::tab {
    background-color: #102235;
    color: #9fb5ca;
    border: 1px solid transparent;
    border-bottom: none;
    padding: 8px 12px;
    margin-right: 3px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}

QTabWidget#infoTabs QTabBar::tab {
    padding: 8px 7px;
    margin-right: 2px;
}

QTabWidget#viewerTabs QTabBar::tab {
    padding: 8px 6px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    color: #effaff;
    background-color: #18546b;
    border-color: #347e99;
}

QTabBar::tab:hover:!selected {
    color: #d7efff;
    background-color: rgba(255, 255, 255, 24);
}

QTabBar::tab:disabled {
    color: #53677b;
    background-color: rgba(255, 255, 255, 7);
}

QPushButton {
    min-height: 30px;
    background-color: rgba(215, 239, 252, 23);
    color: #e5f3ff;
    border: 1px solid rgba(180, 220, 250, 45);
    border-radius: 9px;
    padding: 5px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: rgba(255, 255, 255, 32);
    border-color: rgba(122, 218, 247, 100);
}

QPushButton:pressed {
    background-color: rgba(34, 135, 172, 85);
    padding-top: 7px;
}

QPushButton:disabled {
    color: #66798c;
    background-color: rgba(255, 255, 255, 8);
    border-color: rgba(255, 255, 255, 14);
}

QPushButton[variant="primary"] {
    color: #04131c;
    background-color: #55d7f2;
    border-color: #83e8fa;
}

QPushButton[variant="primary"]:hover {
    background-color: #75e4f8;
}

QPushButton[variant="accent"] {
    background-color: rgba(105, 91, 230, 175);
    border-color: rgba(167, 154, 255, 170);
}

QPushButton[variant="success"] {
    background-color: rgba(31, 159, 121, 165);
    border-color: rgba(90, 226, 179, 150);
}

QPushButton[variant="warning"] {
    background-color: rgba(207, 140, 34, 160);
    border-color: rgba(255, 197, 91, 155);
}

QPushButton[variant="danger"] {
    background-color: rgba(190, 67, 83, 145);
    border-color: rgba(255, 123, 139, 145);
}

QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 28px;
    background-color: #071525;
    color: #e2f0fb;
    border: 1px solid rgba(153, 207, 245, 55);
    border-radius: 8px;
    padding: 3px 28px 3px 9px;
}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: rgba(91, 211, 244, 125);
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    color: #e2f0fb;
    background-color: #0c1c2e;
    border: 1px solid #27445f;
    selection-background-color: #246a86;
    outline: 0;
}

QSlider::groove:horizontal {
    height: 5px;
    background-color: rgba(255, 255, 255, 28);
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background-color: #43c8e5;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -6px 0;
    background-color: #e9fbff;
    border: 3px solid #35b9d7;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #ffffff;
    border-color: #71e3f7;
}

QCheckBox {
    spacing: 8px;
    color: #d6e7f6;
}

QCheckBox::indicator {
    width: 17px;
    height: 17px;
    border: 1px solid rgba(154, 207, 245, 80);
    border-radius: 5px;
    background-color: rgba(4, 13, 25, 190);
}

QCheckBox::indicator:checked {
    background-color: #42c8e5;
    border-color: #7de7fa;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background-color: rgba(142, 202, 235, 75);
    min-height: 28px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background-color: rgba(91, 211, 244, 125);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
    border: none;
}

QToolTip {
    color: #effaff;
    background-color: #10253a;
    border: 1px solid #3c6b88;
    border-radius: 6px;
    padding: 5px;
}
"""


def _enable_windows_dark_title_bar(window) -> None:
    """Escurece somente a barra nativa para combinar com o tema opaco."""
    if sys.platform != "win32" or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return
    try:
        import ctypes

        hwnd = int(window.winId())
        dwm_set_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        enabled = ctypes.c_int(1)
        for attribute in (20, 19):
            result = dwm_set_attribute(
                hwnd,
                attribute,
                ctypes.byref(enabled),
                ctypes.sizeof(enabled),
            )
            if result == 0:
                break
    except (AttributeError, OSError, ValueError):
        pass


def apply_app_theme(window) -> None:
    """Aplica o tema opaco sem bibliotecas externas."""
    app = QApplication.instance()
    if app is not None:
        app.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#07101d"))
        palette.setColor(QPalette.WindowText, QColor("#e8f1ff"))
        palette.setColor(QPalette.Base, QColor("#07111f"))
        palette.setColor(QPalette.AlternateBase, QColor("#0c1c2e"))
        palette.setColor(QPalette.Text, QColor("#e8f1ff"))
        palette.setColor(QPalette.Button, QColor("#13263b"))
        palette.setColor(QPalette.ButtonText, QColor("#e8f1ff"))
        palette.setColor(QPalette.Highlight, QColor("#28b8d8"))
        palette.setColor(QPalette.HighlightedText, QColor("#06111d"))
        palette.setColor(
            QPalette.Disabled, QPalette.Text, QColor("#66798c")
        )
        palette.setColor(
            QPalette.Disabled, QPalette.ButtonText, QColor("#66798c")
        )
        app.setPalette(palette)
    window.setFont(QFont("Segoe UI", 10))
    window.setStyleSheet(APP_STYLESHEET)
    # A janela permanece totalmente opaca. Os contrastes do tema são feitos
    # por cores e bordas internas, preservando texto e visualização 3D.
    window.setWindowOpacity(1.0)
    window.window_opacity = 1.0
    _enable_windows_dark_title_bar(window)
