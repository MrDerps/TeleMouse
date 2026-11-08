"""Axion design tokens (dark) and the app-wide Qt stylesheet. Spec: docs/handoff/FROM_BACKEND.md, 'Axion' brief."""
from __future__ import annotations

from PyQt6 import QtGui

C = {
    "bg": "#09090B",          # page, top bar, status bar
    "panel": "#121214",
    "border": "#232327",
    "divider": "#1E1E22",
    "text": "#EDEDEF",
    "text2": "#A1A1AA",
    "label": "#8B8B94",
    "disabled": "#5C5C66",
    "accent": "#5B9BFF",
    "on_accent": "#0B0B0C",
    "link": "#8AB8FF",
    "tint": "#0E1A2E",
    "emg": "#7CC4FA",
    "imu": "#9B9CF5",
    "cam": "#6FD6C3",
    "bad": "#F87171",
    "warn": "#FBBF24",
    "ok": "#4ADE80",
    "button": "#18181B",
    "button_hover": "#202024",
}
SENSOR_COLOR = {"emg": C["emg"], "imu": C["imu"], "hand": C["cam"]}
FAMILIES = ["Inter", "SF Pro Text", "Segoe UI Variable Text", "Segoe UI"]


def font(px=14, weight=400, tracking=0.0, tabular=True) -> QtGui.QFont:
    f = QtGui.QFont()
    f.setFamilies(FAMILIES)
    f.setPixelSize(px)
    f.setWeight(QtGui.QFont.Weight(weight))
    if tracking:
        f.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, 100 + 100 * tracking)
    if tabular:
        try:
            f.setFeature(QtGui.QFont.Tag("tnum"), 1)
        except Exception:
            pass
    return f


QSS = f"""
QWidget {{ background: {C['bg']}; color: {C['text']}; }}
QToolTip {{ background: {C['panel']}; color: {C['text']}; border: 1px solid {C['border']}; padding: 6px; }}
QFrame#Panel {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 10px; }}
QFrame#Panel QWidget {{ background: transparent; }}
QFrame#Card {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 10px; }}
QFrame#Card:hover {{ border: 1px solid #34343A; }}
QFrame#Card QWidget {{ background: transparent; }}
QFrame#AddCard {{ background: {C['tint']}; border: 1px dashed {C['accent']}; border-radius: 10px; }}
QFrame#AddCard QWidget {{ background: transparent; }}
QFrame#Divider {{ background: {C['divider']}; max-height: 1px; min-height: 1px; border: 0; }}
QFrame#Panel QPushButton, QFrame#Card QPushButton, QFrame#AddCard QPushButton {{
    background: {C['button']}; border: 1px solid {C['border']}; border-radius: 8px; padding: 7px 14px; color: {C['text']}; }}
QFrame#Panel QPushButton:hover, QFrame#Card QPushButton:hover {{ background: {C['button_hover']}; }}
QFrame#Panel QPushButton#Primary, QFrame#Card QPushButton#Primary, QFrame#AddCard QPushButton#Primary {{
    background: {C['accent']}; color: {C['on_accent']}; border: 0; font-weight: 600; }}
QFrame#Panel QPushButton#Primary:hover, QFrame#Card QPushButton#Primary:hover {{ background: #74ABFF; }}
QFrame#Panel QPushButton#Primary:disabled, QFrame#Card QPushButton#Primary:disabled {{ background: #2A3A55; color: #6B7A93; }}
QFrame#Panel QPushButton:disabled, QFrame#Card QPushButton:disabled {{ color: {C['disabled']}; }}
QFrame#Panel QPushButton#Danger, QFrame#Card QPushButton#Danger {{ color: {C['bad']}; }}
QFrame#Panel QPushButton#Ghost, QFrame#Card QPushButton#Ghost {{ background: transparent; border: 0; color: {C['link']}; padding: 4px 6px; }}
QFrame#Panel QComboBox, QFrame#Panel QLineEdit, QFrame#Panel QSpinBox, QFrame#Panel QDoubleSpinBox {{
    background: #0E0E10; border: 1px solid {C['border']}; border-radius: 8px; padding: 6px 8px; }}
QFrame#Panel QTableWidget, QFrame#Panel QListWidget {{ background: {C['panel']}; border: 0; }}
QWidget#TopBar, QWidget#StatusBar {{ background: {C['bg']}; }}
QWidget#TopBar {{ border-bottom: 1px solid {C['divider']}; }}
QWidget#StatusBar {{ border-top: 1px solid {C['divider']}; }}
QLabel {{ background: transparent; }}
QLabel#Muted {{ color: {C['text2']}; }}
QLabel#Label {{ color: {C['label']}; }}
QLabel#Link {{ color: {C['link']}; }}
QPushButton {{ background: {C['button']}; color: {C['text']}; border: 1px solid {C['border']};
              border-radius: 8px; padding: 7px 14px; }}
QPushButton:hover {{ background: {C['button_hover']}; }}
QPushButton:disabled {{ color: {C['disabled']}; }}
QPushButton#Primary {{ background: {C['accent']}; color: {C['on_accent']}; border: 0; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #74ABFF; }}
QPushButton#Primary:disabled {{ background: #2A3A55; color: #6B7A93; }}
QPushButton#Danger {{ color: {C['bad']}; }}
QPushButton#Ghost {{ background: transparent; border: 0; color: {C['link']}; padding: 4px 6px; }}
QPushButton#Ghost:hover {{ color: #B5D1FF; }}
QPushButton#Tab {{ background: transparent; border: 0; border-radius: 0; color: {C['text2']};
                  padding: 15px 4px 13px 4px; margin: 0 10px; border-bottom: 2px solid transparent; }}
QPushButton#Tab:hover {{ color: {C['text']}; }}
QPushButton#Tab:checked {{ color: {C['text']}; border-bottom: 2px solid {C['accent']}; }}
QPushButton#Seg {{ background: transparent; border: 0; border-radius: 6px; color: {C['text2']}; padding: 5px 12px; }}
QPushButton#Seg:checked {{ background: #26262B; color: {C['text']}; }}
QWidget#SegBox {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 8px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: #0E0E10; border: 1px solid {C['border']}; border-radius: 8px; padding: 6px 8px; color: {C['text']}; }}
QComboBox QAbstractItemView {{ background: {C['panel']}; border: 1px solid {C['border']}; selection-background-color: {C['tint']}; }}
QTableWidget, QListWidget {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 10px;
    gridline-color: {C['divider']}; alternate-background-color: #141417; }}
QTableWidget::item, QListWidget::item {{ padding: 6px; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: {C['tint']}; color: {C['text']}; }}
QHeaderView::section {{ background: {C['panel']}; color: {C['label']}; border: 0;
    border-bottom: 1px solid {C['divider']}; padding: 8px; font-size: 11px; }}
QTableCornerButton::section {{ background: {C['panel']}; border: 0; }}
QProgressBar {{ background: #0E0E10; border: 1px solid {C['border']}; border-radius: 6px; height: 10px;
               text-align: center; color: {C['text2']}; font-size: 11px; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 4px; background: #2A2A30; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C['text']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #3A3A40; background: #0E0E10; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border: 1px solid {C['accent']}; }}
QScrollArea {{ border: 0; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #2A2A30; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QDialog {{ background: {C['bg']}; }}
QMessageBox {{ background: {C['panel']}; }}
QGroupBox {{ border: 1px solid {C['border']}; border-radius: 10px; margin-top: 14px; padding: 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {C['label']}; }}
"""
