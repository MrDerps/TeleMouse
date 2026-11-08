"""Gesture actions. The schema lives in data/gestures.json under each gesture's "action":

    {"type": "none"}
    {"type": "click", "button": "left"|"right"|"middle"}
    {"type": "double_click", "button": "left"}
    {"type": "scroll", "dx": 0, "dy": -5, "continuous": false}   continuous: repeat while a hold is active
    {"type": "drag", "button": "left"}                           hold: press on start, release on end
    {"type": "key", "combo": "ctrl+z"}
    {"type": "mode_toggle", "mode": "pointer"}                   turn the air-mouse pointer on/off
    {"type": "pointer_hold"}                                     hold gesture: the cursor follows the hand while held
"""
from __future__ import annotations

ACTION_TYPES = ["none", "click", "double_click", "scroll", "drag", "key", "pointer_hold", "mode_toggle"]
BUTTONS = ["left", "right", "middle"]


def describe(a: dict | None) -> str:
    a = a or {"type": "none"}
    t = a.get("type", "none")
    if t == "click":
        return f"{a.get('button', 'left')} click"
    if t == "double_click":
        return f"double {a.get('button', 'left')} click"
    if t == "scroll":
        return f"scroll dx={a.get('dx', 0)} dy={a.get('dy', 0)}" + (" (continuous)" if a.get("continuous") else "")
    if t == "drag":
        return f"drag ({a.get('button', 'left')})"
    if t == "key":
        return f"key {a.get('combo', '')}"
    if t == "pointer_hold":
        return "move mouse while held"
    if t == "mode_toggle":
        return f"toggle {a.get('mode', 'pointer')}"
    return "no action"
