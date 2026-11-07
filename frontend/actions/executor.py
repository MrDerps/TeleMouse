"""Turns backend `gesture` events into real mouse/keyboard input (pynput).

Runs in the backend-link thread the moment an event arrives (no UI-timer delay).
Safety:
  * nothing happens unless actions are enabled (off by default) and the kill switch is not set
  * an event below the confidence threshold is ignored (hold `end` events always release)
  * release_all() lets go of any held button / stops continuous scroll; called by the F9 kill switch,
    when actions are disabled, when the backend goes offline, and on exit
  * a global F9 listener (any window) triggers the kill switch
"""
from __future__ import annotations

import threading
import time

from . import describe

KEY_NAMES = {"ctrl": "ctrl", "control": "ctrl", "shift": "shift", "alt": "alt", "win": "cmd", "cmd": "cmd",
             "enter": "enter", "esc": "esc", "escape": "esc", "tab": "tab", "space": "space",
             "backspace": "backspace", "delete": "delete", "up": "up", "down": "down", "left": "left",
             "right": "right", "home": "home", "end": "end", "pageup": "page_up", "pagedown": "page_down"}


class Executor:
    def __init__(self, state, log=print):
        """state: object with .actions_enabled, .killed, .conf_threshold, .gestures (list of dicts)"""
        from pynput import keyboard, mouse
        self.state, self.log = state, log
        self._kb_mod, self._mouse_mod = keyboard, mouse
        self.mouse = mouse.Controller()
        self.kb = keyboard.Controller()
        self.lock = threading.Lock()
        self.held = {}                 # gesture id -> ("drag", button) | ("scroll", stop_event)
        self.pointer_on = True         # a mode_toggle gesture turns the pointer off/on
        self.listener = None

    # ------------------------------------------------------------------ kill switch
    def start_hotkey(self, on_kill):
        kb = self._kb_mod

        def on_press(key):
            if key == kb.Key.f9:
                on_kill()
        self.listener = kb.Listener(on_press=on_press)
        self.listener.daemon = True
        self.listener.start()

    def stop(self):
        self.release_all()
        if self.listener:
            self.listener.stop()

    def release_all(self):
        with self.lock:
            held, self.held = self.held, {}
        for gid, (kind, x) in held.items():
            try:
                if kind == "drag":
                    self.mouse.release(x)
                elif kind == "scroll":
                    x.set()
            except Exception as e:
                self.log(f"release {gid}: {e}")

    # ------------------------------------------------------------------ events
    def handle(self, e: dict) -> str:
        st = self.state
        gid, kind, conf = e.get("id"), e.get("kind"), float(e.get("conf", 0.0))
        g = next((x for x in st.gestures if x["id"] == gid), None)
        a = (g or {}).get("action") or {"type": "none"}
        if kind == "end":                               # always allowed: it only ever releases
            return self._end(gid) or "end"
        if st.killed:
            return "ignored (kill switch)"
        if not st.actions_enabled:
            return f"ignored (actions off): {describe(a)}"
        if conf < st.conf_threshold:
            return f"ignored (conf {conf:.2f} < {st.conf_threshold:.2f})"
        try:
            return self._do(gid, kind, a)
        except Exception as ex:
            return f"error: {ex}"

    def _button(self, name):
        B = self._mouse_mod.Button
        return {"left": B.left, "right": B.right, "middle": B.middle}.get(name or "left", B.left)

    def _do(self, gid, kind, a):
        t = a.get("type", "none")
        if t == "click":
            self.mouse.click(self._button(a.get("button")), 1)
        elif t == "double_click":
            self.mouse.click(self._button(a.get("button")), 2)
        elif t == "scroll":
            dx, dy = int(a.get("dx", 0)), int(a.get("dy", 0))
            if kind == "start" and a.get("continuous"):
                stop = threading.Event()
                with self.lock:
                    self.held[gid] = ("scroll", stop)
                threading.Thread(target=self._scroll_loop, args=(dx, dy, stop), daemon=True).start()
                return f"scrolling {dx},{dy} while held"
            self.mouse.scroll(dx, dy)
        elif t == "drag":
            if kind != "start":
                return "drag needs a hold gesture (ignored)"
            b = self._button(a.get("button"))
            self.mouse.press(b)
            with self.lock:
                self.held[gid] = ("drag", b)
            return f"drag: {a.get('button', 'left')} pressed"
        elif t == "key":
            self._combo(a.get("combo", ""))
        elif t == "mode_toggle":
            self.pointer_on = not self.pointer_on
            return f"pointer {'ON' if self.pointer_on else 'OFF'}"
        else:
            return "no action assigned"
        return "done: " + describe(a)

    def _end(self, gid):
        with self.lock:
            h = self.held.pop(gid, None)
        if not h:
            return None
        kind, x = h
        if kind == "drag":
            self.mouse.release(x)
            return "drag released"
        x.set()
        return "scroll stopped"

    def _scroll_loop(self, dx, dy, stop):
        t_end = time.perf_counter() + 30.0              # safety cap: a lost `end` can't scroll forever
        while not stop.is_set() and time.perf_counter() < t_end:
            if self.state.killed or not self.state.actions_enabled:
                break
            self.mouse.scroll(dx, dy)
            stop.wait(0.08)

    def _combo(self, combo):
        kb = self._kb_mod
        keys = []
        for part in [p.strip().lower() for p in combo.split("+") if p.strip()]:
            if part in KEY_NAMES:
                keys.append(getattr(kb.Key, KEY_NAMES[part]))
            elif len(part) >= 2 and part[0] == "f" and part[1:].isdigit():
                keys.append(getattr(kb.Key, part))
            elif len(part) == 1:
                keys.append(part)
            else:
                raise ValueError(f"unknown key {part!r}")
        for k in keys:
            self.kb.press(k)
        for k in reversed(keys):
            self.kb.release(k)
