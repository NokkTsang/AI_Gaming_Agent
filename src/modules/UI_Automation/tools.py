from __future__ import annotations

from typing import List, Optional, Tuple
import os

import pyautogui
from smolagents import tool

from .automator import UIAutomator


_AUTOMATOR = UIAutomator()

# Simple policy knobs
POLICY_MAX_SCROLL_CLICKS = int(os.getenv("UI_MAX_SCROLL_CLICKS", "10"))
POLICY_MAX_WAIT_SECONDS = float(os.getenv("UI_MAX_WAIT_SECONDS", "5"))
POLICY_MAX_TYPE_CHARS = int(os.getenv("UI_MAX_TYPE_CHARS", "512"))
POLICY_ALLOW_ALT_F4 = os.getenv("UI_ALLOW_ALT_F4", "1") != "0"


def _center_of_box(box: List[float]) -> Tuple[float, float]:
    if len(box) == 2:
        x1, y1 = box
        x2, y2 = x1, y1
    else:
        x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _screen_size() -> Tuple[int, int]:
    w, h = pyautogui.size()
    return int(w), int(h)


def _denorm(nx: float, ny: float) -> Tuple[float, float]:
    # Always use the latest screen size (handles multi-monitor/resize)
    w, h = _screen_size()
    return nx * w, ny * h


def _validate_box(box: List[float], name: str = "box") -> List[float]:
    if not isinstance(box, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of floats")
    if len(box) not in (2, 4):
        raise ValueError(f"{name} must have length 2 or 4, got {len(box)}")
    vals = [float(v) for v in box]
    for v in vals:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{name} values must be in [0,1], got {vals}")
    return vals


@tool
def get_screen_size() -> dict:
    """Return current screen size as a dict: {"width": int, "height": int}."""
    w, h = _screen_size()
    return {"width": int(w), "height": int(h)}


@tool
def click_box(box: List[float], button: str = "left") -> str:
    """Click the center of a normalized box on screen.

    Args:
        box: [x1, y1, x2, y2] or [x, y], values in [0,1] relative to screen.
        button: "left" or "right".
    Returns: a short status string.
    """
    box = _validate_box(list(box), name="box")
    cx, cy = _center_of_box(list(box))
    x, y = _denorm(cx, cy)
    _AUTOMATOR.click(x, y, button=button)
    return f"clicked {button} at ({int(x)}, {int(y)})"


@tool
def move_to_box(box: List[float], duration: float = 0.0) -> str:
    """Move the mouse to the center of a normalized box.

    Args:
        box: [x1,y1,x2,y2] or [x,y] in [0,1].
        duration: seconds for the move.
    """
    box = _validate_box(list(box), name="box")
    cx, cy = _center_of_box(list(box))
    x, y = _denorm(cx, cy)
    _AUTOMATOR.move_to(x, y, duration=duration)
    return f"moved to ({int(x)}, {int(y)})"


@tool
def drag_box(start_box: List[float], end_box: List[float], duration: float = 0.3) -> str:
    """Drag from the center of start_box to the center of end_box (normalized).

    Args:
        start_box: [x1,y1,x2,y2] or [x,y] in [0,1].
        end_box: [x1,y1,x2,y2] or [x,y] in [0,1].
        duration: drag duration in seconds.
    """
    start_box = _validate_box(list(start_box), name="start_box")
    end_box = _validate_box(list(end_box), name="end_box")
    sx, sy = _denorm(*_center_of_box(list(start_box)))
    ex, ey = _denorm(*_center_of_box(list(end_box)))
    _AUTOMATOR.move_to(sx, sy)
    _AUTOMATOR.drag_to(ex, ey, duration=duration)
    return f"dragged from ({int(sx)}, {int(sy)}) to ({int(ex)}, {int(ey)})"


@tool
def scroll(direction: str, clicks: int = 5, box: Optional[List[float]] = None) -> str:
    """Scroll the screen optionally at a given box center.

    Args:
        direction: 'up' or 'down'.
        clicks: magnitude of the scroll.
        box: optional [x1,y1,x2,y2] or [x,y] normalized; when provided, scroll at that point.
    """
    direction = (direction or "").strip().lower()
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    clicks = abs(int(clicks))
    clicks = min(clicks, POLICY_MAX_SCROLL_CLICKS)
    clicks = clicks if direction == "up" else -clicks
    if box is None:
        _AUTOMATOR.scroll(clicks)
        return f"scrolled {direction}"
    box = _validate_box(list(box), name="box")
    x, y = _denorm(*_center_of_box(list(box)))
    _AUTOMATOR.scroll(clicks, int(x), int(y))
    return f"scrolled {direction} at ({int(x)}, {int(y)})"


@tool
def hotkey(combo: str) -> str:
    """Press a hotkey combo separated by spaces, e.g. 'ctrl c' or 'alt tab'.

    Recognizes arrowleft/arrowright/arrowup/arrowdown and maps to left/right/up/down.
    """
    KEY_MAP = {
        "arrowleft": "left",
        "arrowright": "right",
        "arrowup": "up",
        "arrowdown": "down",
        "space": " ",
    }
    raw_keys = [k for k in combo.split() if k]
    keys = [KEY_MAP.get(k.lower(), k) for k in raw_keys]
    # Policy: limit number of keys and optionally guard Alt+F4
    if len(keys) > 3:
        raise ValueError("hotkey supports up to 3 keys")
    lower = [k.lower() for k in keys]
    if ("alt" in lower and "f4" in lower) and not POLICY_ALLOW_ALT_F4:
        raise PermissionError("Alt+F4 is not allowed by policy")
    _AUTOMATOR.hotkey(*keys)
    return f"hotkey: {'+'.join(keys)}"


@tool
def type_text(text: str) -> str:
    """Type plain text as keyboard input. Use '\\n' to represent Enter if needed."""
    # Enforce a maximum length for safety
    submit = False
    if text.endswith("\\n") or text.endswith("\n"):
        text = text.rstrip("\\n").rstrip("\n")
        submit = True
    if len(text) > POLICY_MAX_TYPE_CHARS:
        text = text[:POLICY_MAX_TYPE_CHARS]
    _AUTOMATOR.type_text(text)
    if submit:
        _AUTOMATOR.hotkey("enter")
    return "typed"


@tool
def wait(seconds: float = 1.0) -> str:
    """Sleep for a number of seconds."""
    sec = max(0.0, float(seconds))
    if sec > POLICY_MAX_WAIT_SECONDS:
        sec = POLICY_MAX_WAIT_SECONDS
    _AUTOMATOR.wait(sec)
    return f"waited {sec}s"


try:
    import pygetwindow as gw  # optional

    @tool
    def focus_window(title_substring: str) -> str:
        """Bring the first window whose title contains the substring to the foreground.

        Returns a status string; if not found, returns 'not found'.
        """
        windows = [w for w in gw.getAllWindows() if title_substring.lower() in w.title.lower()]
        if not windows:
            return "not found"
        w = windows[0]
        if w.isMinimized:
            w.restore()
        w.activate()
        return f"focused: {w.title}"
except Exception:
    # Optional dependency not available; tool won't be registered
    pass
