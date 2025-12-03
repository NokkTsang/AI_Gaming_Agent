from __future__ import annotations

import ast
from typing import Dict, Any, List, Tuple

from .atomic_actions import UIAutomator


def _center_of_box(box: List[float]) -> Tuple[float, float]:
    # box can be [x1, y1, x2, y2] or [x, y]
    if len(box) == 2:
        x1, y1 = box
        x2, y2 = x1, y1
    else:
        x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


KEY_MAP = {
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    "space": " ",
}


class ActionExecutor:
    """
    Execute generic UI action dictionaries with UIAutomator (automator.py).

    Expected action dict shape (canonical):
    {
      "action_type": str,  # e.g. "click", "drag", "scroll", "hotkey", "type", "wait", etc.
      "action_inputs": {   # optional, tool-specific params
          ...
      }
    }

    Boxes can be provided either as:
    - normalized Python lists/tuples: [x, y] or [x1, y1, x2, y2] (values in [0,1]), or
    - string representations of such lists (e.g., "[0.1, 0.2, 0.3, 0.4]").

    This class is intentionally decoupled from any specific parser; it accepts
    the generic structure above regardless of the upstream source.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        screen_left: int = 0,
        screen_top: int = 0,
        automator: UIAutomator | None = None,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_left = screen_left  # Offset from (0,0) for multi-monitor
        self.screen_top = screen_top
        self.automator = automator or UIAutomator()

    def _denorm(self, nx: float, ny: float) -> Tuple[float, float]:
        # Convert normalized [0,1] to absolute screen coordinates
        # accounting for monitor offset in multi-monitor setups
        x = nx * self.screen_width + self.screen_left
        y = ny * self.screen_height + self.screen_top
        return x, y

    def _coerce_box(self, box_val: Any) -> List[float]:
        """Accept a box as list/tuple or as string; return validated list of floats in [0,1]."""
        if isinstance(box_val, (list, tuple)):
            vals = [float(v) for v in box_val]
        else:
            vals = list(ast.literal_eval(str(box_val)))
            vals = [float(v) for v in vals]
        if len(vals) not in (2, 4):
            raise ValueError(f"box must have length 2 or 4, got {len(vals)}: {vals}")
        for v in vals:
            if v < 0.0 or v > 1.0:
                raise ValueError(f"box values must be normalized in [0,1], got {vals}")
        return vals

    def _key(self, k: str) -> str:
        return KEY_MAP.get(k.lower(), k)

    def execute(self, action: Dict[str, Any]) -> None:
        action_type = action.get("action_type")
        inputs = action.get("action_inputs", {})

        if action_type in {
            "click",
            "left_single",
            "hover",
            "move",
            "left_double",
            "right_single",
        }:
            start_box = inputs.get("start_box")
            if start_box:
                box = self._coerce_box(start_box)
                cx, cy = _center_of_box(box)
                x, y = self._denorm(cx, cy)
                if action_type in {"click", "left_single"}:
                    print(f"  → Executing: CLICK at ({int(x)}, {int(y)})")
                    self.automator.click(x, y, button="left")
                elif action_type == "left_double":
                    print(f"  → Executing: DOUBLE-CLICK at ({int(x)}, {int(y)})")
                    self.automator.double_click(x, y, button="left")
                elif action_type == "right_single":
                    print(f"  → Executing: RIGHT-CLICK at ({int(x)}, {int(y)})")
                    self.automator.right_click(x, y)
                elif action_type in {"move"}:
                    print(f"  → Executing: MOVE to ({int(x)}, {int(y)})")
                    self.automator.move(x, y, duration=0.5)
                else:
                    print(f"  → Executing: HOVER at ({int(x)}, {int(y)})")
                    self.automator.move_to(x, y)

        elif action_type in {"drag", "select"}:
            sb, eb = inputs.get("start_box"), inputs.get("end_box")
            if sb and eb:
                sbox, ebox = self._coerce_box(sb), self._coerce_box(eb)
                sx, sy = self._denorm(*_center_of_box(sbox))
                ex, ey = self._denorm(*_center_of_box(ebox))
                self.automator.move_to(sx, sy)
                self.automator.drag_to(ex, ey)

        elif action_type == "scroll":
            sb = inputs.get("start_box")
            direction = (inputs.get("direction") or "").lower()
            clicks = 5 if ("up" in direction) else -5
            if sb:
                box = self._coerce_box(sb)
                x, y = self._denorm(*_center_of_box(box))
                self.automator.scroll(clicks, int(x), int(y))
            else:
                self.automator.scroll(clicks)

        elif action_type in {"hotkey"}:
            hotkey = inputs.get("key") or inputs.get("hotkey") or ""
            if hotkey:
                # Click center of screen first to ensure window focus (general-purpose fix)
                # This handles cases where keyboard input goes to wrong window
                center_x = self.screen_left + self.screen_width / 2
                center_y = self.screen_top + self.screen_height / 2
                self.automator.click(center_x, center_y, button="left", clicks=1)
                import time

                time.sleep(0.1)  # Brief pause after focus click

                keys = [self._key(k) for k in hotkey.split()]
                self.automator.hotkey(*keys)

        elif action_type in {"press", "keydown"}:
            key = inputs.get("key") or inputs.get("press")
            if key:
                self.automator.key_down(self._key(key))

        elif action_type in {"release", "keyup"}:
            key = inputs.get("key") or inputs.get("press")
            if key:
                self.automator.key_up(self._key(key))

        elif action_type == "type":
            content = inputs.get("content") or ""
            if content.endswith("\\n") or content.endswith("\n"):
                content = content.rstrip("\\n").rstrip("\n")
                self.automator.type_text(content)
                self.automator.hotkey("enter")
            else:
                self.automator.type_text(content)

        elif action_type == "wait":
            # Optional duration later; default 5s to align with prompt doc
            self.automator.wait(5)

        elif action_type == "finished":
            # No-op here; upstream handles termination
            return

        else:
            # Unknown action: ignore or log
            return
