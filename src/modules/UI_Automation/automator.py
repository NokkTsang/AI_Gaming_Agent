import time
from typing import Optional, Tuple, Dict, Any

import pyautogui

"""
Fundamental atomic keyboard/mouse actions:
    - left click
    - right click
    - double click
    - move mouse
    - scroll wheel
    - press key
"""


class UIAutomator:
    """
    Thin wrapper around pyautogui with a stable API for our agent.

    Coordinates are expected to be absolute screen pixels unless noted.
    """

    def __init__(self, fail_safe: bool = True, pause: float = 0.05) -> None:
        pyautogui.FAILSAFE = fail_safe
        pyautogui.PAUSE = pause

    # ---- Mouse ----
    def move_to(self, x: float, y: float, duration: float = 0.0) -> None:
        pyautogui.moveTo(x, y, duration=duration)

    def click(
        self,
        x: float,
        y: float,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        pyautogui.click(x, y, button=button, clicks=clicks, interval=interval)

    def double_click(self, x: float, y: float, button: str = "left") -> None:
        pyautogui.doubleClick(x, y, button=button)

    def right_click(self, x: float, y: float) -> None:
        pyautogui.click(x, y, button="right")

    def drag_to(
        self, x: float, y: float, duration: float = 0.3, button: str = "left"
    ) -> None:
        pyautogui.dragTo(x, y, duration=duration, button=button)

    # ---- Keyboard ----
    def hotkey(self, *keys: str) -> None:
        pyautogui.hotkey(*keys)

    def key_down(self, key: str) -> None:
        pyautogui.keyDown(key)

    def key_up(self, key: str) -> None:
        pyautogui.keyUp(key)

    def type_text(self, text: str, interval: float = 0.02) -> None:
        pyautogui.write(text, interval=interval)

    # ---- Scroll ----
    def scroll(
        self, clicks: int, x: Optional[int] = None, y: Optional[int] = None
    ) -> None:
        if x is None or y is None:
            pyautogui.scroll(clicks)
        else:
            pyautogui.scroll(clicks, x=x, y=y)

    # ---- Utility ----
    def wait(self, seconds: float) -> None:
        time.sleep(seconds)
