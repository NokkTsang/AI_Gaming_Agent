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
        print(f"[DEBUG AUTOMATOR] move_to({x}, {y}, duration={duration})")
        pyautogui.moveTo(x, y, duration=duration)

    def click(
        self,
        x: float,
        y: float,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.1,
    ) -> None:
        print(
            f"[DEBUG AUTOMATOR] click({x}, {y}, button='{button}', clicks={clicks}, interval={interval})"
        )
        # First move to ensure we're at the right position
        pyautogui.moveTo(x, y, duration=0.5)
        time.sleep(0.3)  # Wait for hover effects to register

        # Verify position
        actual_pos = pyautogui.position()
        print(
            f"[DEBUG AUTOMATOR] Cursor moved to: {actual_pos}, expected: ({int(x)}, {int(y)})"
        )

        # Focus click first (for browser games/apps that need window focus)
        print(f"[DEBUG AUTOMATOR] Focus click to activate window")
        pyautogui.click(x, y, button=button)
        time.sleep(0.2)  # Wait for focus to register

        # Perform the actual clicks with proper timing
        for i in range(clicks):
            print(f"[DEBUG AUTOMATOR] Performing click {i+1}/{clicks}")
            # Use mouseDown/Up with longer hold for game compatibility
            pyautogui.mouseDown(button=button)
            time.sleep(
                0.2
            )  # Hold for 200ms to ensure registration (increased from 100ms)
            pyautogui.mouseUp(button=button)
            time.sleep(0.1)  # Brief pause after release

            if clicks > 1 and i < clicks - 1:
                time.sleep(interval)

        time.sleep(0.5)  # Longer wait to ensure click fully processes
        print(f"[DEBUG AUTOMATOR] click completed")

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
