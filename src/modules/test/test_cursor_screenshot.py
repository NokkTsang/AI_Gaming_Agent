#!/usr/bin/env python3
"""
Test script to demonstrate cursor inclusion in screenshots.
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from modules.screen_input.screen_capture import take_screenshot
import pyautogui


def test_cursor_in_screenshot():
    """Test that cursor is drawn on screenshot."""
    print("Testing cursor drawing in screenshots...")
    print("\nMove your mouse to different positions to see the cursor captured!\n")

    output_dir = "cursor_test_screenshots"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(3):
        print(f"Capturing screenshot {i+1}/3 in 2 seconds...")
        print("Move your mouse now!")
        time.sleep(2)

        # Get current cursor position
        cursor_x, cursor_y = pyautogui.position()
        print(f"Cursor at: ({cursor_x}, {cursor_y})")

        # Take screenshot WITH cursor
        screenshot_path_with, region_with = take_screenshot(
            tid=f"test_with_cursor_{i}",
            output_dir=output_dir,
            draw_cursor=True,  # Include cursor
        )
        print(f"✓ Screenshot with cursor saved: {screenshot_path_with}")

        # Take screenshot WITHOUT cursor (for comparison)
        screenshot_path_without, region_without = take_screenshot(
            tid=f"test_without_cursor_{i}",
            output_dir=output_dir,
            draw_cursor=False,  # Exclude cursor
        )
        print(f"✓ Screenshot without cursor saved: {screenshot_path_without}")
        print()

    print(f"\n✅ Test complete! Check the '{output_dir}' folder to see the difference.")
    print(f"   - Files with 'with_cursor' show cursor position")
    print(f"   - Files with 'without_cursor' have no cursor indicator")


if __name__ == "__main__":
    test_cursor_in_screenshot()
