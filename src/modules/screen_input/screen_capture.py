import time
from typing import Tuple, Optional
import mss
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import sys

# You may want to load these from your config system
DEFAULT_OUTPUT_DIR = "src/modules/screen_input/screenshots"
FULLSCREEN = None


def get_fullscreen_region() -> Tuple[int, int, int, int]:
    """Detects and returns the region for the primary monitor (full screen)."""
    # On macOS, use ImageGrab to get proper Retina resolution
    if sys.platform == "darwin":
        # ImageGrab.grab() returns full screen at native resolution
        return None  # Signal to use ImageGrab.grab() without bbox

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 1 is the primary monitor
        return (monitor["left"], monitor["top"], monitor["width"], monitor["height"])


def take_screenshot(
    tid: Optional[float] = None,
    screen_region: Optional[Tuple[int, int, int, int]] = FULLSCREEN,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    draw_axis: bool = False,
    crop_border: bool = False,
) -> str:
    """
    Capture a screenshot of the specified region and save as JPEG.
    Optionally draw axis and crop border.
    Returns the path to the saved screenshot.
    """
    import os

    if tid is None:
        tid = time.time()
    if screen_region is FULLSCREEN:
        screen_region = get_fullscreen_region()

    os.makedirs(output_dir, exist_ok=True)
    screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")

    # On macOS, use ImageGrab for proper Retina resolution
    if sys.platform == "darwin" and screen_region is None:
        image = ImageGrab.grab(
            all_screens=False
        )  # Capture primary screen at full resolution
        # Convert RGBA to RGB for JPEG saving
        if image.mode == "RGBA":
            image = image.convert("RGB")
    else:
        # Use mss for other platforms or specific regions
        region = {
            "left": screen_region[0],
            "top": screen_region[1],
            "width": screen_region[2],
            "height": screen_region[3],
        }
        with mss.mss() as sct:
            screen_image = sct.grab(region)
            image = Image.frombytes(
                "RGB", screen_image.size, screen_image.bgra, "raw", "BGRX"
            )

    if draw_axis:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        cx, cy = width // 2, height // 2
        draw.line((cx, 0, cx, height), fill="blue", width=3)
        draw.line((0, cy, width, cy), fill="blue", width=3)
        # Optionally add axis labels here

    image.save(screen_image_filename, "JPEG")

    if crop_border:
        # Implement border cropping here
        pass

    return screen_image_filename


if __name__ == "__main__":
    screenshot_path = take_screenshot(
        tid=None,
        screen_region=FULLSCREEN,  # Or specify a custom region in list format [left, top, width, h
        output_dir=DEFAULT_OUTPUT_DIR,
        draw_axis=False,
        crop_border=False,
    )
    print(f"Screenshot saved to {screenshot_path}")
