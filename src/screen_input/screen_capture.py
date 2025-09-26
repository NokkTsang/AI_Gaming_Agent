import time
from typing import Tuple, Optional
import mss
from PIL import Image, ImageDraw, ImageFont

# You may want to load these from your config system
DEFAULT_REGION = (0, 0, 1920, 1080)  # left, top, width, height
DEFAULT_OUTPUT_DIR = "./screenshots"


def take_screenshot(
    tid: Optional[float] = None,
    screen_region: Optional[Tuple[int, int, int, int]] = None,
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
    if screen_region is None:
        screen_region = DEFAULT_REGION

    region = {
        "left": screen_region[0],
        "top": screen_region[1],
        "width": screen_region[2],
        "height": screen_region[3],
    }

    os.makedirs(output_dir, exist_ok=True)
    screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")

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
        # Placeholder: implement your border cropping here
        pass

    return screen_image_filename

if __name__ == "__main__":
    screenshot_path = take_screenshot(
        tid=None,
        screen_region=(0, 0, 800, 600),
        output_dir="./screenshots",
        draw_axis=True,
        crop_border=False,
    )
    print(f"Screenshot saved to {screenshot_path}")