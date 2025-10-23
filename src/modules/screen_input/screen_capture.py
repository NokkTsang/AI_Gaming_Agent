import time
from typing import Tuple, Optional, List
import difflib
import mss
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import sys
import os

# You may want to load these from your config system
DEFAULT_OUTPUT_DIR = "src/modules/screen_input/screenshots"
FULLSCREEN = None


_DPI_AWARE_SET = False


def _ensure_windows_dpi_awareness():
    """Ensure the process is DPI-aware so pygetwindow/mss coordinates align on high-DPI displays.

    On Windows with scaling (125%/150%), window coordinates and captured pixels
    can mismatch unless the process is marked DPI-aware.
    """
    global _DPI_AWARE_SET
    if _DPI_AWARE_SET:
        return
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        try:
            # Per-monitor DPI awareness v2 (Windows 10+)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            # Fallback: system-wide DPI aware
            ctypes.windll.user32.SetProcessDPIAware()
        _DPI_AWARE_SET = True
        # print("[DEBUG] DPI awareness enabled")
    except Exception:
        # Ignore if not available
        pass


def get_fullscreen_region() -> Tuple[int, int, int, int]:
    """Detects and returns the region for the primary monitor (full screen)."""
    # On macOS, use ImageGrab to get proper Retina resolution
    if sys.platform == "darwin":
        # ImageGrab.grab() returns full screen at native resolution
        return None  # Signal to use ImageGrab.grab() without bbox

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 1 is the primary monitor
        return (monitor["left"], monitor["top"], monitor["width"], monitor["height"])


def _enum_windows_with_titles() -> List[Tuple[int, str]]:
    """Windows-only: enumerate top-level visible windows and return list of (hwnd, title)."""
    if not sys.platform.startswith("win"):
        return []
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        IsWindowVisible = user32.IsWindowVisible
        GetWindowTextLengthW = user32.GetWindowTextLengthW
        GetWindowTextW = user32.GetWindowTextW

        results: List[Tuple[int, str]] = []

        def callback(hwnd, lParam):
            if IsWindowVisible(hwnd):
                length = GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    if title:
                        results.append((int(hwnd), title))
            return True

        EnumWindows(EnumWindowsProc(callback), 0)
        return results
    except Exception:
        return []


def _select_window_by_title(query: str) -> Optional[int]:
    """Windows-only: select best HWND by substring first, else fuzzy match."""
    if not sys.platform.startswith("win"):
        return None
    windows = _enum_windows_with_titles()
    if not windows:
        print("   Warning: No windows detected")
        return None
    print("\n[Window List]")
    for i, (_, title) in enumerate(windows, 1):
        print(f"  {i}. {title}")

    # Substring match
    subs = [(hwnd, title) for hwnd, title in windows if query.lower() in title.lower()]
    if subs:
        # choose the longest title among matches
        hwnd, title = max(subs, key=lambda x: len(x[1]))
        print(f"\n[Match] Substring matched window: '{title}' for query '{query}'")
        return hwnd

    # Fuzzy match
    titles = [title for _, title in windows]
    best = difflib.get_close_matches(query, titles, n=1, cutoff=0.3)
    if best:
        best_title = best[0]
        for hwnd, title in windows:
            if title == best_title:
                print(f"\n[Match] Fuzzy selected window: '{title}' for query '{query}'")
                return hwnd
    print(f"   Warning: No similar window found for: '{query}'")
    return None


def _capture_window_printwindow(title_substring: str) -> Optional[Image.Image]:
    """Capture a window bitmap via Win32 PrintWindow API (can work for background windows).

    Note: Success depends on the target application. Some apps/windows may not render
    when minimized or may return blank images.

    Returns a PIL Image on success, else None.
    """
    try:
        import win32gui
        import win32ui
        import win32con
        import win32api
        import ctypes
        from ctypes import wintypes
    except Exception as e:
        # Missing Windows-specific dependencies
        print(f"   Info: PrintWindow not available: {e}")
        return None

    # Find window by title (substring/fuzzy)
    hwnd = _select_window_by_title(title_substring)
    if not hwnd:
        print(f"   Warning: No window found for PrintWindow: '{title_substring}'")
        return None

    try:
        # Preferred: use DWM extended frame bounds to avoid shadow/border offsets
        try:
            DWMWA_EXTENDED_FRAME_BOUNDS = 9
            rect = wintypes.RECT()
            ctypes.windll.dwmapi.DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS,
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
            left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            # Fallback to classic window rect
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)

        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            print(f"   Warning: Window size invalid for PrintWindow: {width}x{height}")
            return None

        # Create device contexts & bitmap
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # Try PrintWindow via ctypes (works even if win32gui lacks wrapper)
        PW_RENDERFULLCONTENT = 0x00000002  # may render more complete content on newer Windows
        try:
            result = ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
        except Exception:
            # Fallback to win32gui if available
            result = getattr(win32gui, "PrintWindow", lambda *args, **kwargs: 0)(
                hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT
            )

        # Convert to PIL Image
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        # If bitmap size differs from DWM bounds, crop to expected width/height
        try:
            if img.size != (width, height):
                img = img.crop((0, 0, width, height))
        except Exception:
            pass

        # Cleanup GDI objects
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        if result != 1:
            # PrintWindow failed to render content fully
            print("   Info: PrintWindow returned partial/empty content; falling back if needed")
            return None

        return img
    except Exception as e:
        print(f"   Warning: PrintWindow capture failed: {e}")
        try:
            # Ensure DCs are released in case of exception
            win32gui.ReleaseDC(hwnd, hwnd_dc)  # may fail if not set
        except Exception:
            pass
        return None


def take_screenshot(
    tid: Optional[float] = None,
    screen_region: Optional[Tuple[int, int, int, int]] = FULLSCREEN,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    draw_axis: bool = False,
    crop_border: bool = False,
    window_title: Optional[str] = None,
    focus_window: bool = False,
    method: str = "auto",  # 'auto' | 'printwindow'
) -> str:
    """
    Capture a screenshot of the specified region and save as JPEG.
    Optionally draw axis and crop border.
    Returns the path to the saved screenshot.
    """
    _ensure_windows_dpi_awareness()

    if tid is None:
        tid = time.time()
    # If a window title is provided, try Windows background capture via PrintWindow
    if window_title:
        if sys.platform.startswith("win"):
            # Attempt Windows background capture first if allowed (auto/printwindow)
            if method in ("auto", "printwindow"):
                img = _capture_window_printwindow(window_title)
                if img is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")
                    img.convert("RGB").save(screen_image_filename, "JPEG")
                    return screen_image_filename
            # No region fallback; go fullscreen
            print("   Info: Window capture failed or not permitted; using fullscreen.")
        else:
            print("   Info: Window capture is implemented for Windows only. Using fullscreen.")
    if screen_region is FULLSCREEN:
        screen_region = get_fullscreen_region()

    os.makedirs(output_dir, exist_ok=True)
    screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")

    # On macOS, use ImageGrab for proper Retina resolution when capturing full screen
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
        screen_region=FULLSCREEN,  # Or specify a custom region in list format [left, top, width, height]
        output_dir=DEFAULT_OUTPUT_DIR,
        draw_axis=False,
        crop_border=False,
    )
    print(f"Screenshot saved to {screenshot_path}")
