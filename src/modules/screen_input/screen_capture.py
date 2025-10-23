import time
from typing import Tuple, Optional, List
import difflib
import mss
from PIL import Image, ImageDraw, ImageFont, ImageGrab
import sys
import os
import io
import subprocess
import shutil
import re

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


def _enum_windows_windows() -> List[Tuple[int, str]]:
    """Enumerate Windows top-level visible windows: returns (hwnd, title)."""
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


def _enum_windows_macos() -> List[Tuple[int, str]]:
    """Enumerate macOS windows: returns (window_id, title)."""
    try:
        import Quartz
        options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListOptionIncludingWindow
        # Use All windows info to extract titles
        window_info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
        results: List[Tuple[int, str]] = []
        for info in window_info_list or []:
            wid = info.get(Quartz.kCGWindowNumber)
            owner = info.get(Quartz.kCGWindowOwnerName, '') or ''
            name = info.get(Quartz.kCGWindowName, '') or ''
            title = (f"{owner} - {name}" if owner and name else owner or name).strip()
            if wid and title:
                results.append((int(wid), title))
        return results
    except Exception:
        return []


def _enum_windows_linux() -> List[Tuple[str, str]]:
    """Enumerate Linux windows using wmctrl or xdotool: returns (win_id, title)."""
    # Prefer wmctrl if available
    results: List[Tuple[str, str]] = []
    try:
        if shutil.which('wmctrl'):
            proc = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    parts = line.split(None, 3)
                    if len(parts) >= 4:
                        wid_hex, _desktop, _host, title = parts
                        title = title.strip()
                        if title:
                            results.append((wid_hex, title))
                return results
        # Fallback: xdotool
        if shutil.which('xdotool'):
            ids_proc = subprocess.run(['xdotool', 'search', '--name', '.+'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if ids_proc.returncode == 0:
                for wid in ids_proc.stdout.split():
                    name_proc = subprocess.run(['xdotool', 'getwindowname', wid], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if name_proc.returncode == 0:
                        title = (name_proc.stdout or '').strip()
                        if title:
                            results.append((wid, title))
        return results
    except Exception:
        return results


def _select_window_by_title(query: str):
    """Select best matching window handle/ID by substring first, else fuzzy match.

    Returns OS-specific identifier:
    - Windows: int HWND
    - macOS: int CGWindowNumber
    - Linux: str window id (hex or decimal)
    """
    if sys.platform.startswith("win"):
        windows = _enum_windows_windows()
    elif sys.platform == 'darwin':
        windows = _enum_windows_macos()
    else:
        windows = _enum_windows_linux()
    if not windows:
        print("   Warning: No windows detected")
        return None
    print("\n[Window List]")
    for i, (_, title) in enumerate(windows, 1):
        print(f"  {i}. {title}")

    # Substring match
    subs = [(wid, title) for wid, title in windows if query.lower() in title.lower()]
    if subs:
        # choose the longest title among matches
        wid, title = max(subs, key=lambda x: len(x[1]))
        print(f"\n[Match] Substring matched window: '{title}' for query '{query}'")
        return wid

    # Fuzzy match
    titles = [title for _, title in windows]
    best = difflib.get_close_matches(query, titles, n=1, cutoff=0.3)
    if best:
        best_title = best[0]
        for wid, title in windows:
            if title == best_title:
                print(f"\n[Match] Fuzzy selected window: '{title}' for query '{query}'")
                return wid
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


def _capture_window_macos(title_substring: str) -> Optional[Image.Image]:
    """Capture a macOS window image via Quartz (best-effort, may be blank if minimized)."""
    try:
        import Quartz
    except Exception as e:
        print(f"   Info: macOS Quartz not available: {e}")
        return None

    wid = _select_window_by_title(title_substring)
    if not wid:
        print(f"   Warning: No window found on macOS: '{title_substring}'")
        return None
    try:
        opts = Quartz.kCGWindowImageBoundsIgnoreFraming
        image_ref = Quartz.CGWindowListCreateImage(Quartz.CGRectInfinite, Quartz.kCGWindowListOptionIncludingWindow, int(wid), opts)
        if not image_ref:
            print("   Info: CGWindowListCreateImage returned null")
            return None
        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)
        data_provider = Quartz.CGImageGetDataProvider(image_ref)
        data = Quartz.CGDataProviderCopyData(data_provider)
        buf = bytes(data)
        # macOS uses BGRA by default
        img = Image.frombuffer("RGBA", (width, height), buf, "raw", "BGRA", bytes_per_row, 1)
        return img.convert("RGB")
    except Exception as e:
        print(f"   Warning: macOS window capture failed: {e}")
        return None


def _capture_window_linux(title_substring: str) -> Optional[Image.Image]:
    """Capture a Linux window.

    Strategy:
    - If `xwd` exists: dump window to XWD and open via PIL (may work when occluded on some setups)
    - Else: use xwininfo to get geometry and grab region via mss (visible-only)
    """
    wid = _select_window_by_title(title_substring)
    if not wid:
        print(f"   Warning: No window found on Linux: '{title_substring}'")
        return None
    # Try xwd first
    if shutil.which('xwd'):
        try:
            proc = subprocess.run(['xwd', '-silent', '-id', str(wid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode == 0 and proc.stdout:
                try:
                    img = Image.open(io.BytesIO(proc.stdout))
                    return img.convert("RGB")
                except Exception:
                    pass
        except Exception as e:
            print(f"   Info: xwd capture failed: {e}")
    # Fallback: region capture using xwininfo geometry
    if shutil.which('xwininfo'):
        try:
            info = subprocess.run(['xwininfo', '-id', str(wid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if info.returncode == 0:
                text = info.stdout
                def find(pattern):
                    m = re.search(pattern, text)
                    return int(m.group(1)) if m else None
                x = find(r"Absolute upper-left X:\s*(\-?\d+)")
                y = find(r"Absolute upper-left Y:\s*(\-?\d+)")
                w = find(r"Width:\s*(\d+)")
                h = find(r"Height:\s*(\d+)")
                if None not in (x, y, w, h) and w > 0 and h > 0:
                    with mss.mss() as sct:
                        region = {"left": x, "top": y, "width": w, "height": h}
                        screen_image = sct.grab(region)
                        return Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
        except Exception as e:
            print(f"   Info: xwininfo region capture failed: {e}")
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
    # If a window title is provided, try OS-specific window capture first
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
        elif sys.platform == 'darwin':
            if method in ("auto",):
                img = _capture_window_macos(window_title)
                if img is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")
                    img.convert("RGB").save(screen_image_filename, "JPEG")
                    return screen_image_filename
            print("   Info: macOS window capture failed; using fullscreen.")
        else:
            if method in ("auto",):
                img = _capture_window_linux(window_title)
                if img is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")
                    img.convert("RGB").save(screen_image_filename, "JPEG")
                    return screen_image_filename
            print("   Info: Linux window capture failed; using fullscreen.")
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
