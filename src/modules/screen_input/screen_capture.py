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
import pyautogui

# You may want to load these from your config system
DEFAULT_OUTPUT_DIR = "src/modules/screen_input/screenshots"
FULLSCREEN = None


_DPI_AWARE_SET = False


def _draw_cursor_on_image(
    image: Image.Image, region: Tuple[int, int, int, int]
) -> Image.Image:
    """Draw cursor on the captured image.

    Args:
        image: PIL Image to draw cursor on
        region: (left, top, width, height) of the captured region

    Returns:
        Image with cursor drawn on it
    """
    try:
        # Get current cursor position
        cursor_x, cursor_y = pyautogui.position()

        # Calculate cursor position relative to the captured region
        region_left, region_top, region_width, region_height = region
        rel_x = cursor_x - region_left
        rel_y = cursor_y - region_top

        # Check if cursor is within the captured region
        if 0 <= rel_x < region_width and 0 <= rel_y < region_height:
            # Draw cursor on image
            draw = ImageDraw.Draw(image)

            # Draw a simple cursor (triangle pointing up-left)
            cursor_size = 15
            cursor_points = [
                (rel_x, rel_y),  # tip
                (rel_x, rel_y + cursor_size),  # bottom
                (rel_x + cursor_size * 0.6, rel_y + cursor_size * 0.6),  # right
            ]

            # Draw cursor with outline for visibility
            draw.polygon(cursor_points, fill="white", outline="black", width=2)

            print(
                f"   Info: Drew cursor at screen position ({cursor_x}, {cursor_y}), image position ({rel_x}, {rel_y})"
            )
        else:
            print(
                f"   Info: Cursor at ({cursor_x}, {cursor_y}) is outside captured region"
            )

    except Exception as e:
        print(f"   Warning: Failed to draw cursor: {e}")

    return image


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


def get_fullscreen_region(monitor_index: int = 1) -> Tuple[int, int, int, int]:
    """Detects and returns the region for a specific monitor.

    Args:
        monitor_index: 0 for all monitors combined, 1 for primary, 2 for secondary, etc.

    Returns:
        Tuple of (left, top, width, height)
    """
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        return (monitor["left"], monitor["top"], monitor["width"], monitor["height"])


def _enum_windows_windows() -> List[Tuple[int, str]]:
    """Enumerate Windows top-level visible windows: returns (hwnd, title)."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, wintypes.HWND, wintypes.LPARAM
        )
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

        options = (
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListOptionIncludingWindow
        )
        # Use All windows info to extract titles
        window_info_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
        )
        results: List[Tuple[int, str]] = []
        for info in window_info_list or []:
            wid = info.get(Quartz.kCGWindowNumber)
            owner = info.get(Quartz.kCGWindowOwnerName, "") or ""
            name = info.get(Quartz.kCGWindowName, "") or ""
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
        if shutil.which("wmctrl"):
            proc = subprocess.run(
                ["wmctrl", "-l"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
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
        if shutil.which("xdotool"):
            ids_proc = subprocess.run(
                ["xdotool", "search", "--name", ".+"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if ids_proc.returncode == 0:
                for wid in ids_proc.stdout.split():
                    name_proc = subprocess.run(
                        ["xdotool", "getwindowname", wid],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if name_proc.returncode == 0:
                        title = (name_proc.stdout or "").strip()
                        if title:
                            results.append((wid, title))
        return results
    except Exception:
        return results


def _select_window_by_title(query: str, verbose: bool = True):
    """Select best matching window handle/ID by substring first, else fuzzy match.

    Returns OS-specific identifier:
    - Windows: int HWND
    - macOS: int CGWindowNumber
    - Linux: str window id (hex or decimal)
    """
    if sys.platform.startswith("win"):
        windows = _enum_windows_windows()
    elif sys.platform == "darwin":
        windows = _enum_windows_macos()
    else:
        windows = _enum_windows_linux()
    if not windows:
        if verbose:
            print("   Warning: No windows detected")
        return None

    # Substring match
    subs = [(wid, title) for wid, title in windows if query.lower() in title.lower()]
    if subs:
        # choose the longest title among matches
        wid, title = max(subs, key=lambda x: len(x[1]))
        if verbose:
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


def get_window_monitor(title_substring: str) -> Tuple[int, Tuple[int, int, int, int]]:
    """Detect which monitor a window is on and return monitor info.

    Returns:
        Tuple of (monitor_index, (left, top, width, height))
        Returns (1, primary_monitor_region) as fallback
    """
    try:
        import mss

        with mss.mss() as sct:
            # Get window bounds first
            window_bounds = None

            if sys.platform == "darwin":
                try:
                    import Quartz

                    wid = _select_window_by_title(title_substring)
                    if wid:
                        window_info_list = Quartz.CGWindowListCopyWindowInfo(
                            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
                        )
                        for info in window_info_list or []:
                            if info.get(Quartz.kCGWindowNumber) == wid:
                                bounds = info.get(Quartz.kCGWindowBounds)
                                if bounds:
                                    x = int(bounds.get("X", 0))
                                    y = int(bounds.get("Y", 0))
                                    w = int(bounds.get("Width", 0))
                                    h = int(bounds.get("Height", 0))
                                    window_bounds = (x, y, w, h)
                                break
                except:
                    pass

            elif sys.platform.startswith("win"):
                try:
                    import win32gui

                    hwnd = _select_window_by_title(title_substring)
                    if hwnd:
                        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                        window_bounds = (left, top, right - left, bottom - top)
                except:
                    pass

            else:  # Linux
                try:
                    wid = _select_window_by_title(title_substring)
                    if wid and shutil.which("xwininfo"):
                        info = subprocess.run(
                            ["xwininfo", "-id", str(wid)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        if info.returncode == 0:
                            text = info.stdout

                            def find(pattern):
                                m = re.search(pattern, text)
                                return int(m.group(1)) if m else None

                            x = find(r"Absolute upper-left X:\s*(\-?\d+)")
                            y = find(r"Absolute upper-left Y:\s*(\-?\d+)")
                            w = find(r"Width:\s*(\d+)")
                            h = find(r"Height:\s*(\d+)")
                            if None not in (x, y, w, h):
                                window_bounds = (x, y, w, h)
                except:
                    pass

            # If we got window bounds, find which monitor contains the center
            if window_bounds:
                wx, wy, ww, wh = window_bounds
                window_center_x = wx + ww // 2
                window_center_y = wy + wh // 2

                # Check each monitor
                for i, monitor in enumerate(
                    sct.monitors[1:], start=1
                ):  # Skip monitor[0] (all)
                    m_left = monitor["left"]
                    m_top = monitor["top"]
                    m_right = m_left + monitor["width"]
                    m_bottom = m_top + monitor["height"]

                    # Check if window center is within this monitor
                    if (
                        m_left <= window_center_x < m_right
                        and m_top <= window_center_y < m_bottom
                    ):
                        print(f"   Info: Window is on monitor {i}")
                        return (i, (m_left, m_top, monitor["width"], monitor["height"]))

            # Fallback to primary monitor
            monitor = sct.monitors[1]
            return (
                1,
                (monitor["left"], monitor["top"], monitor["width"], monitor["height"]),
            )

    except Exception as e:
        print(f"   Warning: Failed to detect window monitor: {e}")
        # Return primary monitor as fallback
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            return (
                1,
                (monitor["left"], monitor["top"], monitor["width"], monitor["height"]),
            )


def focus_window_by_title(title_substring: str) -> bool:
    """Bring window to front and activate it.

    Returns True if successful, False otherwise.
    """
    if sys.platform == "darwin":
        try:
            import Quartz
            import subprocess

            # Get window ID
            wid = _select_window_by_title(title_substring)
            if not wid:
                return False

            # Get window info to find the owning application
            window_info_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
            )
            owner_name = None
            for info in window_info_list or []:
                if info.get(Quartz.kCGWindowNumber) == wid:
                    owner_name = info.get(Quartz.kCGWindowOwnerName)
                    break

            if owner_name:
                # Use AppleScript to activate the application
                script = f'tell application "{owner_name}" to activate'
                subprocess.run(
                    ["osascript", "-e", script], capture_output=True, timeout=2
                )
                print(
                    f"   Info: Focused window '{title_substring}' (app: {owner_name})"
                )
                return True
            return False
        except Exception as e:
            print(f"   Warning: Failed to focus window: {e}")
            return False
    elif sys.platform.startswith("win"):
        try:
            import win32gui
            import win32con

            hwnd = _select_window_by_title(title_substring)
            if not hwnd:
                return False

            # Bring window to foreground
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            print(f"   Info: Focused window '{title_substring}'")
            return True
        except Exception as e:
            print(f"   Warning: Failed to focus window: {e}")
            return False
    else:  # Linux
        try:
            wid = _select_window_by_title(title_substring)
            if not wid:
                return False

            # Try wmctrl first
            if shutil.which("wmctrl"):
                subprocess.run(
                    ["wmctrl", "-i", "-a", str(wid)], capture_output=True, timeout=2
                )
                print(f"   Info: Focused window '{title_substring}'")
                return True
            # Fallback to xdotool
            elif shutil.which("xdotool"):
                subprocess.run(
                    ["xdotool", "windowactivate", str(wid)],
                    capture_output=True,
                    timeout=2,
                )
                print(f"   Info: Focused window '{title_substring}'")
                return True
            return False
        except Exception as e:
            print(f"   Warning: Failed to focus window: {e}")
            return False


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
        PW_RENDERFULLCONTENT = (
            0x00000002  # may render more complete content on newer Windows
        )
        try:
            result = ctypes.windll.user32.PrintWindow(
                hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT
            )
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
            print(
                "   Info: PrintWindow returned partial/empty content; falling back if needed"
            )
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

    wid = _select_window_by_title(title_substring, verbose=False)
    if not wid:
        print(f"   Warning: No window found on macOS: '{title_substring}'")
        return None
    try:
        opts = Quartz.kCGWindowImageBoundsIgnoreFraming
        image_ref = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,  # Use CGRectNull to capture only the window bounds
            Quartz.kCGWindowListOptionIncludingWindow,
            int(wid),
            opts,
        )
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
        img = Image.frombuffer(
            "RGBA", (width, height), buf, "raw", "BGRA", bytes_per_row, 1
        )
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
    if shutil.which("xwd"):
        try:
            proc = subprocess.run(
                ["xwd", "-silent", "-id", str(wid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode == 0 and proc.stdout:
                try:
                    img = Image.open(io.BytesIO(proc.stdout))
                    return img.convert("RGB")
                except Exception:
                    pass
        except Exception as e:
            print(f"   Info: xwd capture failed: {e}")
    # Fallback: region capture using xwininfo geometry
    if shutil.which("xwininfo"):
        try:
            info = subprocess.run(
                ["xwininfo", "-id", str(wid)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
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
                        return Image.frombytes(
                            "RGB", screen_image.size, screen_image.bgra, "raw", "BGRX"
                        )
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
    monitor_index: int = 1,  # 0=all, 1=primary, 2=secondary, etc.
    draw_cursor: bool = True,  # Whether to draw cursor on screenshot
) -> Tuple[str, Tuple[int, int, int, int]]:
    """
    Capture a screenshot of the specified region and save as JPEG.

    Args:
        window_title: If provided, captures ONLY the window with black background (noise reduction)
        monitor_index: Which monitor to capture (0=all, 1=primary, 2=secondary)
        focus_window: Whether to focus the window before capturing
        draw_cursor: Whether to draw cursor position on the screenshot (default: True)

    Returns:
        Tuple of (screenshot_path, (left, top, width, height) of captured region)
        The region info is crucial for coordinate mapping
    """
    _ensure_windows_dpi_awareness()

    if tid is None:
        tid = time.time()

    # Window mode: Capture ONLY the window for better resolution
    if window_title:
        # Focus the window first to ensure it's active (important for typing actions)
        if focus_window:
            focus_window_by_title(window_title)
            time.sleep(0.3)  # Give window time to come to front

        os.makedirs(output_dir, exist_ok=True)
        screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")

        # Use platform-specific window capture
        window_image = None
        if sys.platform == "darwin":
            window_image = _capture_window_macos(window_title)
        elif sys.platform.startswith("win"):
            if method == "printwindow":
                window_image = _capture_window_printwindow(window_title)
        elif sys.platform.startswith("linux"):
            window_image = _capture_window_linux(window_title)

        if window_image:
            # Get window bounds for coordinate mapping
            window_bounds = None
            if sys.platform == "darwin":
                try:
                    import Quartz

                    wid = _select_window_by_title(window_title, verbose=False)
                    if wid:
                        window_info_list = Quartz.CGWindowListCopyWindowInfo(
                            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
                        )
                        for info in window_info_list or []:
                            if info.get(Quartz.kCGWindowNumber) == wid:
                                bounds = info.get(Quartz.kCGWindowBounds)
                                if bounds:
                                    x = int(bounds.get("X", 0))
                                    y = int(bounds.get("Y", 0))
                                    w = int(bounds.get("Width", 0))
                                    h = int(bounds.get("Height", 0))
                                    window_bounds = (x, y, w, h)
                                break
                except Exception as e:
                    print(f"   Warning: Could not get window bounds: {e}")

            elif sys.platform.startswith("win"):
                try:
                    import win32gui

                    hwnd = _select_window_by_title(window_title, verbose=False)
                    if hwnd:
                        rect = win32gui.GetWindowRect(hwnd)
                        window_bounds = (
                            rect[0],
                            rect[1],
                            rect[2] - rect[0],
                            rect[3] - rect[1],
                        )
                except Exception as e:
                    print(f"   Warning: Could not get window bounds: {e}")

            elif sys.platform.startswith("linux"):
                try:
                    wid = _select_window_by_title(window_title, verbose=False)
                    if wid and shutil.which("xwininfo"):
                        proc = subprocess.run(
                            ["xwininfo", "-id", str(wid)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                        if proc.returncode == 0:
                            text = proc.stdout

                            def find(pattern):
                                m = re.search(pattern, text)
                                return int(m.group(1)) if m else None

                            x = find(r"Absolute upper-left X:\s*(\-?\d+)")
                            y = find(r"Absolute upper-left Y:\s*(\-?\d+)")
                            w = find(r"Width:\s*(\d+)")
                            h = find(r"Height:\s*(\d+)")
                            if None not in (x, y, w, h):
                                window_bounds = (x, y, w, h)
                except Exception as e:
                    print(f"   Warning: Could not get window bounds: {e}")

            if window_bounds:
                # Draw cursor if requested
                if draw_cursor:
                    window_image = _draw_cursor_on_image(window_image, window_bounds)

                window_image.save(screen_image_filename, "JPEG")
                print(
                    f"   Info: Captured window only - position ({window_bounds[0]}, {window_bounds[1]}) size {window_bounds[2]}x{window_bounds[3]}"
                )
                return screen_image_filename, window_bounds
            else:
                # Fallback: use image size and assume position (0, 0)
                w, h = window_image.size
                fallback_bounds = (0, 0, w, h)

                # Draw cursor if requested
                if draw_cursor:
                    window_image = _draw_cursor_on_image(window_image, fallback_bounds)

                window_image.save(screen_image_filename, "JPEG")
                print(f"   Warning: Could not get window position, assuming (0, 0)")
                return screen_image_filename, fallback_bounds
        else:
            # Window capture failed - this can happen if window is minimized or not accessible
            print(f"   ⚠️  ERROR: Could not capture window '{window_title}'")
            print(f"   Possible causes:")
            print(f"     - Window is minimized")
            print(f"     - Window name changed")
            print(
                f"     - macOS permissions issue (System Settings → Privacy → Screen Recording)"
            )
            raise RuntimeError(f"Window capture failed for '{window_title}'")

    # Fullscreen mode: use specified monitor_index
    if screen_region is FULLSCREEN:
        screen_region = get_fullscreen_region(monitor_index)

    os.makedirs(output_dir, exist_ok=True)
    screen_image_filename = os.path.join(output_dir, f"screen_{tid}.jpg")

    # Use mss consistently across all platforms
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

    # Draw cursor if requested
    if draw_cursor:
        image = _draw_cursor_on_image(image, screen_region)

    image.save(screen_image_filename, "JPEG")

    if crop_border:
        # Implement border cropping here
        pass

    return screen_image_filename, screen_region


if __name__ == "__main__":
    screenshot_path = take_screenshot(
        tid=None,
        screen_region=FULLSCREEN,  # Or specify a custom region in list format [left, top, width, height]
        output_dir=DEFAULT_OUTPUT_DIR,
        draw_axis=False,
        crop_border=False,
    )
    print(f"Screenshot saved to {screenshot_path}")
