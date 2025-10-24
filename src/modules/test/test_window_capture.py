"""
Window capture test: captures a specified window (or auto-selected one) using
Windows PrintWindow API (background-capable) when available; otherwise falls
back to fullscreen. Saves images under src/modules/test with timestamped filenames.

Run from project root:
    python -m src.modules.test.test_window_capture [optional_window_title]
"""

import os
import sys
from datetime import datetime

# Ensure repo root is on sys.path so `modules.*` imports work when not using -m
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("Window Capture Test")
print("=" * 80)

# Try to get window title from argv, then env variable
user_title = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
if not user_title:
    user_title = os.getenv("WINDOW_TITLE")


def _pick_default_window() -> str | None:
    """Enumerate top-level windows and pick a reasonable default title."""
    # Windows
    if sys.platform.startswith("win"):
        return _pick_default_window_windows()
    # macOS
    elif sys.platform == "darwin":
        return _pick_default_window_macos()
    # Linux
    else:
        return _pick_default_window_linux()


def _pick_default_window_windows() -> str | None:
    """Windows-specific window enumeration."""
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

        titles = []

        def callback(hwnd, lParam):
            if IsWindowVisible(hwnd):
                length = GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    if title:
                        titles.append(title)
            return True

        EnumWindows(EnumWindowsProc(callback), 0)
        preferred = [
            t
            for t in titles
            if any(
                k in t.lower()
                for k in ["code", "powershell", "terminal", "chrome", "notepad"]
            )
        ]
        if preferred:
            return preferred[0]
        return titles[0] if titles else None
    except Exception as e:
        print(f"! Could not enumerate windows: {e}")
        return None


def _pick_default_window_macos() -> str | None:
    """macOS-specific window enumeration."""
    try:
        import Quartz

        window_info_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
        )
        titles = []
        for info in window_info_list or []:
            owner = info.get(Quartz.kCGWindowOwnerName, "") or ""
            name = info.get(Quartz.kCGWindowName, "") or ""
            title = (f"{owner} - {name}" if owner and name else owner or name).strip()
            if title:
                titles.append(title)

        preferred = [
            t
            for t in titles
            if any(
                k in t.lower()
                for k in ["code", "terminal", "chrome", "safari", "edge", "kingdom"]
            )
        ]
        if preferred:
            return preferred[0]
        return titles[0] if titles else None
    except Exception as e:
        print(f"! Could not enumerate windows: {e}")
        return None


def _pick_default_window_linux() -> str | None:
    """Linux-specific window enumeration."""
    try:
        import subprocess
        import shutil

        if shutil.which("wmctrl"):
            proc = subprocess.run(
                ["wmctrl", "-l"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                titles = []
                for line in proc.stdout.splitlines():
                    parts = line.split(None, 3)
                    if len(parts) >= 4:
                        title = parts[3].strip()
                        if title:
                            titles.append(title)
                preferred = [
                    t
                    for t in titles
                    if any(
                        k in t.lower()
                        for k in ["code", "terminal", "chrome", "firefox"]
                    )
                ]
                if preferred:
                    return preferred[0]
                return titles[0] if titles else None
        return None
    except Exception as e:
        print(f"! Could not enumerate windows: {e}")
        return None


if not user_title:
    auto_title = _pick_default_window()
    if auto_title:
        print(f"No window title provided. Auto-selected: '{auto_title}'")
        user_title = auto_title
    else:
        print("No window title provided and no windows detected. Exiting.")
        sys.exit(1)

from src.modules.screen_input.screen_capture import take_screenshot

# Prepare output dir and timestamp
output_dir = os.path.join(repo_root, "src", "modules", "test")
os.makedirs(output_dir, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

saved_paths = []

print(f"\nTarget window: {user_title}")

"""Try background-capable capture first (auto/printwindow), then full."""

# Auto/printwindow attempt first (Windows background-capable if pywin32 installed)
try:
    path_auto = take_screenshot(
        tid=f"{stamp}_auto",
        output_dir=output_dir,
        window_title=user_title,
        method="auto",
        focus_window=False,
    )
    saved_paths.append(("auto", path_auto))
    print(f"✅ Saved (auto): {path_auto}")
except Exception as e:
    print(f"❌ Auto method failed: {e}")

# Full screen as fallback
try:
    path_full = take_screenshot(tid=f"{stamp}_full", output_dir=output_dir)
    saved_paths.append(("fullscreen", path_full))
    print(f"✅ Saved (fullscreen fallback): {path_full}")
except Exception as e:
    print(f"❌ Fullscreen fallback failed: {e}")

print("\nSummary:")
for kind, p in saved_paths:
    print(f"  - {kind}: {p}")

print("\nDone.")
