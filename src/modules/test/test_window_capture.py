"""
Window capture test: Lists available windows and
allows user to select one to capture,
or captures fullscreen if no selection is made.

Run from project root:
    python -m src.modules.test.test_window_capture
"""

import os
import sys
from datetime import datetime
from src.modules.screen_input.screen_capture import take_screenshot

# Ensure repo root is on sys.path so `modules.*` imports work when not using -m
repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("Window Capture Test")
print("=" * 80)


def _enumerate_windows() -> list[tuple[int, str]]:
    """Enumerate all visible windows and return list of (id, title) tuples."""
    # Windows
    if sys.platform.startswith("win"):
        return _enumerate_windows_windows()
    # macOS
    elif sys.platform == "darwin":
        return _enumerate_windows_macos()
    # Linux
    else:
        return _enumerate_windows_linux()


def _enumerate_windows_windows() -> list[tuple[int, str]]:
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

        results = []

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
    except Exception as e:
        print(f"! Could not enumerate windows: {e}")
        return []


def _enumerate_windows_macos() -> list[tuple[int, str]]:
    """macOS-specific window enumeration."""
    try:
        import Quartz

        window_info_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
        )
        results = []
        for info in window_info_list or []:
            wid = info.get(Quartz.kCGWindowNumber)
            owner = info.get(Quartz.kCGWindowOwnerName, "") or ""
            name = info.get(Quartz.kCGWindowName, "") or ""
            title = (
                f"{owner} - {name}" if owner and name else owner or name
            ).strip()
            if wid and title:
                results.append((int(wid), title))
        return results
    except Exception as e:
        print(f"! Could not enumerate windows: {e}")
        return []


def _enumerate_windows_linux() -> list[tuple[str, str]]:
    """Linux-specific window enumeration."""
    try:
        import subprocess
        import shutil

        results = []
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
                        wid_hex = parts[0]
                        title = parts[3].strip()
                        if title:
                            results.append((wid_hex, title))
        return results
    except Exception as e:
        print(f"! Could not enumerate windows: {e}")
        return []


# List available windows
print("\n🪟 Available windows:")
windows = _enumerate_windows()

if windows:
    for idx, (wid, title) in enumerate(windows[:20], 1):
        print(f"   {idx}. {title}")
else:
    print("   No windows detected")

# Prompt user for selection
print("\n📋 Select a window to capture:")
if windows:
    print(f"   - Enter window number (1-{len(windows[:20])})")
print("   - Press ENTER for full screen capture")

user_input = input("\nYour choice: ").strip()

# Prepare output dir and timestamp
output_dir = os.path.join(repo_root, "src", "modules", "test")
os.makedirs(output_dir, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

saved_paths = []

# Determine capture target
if user_input and user_input.isdigit() and windows:
    idx = int(user_input) - 1
    if 0 <= idx < len(windows[:20]):
        _, window_title = windows[idx]
        print(f"\n📸 Capturing window: {window_title}")
        
        try:
            path_window, _ = take_screenshot(
                tid=f"{stamp}_window",
                output_dir=output_dir,
                window_title=window_title,
                method="auto",
                focus_window=True
            )
            saved_paths.append(("window", path_window))
            print(f"✅ Saved: {path_window}")
        except Exception as e:
            print(f"❌ Window capture failed: {e}")
    else:
        print(f"❌ Invalid selection: {user_input}")
else:
    # Full screen capture
    print("\n📸 Capturing full screen...")
    try:
        path_full, _ = take_screenshot(
            tid=f"{stamp}_fullscreen",
            output_dir=output_dir
        )
        saved_paths.append(("fullscreen", path_full))
        print(f"✅ Saved: {path_full}")
    except Exception as e:
        print(f"❌ Fullscreen capture failed: {e}")

print("\nSummary:")
if saved_paths:
    for kind, p in saved_paths:
        print(f"  - {kind}: {p}")
else:
    print("  No screenshots captured")

print("\nDone.")
