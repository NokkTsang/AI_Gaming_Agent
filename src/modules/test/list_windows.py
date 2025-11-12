"""
List all open window titles (cross-platform best effort).

Run from project root:
  python -m src.modules.test.list_windows

On Windows: uses EnumWindows + GetWindowText
On macOS: uses Quartz window list (requires pyobjc-framework-Quartz)
On Linux: uses wmctrl (preferred) or xdotool
"""

import sys
import shutil
import subprocess


def list_windows_windows():
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
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
        return titles
    except Exception as e:
        print(f"! Windows enumeration failed: {e}")
        return []


def list_windows_macos():
    try:
        import Quartz
        info_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
        titles = []
        for info in info_list or []:
            owner = info.get(Quartz.kCGWindowOwnerName, '') or ''
            name = info.get(Quartz.kCGWindowName, '') or ''
            title = (f"{owner} - {name}" if owner and name else owner or name).strip()
            if title:
                titles.append(title)
        return titles
    except Exception as e:
        print(f"! macOS Quartz enumeration failed: {e}")
        return []


def list_windows_linux():
    titles = []
    try:
        if shutil.which('wmctrl'):
            p = subprocess.run(['wmctrl', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode == 0:
                for line in p.stdout.splitlines():
                    parts = line.split(None, 3)
                    if len(parts) >= 4:
                        titles.append(parts[3].strip())
                return titles
        if shutil.which('xdotool'):
            ids = subprocess.run(['xdotool', 'search', '--name', '.+'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if ids.returncode == 0:
                for wid in ids.stdout.split():
                    name = subprocess.run(['xdotool', 'getwindowname', wid], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if name.returncode == 0:
                        t = (name.stdout or '').strip()
                        if t:
                            titles.append(t)
        return titles
    except Exception as e:
        print(f"! Linux enumeration failed: {e}")
        return titles


def main():
    if sys.platform.startswith('win'):
        titles = list_windows_windows()
    elif sys.platform == 'darwin':
        titles = list_windows_macos()
    else:
        titles = list_windows_linux()

    print("[Window List]")
    if not titles:
        print("  (none)")
    else:
        for i, t in enumerate(titles, 1):
            print(f"  {i}. {t}")


if __name__ == '__main__':
    main()
