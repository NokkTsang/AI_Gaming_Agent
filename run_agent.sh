#!/bin/bash
# Interactive agent launcher with window selection

cd "$(dirname "$0")"

echo "========================================"
echo "   AI Gaming Agent Launcher"
echo "========================================"
echo ""
echo "Select capture mode:"
echo ""
echo "  1) Fullscreen"
echo "     - Captures full screen of selected monitor(s)"
echo ""
echo "  2) Specific Window"
echo "     - Captures ONLY the window content (no background)"
echo "     - Higher resolution for better accuracy"
echo "     - More focused analysis"
echo ""
read -p "Your choice [1-2]: " choice
echo ""

case $choice in
    1)
        # Fullscreen mode - Let user choose monitor
        echo "✓ Fullscreen mode selected"
        echo ""
        echo "Detecting available monitors..."
        
        # Activate venv if it exists
        if [ -d ".venv" ]; then
            source .venv/bin/activate
        fi
        
        # Get monitor information
        python3 -c "
import mss
import sys

try:
    with mss.mss() as sct:
        monitors = sct.monitors
        print('Available monitors:')
        print()
        print('  0) All Monitors Combined')
        print(f'     - Total area: {monitors[0][\"width\"]}x{monitors[0][\"height\"]}')
        print(f'     - Note: May have black areas due to different resolutions')
        print()
        for i in range(1, len(monitors)):
            m = monitors[i]
            print(f'  {i}) Monitor {i}')
            print(f'     - Resolution: {m[\"width\"]}x{m[\"height\"]}')
            print(f'     - Position: ({m[\"left\"]}, {m[\"top\"]})')
            print()
except Exception as e:
    print(f'Error detecting monitors: {e}')
    sys.exit(1)
"
        
        echo ""
        read -p "Select monitor [0-N] (or 'b' to go back): " monitor_choice
        
        if [[ "$monitor_choice" == "b" ]] || [[ "$monitor_choice" == "B" ]]; then
            echo ""
            exec "$0"  # Restart the script
        fi
        
        if ! [[ "$monitor_choice" =~ ^[0-9]+$ ]]; then
            echo "Error: Invalid monitor number"
            exit 1
        fi
        
        echo ""
        read -p "Enter your task (or 'b' to go back): " task
        
        if [[ "$task" == "b" ]] || [[ "$task" == "B" ]]; then
            echo ""
            exec "$0"  # Restart the script
        fi
        if [ -z "$task" ]; then
            echo "Error: Task cannot be empty"
            exit 1
        fi
        echo ""
        echo "Starting agent with monitor $monitor_choice..."
        echo "Task: $task"
        echo ""
        MONITOR_INDEX="$monitor_choice" python3 -m src.modules.main "$task"
        ;;
    
    2)
        # Specific window mode (captures full screen where window is located)
        echo "✓ Detecting windows..."
        echo ""
        echo "Scanning for open windows..."
        echo "========================================"
        
        # Activate venv if it exists
        if [ -d ".venv" ]; then
            source .venv/bin/activate
        fi
        
        # Enumerate and filter available windows
        python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    from src.modules.screen_input.screen_capture import _enum_windows_macos, _enum_windows_windows, _enum_windows_linux
    
    if sys.platform.startswith('win'):
        windows = _enum_windows_windows()
    elif sys.platform == 'darwin':
        windows = _enum_windows_macos()
    else:
        windows = _enum_windows_linux()
    
    if windows:
        # Filter out system/service windows for cleaner list
        filtered = []
        skip_keywords = ['Window Server', 'CursorUIViewService', 'Control Center', 'Finder', 'Dock', 
                         'SystemUIServer', 'loginwindow', 'Spotlight', 'Notification Center', 'Accessibility']
        
        for wid, title in windows:
            if not any(skip in title for skip in skip_keywords):
                filtered.append(title)
        
        # Remove duplicates and sort
        unique_windows = sorted(set(filtered))
        
        if unique_windows:
            print('Available application windows:')
            print()
            for i, title in enumerate(unique_windows, 1):
                print(f'  {i}. {title}')
        else:
            print('No application windows found (all filtered out).')
    else:
        print('No windows detected by the system.')
        
except Exception as e:
    print(f'Error detecting windows: {e}')
    print()
    print('Tip: Make sure you have installed dependencies:')
    print('  pip install pyobjc-framework-Quartz')
"
        
        # Store window list in a temporary file
        python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    from src.modules.screen_input.screen_capture import _enum_windows_macos, _enum_windows_windows, _enum_windows_linux
    
    if sys.platform.startswith('win'):
        windows = _enum_windows_windows()
    elif sys.platform == 'darwin':
        windows = _enum_windows_macos()
    else:
        windows = _enum_windows_linux()
    
    if windows:
        filtered = []
        skip_keywords = ['Window Server', 'CursorUIViewService', 'Control Center', 'Finder', 'Dock', 
                         'SystemUIServer', 'loginwindow', 'Spotlight', 'Notification Center', 'Accessibility']
        
        for wid, title in windows:
            if not any(skip in title for skip in skip_keywords):
                filtered.append(title)
        
        unique_windows = sorted(set(filtered))
        
        # Write to temp file
        with open('/tmp/agent_windows.txt', 'w') as f:
            for title in unique_windows:
                f.write(title + '\n')
except:
    pass
" 2>/dev/null
        
        echo ""
        echo "========================================"
        echo ""
        echo "Tips:"
        echo "  - Enter the NUMBER (e.g., '3') to select that window"
        echo "  - Or type the exact window title"
        echo "  - Or type a partial match (e.g., 'Edge' for 'Microsoft Edge')"
        echo "  - Type 'b' to go back to main menu"
        echo ""
        read -p "Enter window number or title (or 'b' to go back): " window_input
        
        if [[ "$window_input" == "b" ]] || [[ "$window_input" == "B" ]]; then
            echo ""
            exec "$0"  # Restart the script
        fi
        
        if [ -z "$window_input" ]; then
            echo "Error: Input cannot be empty"
            exit 1
        fi
        
        # Check if input is a number
        if [[ "$window_input" =~ ^[0-9]+$ ]]; then
            # It's a number - get the window title from the list
            window=$(sed -n "${window_input}p" /tmp/agent_windows.txt)
            if [ -z "$window" ]; then
                echo "Error: Invalid window number"
                exit 1
            fi
            echo "Selected: $window"
        else
            # It's a window title
            window="$window_input"
        fi
        
        echo ""
        read -p "Enter your task (or 'b' to go back): " task
        
        if [[ "$task" == "b" ]] || [[ "$task" == "B" ]]; then
            echo ""
            exec "$0"  # Restart the script
        fi
        
        if [ -z "$task" ]; then
            echo "Error: Task cannot be empty"
            exit 1
        fi
        echo ""
        echo "Starting agent..."
        echo "Window: $window"
        echo "Task: $task"
        echo ""
        WINDOW_TITLE="$window" python3 -m src.modules.main "$task"
        
        # Cleanup
        rm -f /tmp/agent_windows.txt
        ;;
    
    *)
        echo "Invalid choice. Please run again and select 1 or 2."
        exit 1
        ;;
esac

