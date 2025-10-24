@echo off
REM Interactive agent launcher for Windows (Command Prompt)
cd /d "%~dp0"

echo ========================================
echo    AI Gaming Agent Launcher
echo ========================================
echo.
echo Select capture mode:
echo.
echo   1) Fullscreen
echo      - Captures full screen of selected monitor(s)
echo.
echo   2) Specific Window
echo      - Captures ONLY the window content (no background)
echo      - Higher resolution for better accuracy
echo      - More focused analysis
echo.
set /p choice="Your choice [1-2]: "
echo.

if "%choice%"=="1" goto fullscreen
if "%choice%"=="2" goto window
echo Invalid choice. Please run again and select 1 or 2.
exit /b 1

:fullscreen
echo ✓ Fullscreen mode selected
echo.
echo Detecting available monitors...

REM Activate venv if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Get monitor information
python -c "import mss; import sys; sct = mss.mss(); monitors = sct.monitors; print('Available monitors:\n'); print('  0) All Monitors Combined'); print(f'     - Total area: {monitors[0][\"width\"]}x{monitors[0][\"height\"]}'); print(f'     - Note: May have black areas due to different resolutions\n'); [print(f'  {i}) Monitor {i}\n     - Resolution: {monitors[i][\"width\"]}x{monitors[i][\"height\"]}\n     - Position: ({monitors[i][\"left\"]}, {monitors[i][\"top\"]})') for i in range(1, len(monitors))]"

echo.
set /p monitor_choice="Select monitor [0-N] (or 'b' to go back): "

if /i "%monitor_choice%"=="b" (
    echo.
    call "%~f0"
    exit /b 0
)

echo.
set /p task="Enter your task (or 'b' to go back): "

if /i "%task%"=="b" (
    echo.
    call "%~f0"
    exit /b 0
)

if "%task%"=="" (
    echo Error: Task cannot be empty
    exit /b 1
)

echo.
echo Starting agent with monitor %monitor_choice%...
echo Task: %task%
echo.
set MONITOR_INDEX=%monitor_choice%
python -m src.modules.main "%task%"
goto end

:window
echo ✓ Detecting windows...
echo.
echo Scanning for open windows...
echo ========================================

REM Activate venv if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Enumerate windows
python -c "import sys; sys.path.insert(0, '.'); from src.modules.screen_input.screen_capture import _enum_windows_windows; windows = _enum_windows_windows(); filtered = [title for wid, title in windows if not any(skip in title for skip in [''])]; unique_windows = sorted(set(filtered)); print('Available application windows:\n') if unique_windows else print('No windows detected.'); [print(f'  {i}. {title}') for i, title in enumerate(unique_windows, 1)]"

echo.
echo ========================================
echo.
echo Tips:
echo   - Enter the window title or partial match (e.g., 'Edge')
echo   - Type 'b' to go back to main menu
echo.
set /p window_input="Enter window title (or 'b' to go back): "

if /i "%window_input%"=="b" (
    echo.
    call "%~f0"
    exit /b 0
)

if "%window_input%"=="" (
    echo Error: Input cannot be empty
    exit /b 1
)

echo.
set /p task="Enter your task (or 'b' to go back): "

if /i "%task%"=="b" (
    echo.
    call "%~f0"
    exit /b 0
)

if "%task%"=="" (
    echo Error: Task cannot be empty
    exit /b 1
)

echo.
echo Starting agent...
echo Window: %window_input%
echo Task: %task%
echo.
set WINDOW_TITLE=%window_input%
python -m src.modules.main "%task%"

:end
