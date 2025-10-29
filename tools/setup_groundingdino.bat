@echo off
REM Setup script for GroundingDINO object detection (Windows)

echo Setting up GroundingDINO...

REM Get the parent directory (project root)
cd /d "%~dp0.."

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: .venv not found. Please create virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    exit /b 1
)

REM 1. Install GroundingDINO
echo Installing groundingdino-py...
pip install groundingdino-py

REM 2. Create cache directory
echo Creating cache directory...
if not exist "cache" mkdir cache

REM 3. Download model files (SwinB version - better accuracy)
echo Downloading model weights (this may take a few minutes)...
cd cache

REM Download SwinB config
echo   Downloading config file...
curl -L -o GroundingDINO_SwinB_cfg.py https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py

REM Download SwinB checkpoint
echo   Downloading model checkpoint (~600MB)...
curl -L -o groundingdino_swinb_cogcoor.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

cd ..

echo.
echo GroundingDINO setup complete!
echo.
echo Model files installed in .\cache\
dir cache

echo.
echo Your agent can now use precise object detection!
echo Try running your Kingdom Rush task again - the agent will now be able to:
echo   - Detect the red flag icon precisely
echo   - Find tower building sites
echo   - Identify enemy units
echo   - Click any visual element accurately
