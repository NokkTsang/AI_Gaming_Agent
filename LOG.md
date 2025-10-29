### 29/10/2025

- **Maze Navigation Intelligence**: Added directional awareness for maze/navigation games
  - Vision prompt now calculates direction (UP/DOWN/LEFT/RIGHT) by comparing player position vs goal position
  - Vision detects if player moved successfully or hit a wall
  - Planner follows vision's directional guidance instead of random movements
  - Added backtracking strategy: if hitting wall, try different direction
  - Fixes issue where agent took random directions without spatial understanding (+150 tokens)
- **Performance Optimizations**: Reduced loop time from 50s to 5-10s per planning cycle
  - **Action Batching**: Agent can now output 3-5 actions in one planning cycle
    - Planner supports multi-action format: `{"actions": [...]}`
    - Main loop executes all actions in sequence with 0.1s delays
    - Reduces iterations by 3-5x (e.g., 5 arrow presses = 1 iteration instead of 5)
    - Token overhead: +75 tokens (+40 batching, +35 arrow key examples)
  - **Screen Change Detection**: Skips expensive vision analysis when screen unchanged
    - Pixel diff check with dynamic thresholds (0.1% keyboard, 0.2% clicks)
    - Reuses previous observation if screen unchanged, saves ~40s per cached iteration
    - Clear logging: "Screen unchanged [threshold=0.1%], reusing observation [saved ~40s]"
  - **Bug Fixes**: Fixed 4 KeyError bugs where code accessed `action_dict['action_type']` without checking for action sequences
- **Arrow Key Support**: Fixed critical bug where agent typed "up\n" instead of pressing arrow keys
  - Added arrow key examples to planner prompt: `hotkey("up")`, `hotkey("down")`, etc.
  - Added "Arrow keys: 'up', 'down', 'left', 'right'" to hotkey action description
- **Token Tracking**: Created `analyze_tokens.py` script to parse logs and calculate costs at GPT-4.1 rates

- **_To-do_**:
  - Added conversation with the agent at first moment to make clear about the task
  - The agent does not really know the direction of the maze, would it be possible for allowing the agent to write code functions to apply the searching algorithms like A\*?
  - Always keep the agent being general. If it can play the [maze game](https://www.mysteinbach.ca/game-zone/1507/maze/), test if it still can search for terms using a broswer
  - Is it better to include the cursor in the screenshot? Oscar will take care of it using the MacOS
  - Should the agent include extra modules?

### 25/10/2025

- **Multi-monitor support**: Agent now works correctly with multiple monitors
  - Fullscreen mode: Select specific monitor (0=all, 1=primary, 2=secondary, etc.)
  - Window mode: Auto-detects which monitor contains the window, captures only that monitor
  - Fixed coordinate mapping with monitor offsets for accurate clicks
- **Window capture optimization**:
  - Captures only window content (not full screen) for better resolution and token efficiency
  - Fixed macOS window capture to use `CGRectNull` (was capturing all screens with `CGRectInfinite`)
  - Window images maintain higher effective resolution when sent to LLM
- **Interactive launcher** (`run_agent.sh`):
  - Two-level menus: mode selection → monitor/window selection
  - Dynamic monitor detection with resolution and position display
  - "Back" option (`b`) at any prompt to return to main menu
- **Improved logging**:
  - Removed verbose window list spam (was printing 3 times)
  - Cleaner output with single match confirmation
  - Better error messages for window capture failures

### 23/10/2025

- Unified window capture across Windows/macOS/Linux in a single API:
  - Windows: Win32 PrintWindow for true background window capture; automatic fullscreen fallback if unavailable
  - macOS: Quartz CGWindowListCreateImage for a specific window; automatic fullscreen fallback if unavailable
  - Linux: Prefer `xwd` for window dump; fallback to `xwininfo` geometry + mss; automatic fullscreen fallback when tools are missing
- Added fuzzy window title matching (substring first, then fuzzy) and window list printing for easier selection
- Added a new test case `src/modules/test/test_window_capture.py`
- Agent now supports prioritized window capture when a title is configured via environment variable:
  - `WINDOW_TITLE` (e.g., `$env:WINDOW_TITLE = "chrome"` on PowerShell)
  - If not set, behavior remains fullscreen-only
- Requirements updated for macOS:
  - Added `pyobjc-framework-Quartz; sys_platform == "darwin"` for macOS window capture
- Linux system tools (optional, not in pip requirements): `wmctrl`, `xdotool`, `xwininfo`, `xwd`
- Test of window capturing only done on WindowsOS

### 22/10/2025

- Added comprehensive LLM logging: all prompts, responses, and token usage are now logged
- Added automatic log file saving to `src/modules/memory/task_log/` with timestamps
- All terminal output is captured and saved for debugging and analysis
- Added TaskLogger class with TeeOutput for simultaneous console and file logging
- **Hybrid Vision + Object Detection System**: Vision model can now request object detection for precise coordinates
  - Vision LLM handles strategic decisions and context understanding
  - GroundingDINO provides precise visual element localization when requested
  - Automatic fallback: works without GroundingDINO (vision-only mode)
  - Vision model outputs `REQUEST_DETECTION: <object>` for visual elements without text
  - System annotates images with bounding boxes and re-analyzes for precise coordinates
- **Removed YOLO**: Simplified to GroundingDINO only (better for games, zero-shot detection)
- **Performance optimizations**: Image resizing (1024px max), simplified prompts, removed grid system

### 15/10/2025

- Focus on image contextual memory (image to text description)
- Memory store screenshots (older screenshots more blur, and store limited number of screenshots, buffer zone)
- Screenshots frequency
- Memory limit, delete old memory?
- Test if the saved succesfull memory skills can be reused
- Compare strong model without vision support and weak model with vison support (ocr, yolo...)
- Store reflection into JSON everytime?
- Screenshot specific window?

### 8/10/2025

- Add EasyOCR for character detection
- Add YOLO for object detection (later replaced with GroundingDINO)
- Adjust the screen capture to native resolution capture instead of logical resolution capture

### 5/10/2025

- Added action_planning module
- Added self_reflection module
- Added skill_curation module
- Added task_inference module
- Added test module
- Performance not optimal, should optimize it
  - llm cannot compare contextual semantic screenshots, that is, it can not directly notice the change in screenshots
  - llm can struggles on one page for a long time
  - does the atomic actions work on multiple operating systems (MacOS, Windows, Linux)?

### 2/10/2025

- Added ui_automation module
- Added screen_input module
- Added information_gathering module
- Added memory module

### 27/9/2025

- Created modules folder
- Designed developing process

### 23/9/2025

- Added demo and reference from ByteDance Tars

### 18/9/2025

- Initialized project
