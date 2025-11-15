```markdown
### 13/11/2025
- **Three-Stage Vision Pipeline Architecture**:
  
  **Stage 1 - VLM (Vision-Language Model): High-Level Analysis**
  - **Input**: Screenshot
  - **Task**: Analyze overall content to understand game rules, objectives, and visual semantics
  - **Output**: 
    - Game understanding (e.g., "This is a maze game where red=player, green=exit, black=walls, white=paths")
    - Specific detection prompt for Stage 2 (e.g., "Detect positions of: player (red square), exit (green square), walls (black), paths (white)")
  
  **Stage 2 - GroundingDINO: Precise Object Localization**
  - **Input**: 
    - Screenshot (same as Stage 1)
    - Detection prompt from Stage 1
  - **Task**: Locate specific visual elements with precise bounding boxes
  - **Output**: 
    - Structured spatial data (e.g., "Player at [0.3, 0.5], Exit at [0.8, 0.9]")
    - Surrounding context (e.g., "Walls detected at [0.2, 0.5], [0.4, 0.5], Path clear at [0.3, 0.6]")
  
  **Stage 3 - LLM (Reasoning): Decision Making**
  - **Input**: Structured information from Stage 2
  - **Task**: Strategic reasoning and action planning
  - **Output**: 
    - Decision (e.g., "Move DOWN - path is clear, brings player closer to exit")
    - Executable instruction (e.g., `{"action_type": "hotkey", "action_inputs": {"key": "down"}}`)

- **Concrete Maze Example**:
  
  **Stage 1 (VLM) analyzes the screenshot:**
  - "Maze game detected. Red square = player, green square = exit goal, black pixels = impassable walls, white pixels = walkable paths. Player can move in 4 directions (arrow keys)."
  - Outputs prompt for Stage 2: "Detect: player (red), exit (green), and classify surrounding 8 cells (up, down, left, right, diagonals) as wall (black) or path (white)"
  
  **Stage 2 (GroundingDINO) detects objects:**
  - Player position: [0.25, 0.40]
  - Exit position: [0.75, 0.85]
  - Surrounding cells analysis:
    - UP [0.25, 0.35]: BLACK (wall)
    - DOWN [0.25, 0.45]: WHITE (path)
    - LEFT [0.20, 0.40]: BLACK (wall)
    - RIGHT [0.30, 0.40]: WHITE (path)
  
  **Stage 3 (LLM) makes decision:**
  - Analysis: "Player at [0.25, 0.40], exit at [0.75, 0.85]. Exit is DOWN and RIGHT. UP blocked by wall, LEFT blocked by wall. DOWN and RIGHT are both open paths."
  - Decision: "Move DOWN first (gets closer to exit's Y-coordinate)"
  - Instruction: `{"action_type": "hotkey", "action_inputs": {"key": "down"}}`

- **Implementation TODOs**:
  1. ✅ Stage 1 VLM pipeline: Integrate vision analysis with task understanding (currently uses `analyze_screenshot_with_detection`)
  2. ⏳ Stage 2 GroundingDINO: Install and configure for precise spatial detection (currently returns empty if not installed)
  3. ⏳ Stage 3 LLM reasoning: Enhance planner to consume structured spatial data instead of raw text observations
  4. ⏳ Test end-to-end with maze game and validate each stage's output quality

### 5/11/2025

- **Game-TARS Integration Complete**: Implemented all 6 improvements from Game-TARS paper
  - **Two-tier memory system**: 80 full steps + 2400 compressed summaries (was 20 total)
    - Context memory: Full observations + thoughts + actions (recent 80 steps)
    - Summary memory: Compressed thoughts only (up to 2400 steps)
    - Automatic compression with sliding window for long games
  - **Sparse thinking**: Complexity detector determines when deep reasoning needed
    - 6 heuristics: first action, failure, new elements, decision keywords, repetition, continuation
    - LLM fallback for semantic complexity detection
    - Reactive mode (action-only): 2-5s vs Deep mode (full reasoning): 45s
    - Expected 70-85% reactive rate → 3-4x speedup
  - **Task clarification with instruction following**: Structured instructions prevent behavioral inertia
    - Analyzes task for ambiguities, asks user clarification questions
    - Supports both quick selection (a/b/c) and free-text detailed answers
    - Generates structured instruction: goal, action space, constraints, success criteria, failure conditions
    - **Critical fix**: Structured instruction now INJECTED into action planner system prompt
    - Planner sees task-specific rules (HIGHEST PRIORITY) before generic GUI automation rules
    - Makes agent context-aware and general: understands keyboard games vs GUI automation vs web forms
  - **Completion detection**: Validates task completion before calling finished()
    - Checks against explicit success criteria from clarification
    - Confidence scoring with user confirmation if <0.8
  - **Enhanced stuck detection**: Semantic detection with LLM + recovery actions
    - 4 heuristics: identical actions, position looping, failure repetition, no progress
    - LLM semantic stuck detection for complex patterns
    - Recovery suggestions: wait, esc, enter, click_different
  - **Reactive planning**: Fast action-only planning mode (no thought field)
    - Separate system prompt optimized for speed
    - Trust recent action history for continuation
    - 200 tokens vs 800 tokens (deep mode)
- **Performance**: Before: 45-60s/action, 20 steps | After: ~12s/action, 2480 steps | Speedup: 3-4x faster, 124x memory
- **Bug fixes**: Agent now understands task-specific instructions (e.g., "arrow keys only" for maze games)
  - Previous issue: Generic GUI automation prompt overrode user instructions (tried clicking/Tab in keyboard games)
  - Fix: Task clarification creates structured instruction → injected into planner system prompt
  - Agent now reads YOUR specific rules (action space, constraints) instead of blind automation
  - **_To-do_**:
    - If the agent is stucked, it can ask human for help. After human helped, the agent should memorize the method to solve the stucked problem. And next time it should solve the similar problem automatically without asking human again. If not, it can still ask. Change JSON to TOON? The one mentioned by Nokk. Learn from Game-TARS.

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
```
