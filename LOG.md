### 19/11/2025
- **Three-Agent Architecture Implementation Complete**:
  
  **Agent 1: VLM Agent** (`src/modules/agents/vlm_agent.py`)
  - Analyzes screenshots
  - Produces dual outputs in a single API call:
    - `detection_prompt`: Object detection instructions for GroundingDINO Agent
    - `game_context`: Game understanding for LLM Agent (rules, objectives, semantics)
  - Test file: `src/modules/test/test_vlm_agent.py`
  - ✅ **Test Status**: All tests passing, correct outputs verified
  - Example: Maze game → detection_prompt: "player, walls, goal", game_context: "Navigate red square to green goal, avoid black walls"
  
  **Agent 2: GroundingDINO Agent** (`src/modules/agents/dino_agent.py`)
  - Zero-shot object detection based on text prompts from VLM Agent
  - Uses existing `object_detector.py` infrastructure
  - Outputs structured spatial data: examples: bounding boxes, pixel coordinates, confidence scores
  - Test file: `src/modules/test/test_dino_agent.py`
  - ⚠️ **Test Status**: Structure tests passing, but GroundingDINO detection not fully tested
    - Agent gracefully falls back to empty detections when model unavailable
    - Mock detector available for testing agent logic without model
  
  **Agent 3: LLM Agent** (`src/modules/agents/llm_agent.py`)
  - Decision-making agent
  - Inputs: game_context (from VLM) + spatial_data (from DINO)
  - Outputs: action (JSON), reasoning (text, for debug), confidence (0-1, for debug)
  - Test file: `src/modules/test/test_llm_agent.py`
  - ✅ **Test Status**: All tests passing, correct action decisions verified
  - Example output: `{"action_type": "hotkey", "action_inputs": {"key": "down"}}` with reasoning

- **Test Infrastructure**:
  - All three agents have comprehensive test suites
  - Interactive manual tests allow custom inputs (screenshot paths, prompts, JSON data)
  - Unit tests validate output structure and error handling
  - Mock data provided: `src/modules/test/mock_agent3_input.json` (sample maze scenario)
  
- **TODO**:
  1. ⏳ **Complete GroundingDINO Testing**: Download model checkpoint file and verify actual object detection works correctly with real game screenshots
  2. ⏳ **Integrate 3-Agent Architecture into Main**: Replace current `analyze_screenshot_with_detection()` + `ActionPlanner` pipeline with the new 3-agent workflow
     - Keep main.py's advanced features: memory, reflection, skills, sparse thinking, task clarification
     - Replace vision+planning with: VLM Agent → DINO Agent → LLM Agent
     - Maintain compatibility with existing features (completion detection, stuck detection, etc.)

### 13/11/2025
- **Three-Agent Architecture** (VLM → DINO → LLM):
  
  **Agent 1: VLM Agent (Vision-Language Model)**
  - **Input**: Screenshot
  - **Outputs to**:
    - **→ GroundingDINO Agent**: Detection prompt specifying what objects to locate
    - **→ LLM Agent**: Game understanding context (rules, objectives, visual semantics)
  - **Example Output**:
    - To DINO: "Detect: player (red square), exit (green square), surrounding cells (black=wall, white=path)"
    - To LLM: "Maze game. Red=player, green=exit, black=walls, white=paths. Move with arrow keys."
  
  **Agent 2: GroundingDINO Agent (Spatial Localization)**
  - **Input**: 
    - Screenshot (same as Agent 1)
    - Detection prompt from VLM Agent
  - **Output to LLM Agent**: 
    - Structured spatial data with precise coordinates
    - Example: "Player: [0.25, 0.40], Exit: [0.75, 0.85], Walls: {UP, LEFT}, Paths: {DOWN, RIGHT}"
  
  **Agent 3: LLM Agent (Decision Making)**
  - **Inputs from**:
    - **VLM Agent**: Game rules and context understanding
    - **GroundingDINO Agent**: Precise spatial positions and relationships
  - **Output**: Executable action instruction
  - **Example**: `{"action_type": "hotkey", "action_inputs": {"key": "down"}}` based on reasoning: "Exit is DOWN+RIGHT. Both paths clear. Move DOWN first."

- **Concrete Maze Example**:
  
  **Agent 1 (VLM) processes screenshot:**
  - To GroundingDINO: "Detect player (red square), exit (green square), and classify 4 adjacent cells as wall (black) or path (white)"
  - To LLM: "Maze game rules: Red player must reach green exit. Black cells are walls (blocked). White cells are paths (walkable). Use arrow keys to move."
  
  **Agent 2 (GroundingDINO) returns spatial data:**
  - To LLM: 
    ```
    Player: [0.25, 0.40]
    Exit: [0.75, 0.85]
    Adjacent cells:
      UP [0.25, 0.35]: WALL (black)
      DOWN [0.25, 0.45]: PATH (white)
      LEFT [0.20, 0.40]: WALL (black)
      RIGHT [0.30, 0.40]: PATH (white)
    ```
  
  **Agent 3 (LLM) decides action:**
  - Receives game rules from VLM + spatial data from DINO
  - Reasoning: "Player at [0.25, 0.40], exit at [0.75, 0.85]. Exit direction: DOWN and RIGHT. UP/LEFT blocked by walls. DOWN and RIGHT are open."
  - Decision: "Move DOWN (closer to exit Y-coordinate, path is clear)"
  - Output: `{"action_type": "hotkey", "action_inputs": {"key": "down"}}`

- **Implementation TODOs**:
  1. ⏳ Create VLM Agent class with dual output (detection prompt + game context)
  2. ⏳ Create GroundingDINO Agent class (currently just helper function in `object_detector.py`)
  3. ⏳ Create LLM Agent class that consumes both VLM context and DINO spatial data
  4. ⏳ Implement agent coordinator/orchestrator to manage the data flow between 3 agents
  5. ⏳ Test end-to-end with maze game and validate each agent's output quality

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