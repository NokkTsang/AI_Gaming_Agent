# Running the Simple Agent

## Quick Start

1. **Ensure prerequisites**:

   - Python 3.12+ with virtual environment activated
   - `.env` file with `OPENAI_API_KEY`
   - All dependencies installed: `pip install -r requirements.txt`

2. **Run the agent**:

   ```bash
   cd /Users/oscarzhang/Desktop/AI_Gaming_Agent
   python -m src.modules.main
   ```

3. **What it does**:
   - Takes a screenshot every step
   - Sends it to GPT-4o-mini for analysis
   - Plans the next action (click, type, hotkey, etc.)
   - Executes the action via pyautogui
   - Repeats for up to 10 steps or until task is complete

## Default Task

The default task is: **"Open a web browser, go to Google, and search for 'openai'"**

## Customize the Task

Edit `src/modules/main.py` line 131:

```python
task = "Your custom task here"
```

## How It Works

```
┌─────────────────────────────────────────────────┐
│  1. Screen Input (screen_capture.py)           │
│     └─> Takes screenshot of current screen     │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  2. Information Gathering (info_gather.py)      │
│     └─> Sends screenshot to GPT-4o-mini         │
│         Returns: "I see a desktop with..."       │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  3. Action Planning (planner.py)                │
│     └─> Given task + observation + history      │
│         Returns: {"action_type": "click", ...}   │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  4. UI Automation (executor.py)                 │
│     └─> Executes action via pyautogui          │
│         (clicks, types, scrolls, etc.)          │
└─────────────────┬───────────────────────────────┘
                  │
                  └──> Repeat (max 10 steps)
```

## Action Types Supported

- `click(start_box=[x, y])` - Click at normalized position
- `type(content="text")` - Type text
- `hotkey(key="ctrl c")` - Press hotkey combo
- `scroll(direction="up/down")` - Scroll screen
- `drag(start_box, end_box)` - Drag between points
- `wait()` - Wait 5 seconds
- `finished(content="summary")` - Mark complete

## Coordinates

All coordinates are **normalized** [0, 1]:

- `[0, 0]` = top-left
- `[0.5, 0.5]` = center
- `[1, 1]` = bottom-right

## Troubleshooting

### "Missing OPENAI_API_KEY"

- Create `.env` file: `echo 'OPENAI_API_KEY=sk-...' > .env`

### "No module named 'modules'"

- Run from project root: `cd /Users/oscarzhang/Desktop/AI_Gaming_Agent`
- Use: `python -m src.modules.main`

### Agent clicks wrong positions

- Coordinates are normalized [0, 1]
- The LLM must estimate positions from the screenshot description
- For better accuracy, you could add object detection (like SOM) later

### macOS permissions

- Grant "Screen Recording" and "Accessibility" permissions to Terminal/Python

## Next Steps

To extend this minimal agent:

1. **Add Memory**: Store successful action sequences
2. **Add Self-Reflection**: Detect when actions fail and retry
3. **Add Skill Curation**: Save reusable action sequences
4. **Better Object Detection**: Use SOM/GroundingDINO for precise coordinates
5. **Task Inference**: Break down complex tasks into subtasks
