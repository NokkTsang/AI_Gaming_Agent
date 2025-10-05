# Architecture Documentation

## System Overview

The AI Gaming Agent is a memory-centric autonomous agent that learns reusable skills from successful action sequences. It uses a combination of vision LLMs for screen understanding, CodeAgent for action planning, and vector embeddings for efficient skill retrieval.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│                      (Task Description)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MAIN AGENT LOOP                              │
│                   (src/modules/main.py)                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Initialize → 2. Decompose → 3. Execute → 4. Reflect   │   │
│  │      ↑                                            │      │   │
│  │      └────────────── Learn & Curate ──────────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  PERCEPTION  │  │   REASONING  │  │    ACTION    │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Module Breakdown

### 1. Perception Layer

#### Screen Input

```
screen_input/screen_capture.py
├── take_screenshot()     → Captures screen to JPEG
└── get_fullscreen_region() → Gets monitor dimensions
```

#### Information Gathering

```
information_gathering/
├── info_gather.py
│   └── analyze_screenshot()  → GPT-4V vision analysis
└── object_detector.py
    └── detect_objects()      → [Placeholder] SOM/GroundingDINO
```

**Data Flow**:

```
Screen → Screenshot → Vision LLM → Text Observation
                 ↓
            Object Detector → Bounding Boxes [Future]
```

### 2. Memory System

#### Three-Tier Memory

```
memory/
├── short_term.py        (Session State)
│   ├── TaskState.initialize_task()
│   ├── TaskState.add_observation()
│   ├── TaskState.add_action()
│   └── TaskState.get_recent_context()
│
├── long_term.py         (Skill Database)
│   ├── SkillDatabase.add_skill()
│   ├── SkillDatabase.get_skill()
│   └── SkillDatabase.update_skill()
│
└── skill_retrieval.py   (Semantic Search)
    ├── EmbeddingRetriever.retrieve_relevant_skills()
    └── EmbeddingRetriever.rebuild_embeddings()
```

**Memory Architecture**:

```
┌─────────────────────────────────────────────┐
│         SHORT-TERM MEMORY                   │
│  (Current Task State - JSON)                │
│  ┌──────────────────────────────────────┐   │
│  │ - Task Goal                          │   │
│  │ - Subtasks (current index)           │   │
│  │ - Observation History (last 10)      │   │
│  │ - Action History (last 10)           │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    │
                    │ Success Detected
                    ▼
┌─────────────────────────────────────────────┐
│         SKILL CURATION                      │
│  (Extract & Generalize)                     │
│  ┌──────────────────────────────────────┐   │
│  │ Action Sequence → Python Code        │   │
│  │ Subtask Desc → Skill Name            │   │
│  │ Context → Documentation              │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         LONG-TERM MEMORY                    │
│  (Skills Database - JSON)                   │
│  ┌──────────────────────────────────────┐   │
│  │ Skill ID: "skill_0"                  │   │
│  │ Name: "open_web_browser"             │   │
│  │ Description: "Opens Chrome..."       │   │
│  │ Code: "def open_web_browser():..."   │   │
│  │ Success Count: 5                     │   │
│  │ Last Used: "2025-10-06..."           │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                    │
                    │ Embed with sentence-transformers
                    ▼
┌─────────────────────────────────────────────┐
│       SKILL RETRIEVAL (Vector DB)           │
│  (Embeddings Cache - .npy)                  │
│  ┌──────────────────────────────────────┐   │
│  │ skill_0 → [0.23, -0.45, 0.67, ...]   │   │
│  │ skill_1 → [-0.12, 0.89, -0.34, ...]  │   │
│  │ skill_2 → [0.56, 0.23, 0.91, ...]    │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  Query: "open browser"                      │
│  → Cosine Similarity → Top 5 Skills         │
└─────────────────────────────────────────────┘
```

### 3. Reasoning Layer

#### Task Inference

```
task_inference/task_breaker.py
├── decompose_task()           → Break task into subtasks
├── replan_after_failure()     → Adjust plan on failure
└── check_task_completion()    → Verify task complete
```

**Task Decomposition Flow**:

```
"Open browser and search for Python"
         │
         ▼
┌─────────────────────────────────┐
│  TaskBreaker.decompose_task()   │
└─────────────────────────────────┘
         │
         ▼
1. Open web browser application
2. Navigate to search engine
3. Type search query "Python"
4. Submit search
```

#### Self-Reflection

```
self_reflection/reflector.py
├── judge_action_success()     → Compare before/after
└── detect_stuck_state()       → Identify loops
```

**Reflection Logic**:

```
Before Action: "Desktop with icons visible"
Action: click_box([0.5, 0.9])
After Action: "Chrome browser window open"

Reflector Analysis:
  Goal: "Open web browser"
  Change: Desktop → Browser window
  Result: SUCCESS ✓
```

#### Skill Curation

```
skill_curation/skill_manager.py
├── should_save_as_skill()     → Heuristics for skill-worthiness
├── extract_skill()            → Convert sequence to code
└── refine_skill_with_llm()    → LLM-enhanced skill generation
```

**Skill Extraction Example**:

```python
Action Sequence:
1. click_box([0.5, 0.95])
2. wait(2.0)
3. hotkey("cmd t")

        ↓ extract_skill()

Generated Skill:
def open_browser_new_tab():
    """Opens browser and creates new tab."""
    click_box([0.5, 0.95])
    wait(2.0)
    hotkey("cmd t")
```

### 4. Action Planning & Execution

#### Action Planning

```
action_planning/planner.py
├── ActionPlanner.__init__()        → Initialize CodeAgent
├── load_skills()                   → Dynamic skill loading
└── plan_next_action()              → Execute with CodeAgent
```

**Planning Architecture**:

```
┌────────────────────────────────────────────┐
│         ACTION PLANNER                     │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │     CodeAgent (smolagents)           │  │
│  │  ┌────────────────────────────────┐  │  │
│  │  │  Base Tools (from tools.py)    │  │  │
│  │  │  - click_box                   │  │  │
│  │  │  - type_text                   │  │  │
│  │  │  - scroll                      │  │  │
│  │  │  - hotkey                      │  │  │
│  │  │  - wait                        │  │  │
│  │  └────────────────────────────────┘  │  │
│  │               +                      │  │
│  │  ┌────────────────────────────────┐  │  │
│  │  │  Loaded Skills (dynamic)       │  │  │
│  │  │  - open_web_browser()          │  │  │
│  │  │  - search_google()             │  │  │
│  │  │  - navigate_menu()             │  │  │
│  │  └────────────────────────────────┘  │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  Prompt: "Current subtask: Open browser"   │
│          "Screen: Desktop visible"         │
│                                            │
│  → CodeAgent selects tool                  │
│  → Returns: click_box([0.5, 0.95])         │
└────────────────────────────────────────────┘
```

#### UI Automation

```
ui_automation/
├── atomic_actions.py    (Low-level pyautogui wrapper)
│   └── UIAutomator
│       ├── click(x, y)
│       ├── type_text(text)
│       └── hotkey(*keys)
│
└── tools.py             (High-level @tool decorators)
    ├── click_box(box)   → Normalized coordinates
    ├── type_text(text)  → Input validation
    └── scroll(dir)      → Policy enforcement
```

**Execution Flow**:

```
CodeAgent Decision
    ↓
click_box([0.5, 0.1])     ← @tool wrapper
    ↓
Validate: [0.5, 0.1] in [0,1] ✓
    ↓
Denormalize: 0.5*1440, 0.1*900 = (720, 90)
    ↓
UIAutomator.click(720, 90)
    ↓
pyautogui.click(720, 90)  ← System-level action
```

## Data Flow Diagram

### Complete Agent Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT CYCLE (main.py)                        │
└─────────────────────────────────────────────────────────────────┘

1. INPUT PHASE
   User → "Search for Python tutorials"
           ↓
   TaskBreaker.decompose_task()
           ↓
   ["Open browser", "Go to search", "Enter query", "Submit"]
           ↓
   TaskState.initialize_task()

2. PERCEPTION PHASE (per subtask)
   take_screenshot()
           ↓
   screen_1234.jpg
           ↓
   InfoGatherer.analyze_screenshot()
           ↓
   "Desktop visible with dock at bottom..."
           ↓
   TaskState.add_observation()

3. PLANNING PHASE
   TaskState.get_current_subtask() → "Open browser"
           ↓
   SkillRetriever.retrieve_relevant_skills("Open browser")
           ↓
   ["open_web_browser", "launch_application"]
           ↓
   SkillDatabase.get_skill("open_web_browser")
           ↓
   Planner.load_skills([skill_code])
           ↓
   Planner.plan_next_action(task, observation, context)
           ↓
   {"tool": "click_box", "box": [0.5, 0.95]}

4. EXECUTION PHASE
   click_box([0.5, 0.95])
           ↓
   UIAutomator.click(720, 855)
           ↓
   [System executes click]
           ↓
   wait(2.0)
           ↓
   TaskState.add_action(action)

5. REFLECTION PHASE
   take_screenshot()
           ↓
   InfoGatherer.analyze_screenshot()
           ↓
   "Chrome browser window is now open..."
           ↓
   Reflector.judge_action_success(
       before_obs, after_obs, subtask
   )
           ↓
   (True, "Browser successfully opened")

6. LEARNING PHASE (if success)
   SkillManager.should_save_as_skill(actions, subtask)
           ↓
   True (2-15 actions, reusable pattern)
           ↓
   SkillManager.extract_skill(actions, subtask)
           ↓
   ("open_web_browser", "Opens Chrome browser", "def open...")
           ↓
   SkillDatabase.add_skill(...)
           ↓
   SkillRetriever.rebuild_embeddings()
           ↓
   TaskState.advance_subtask()

7. LOOP BACK
   If more subtasks → Go to step 2
   If task complete → End
   If failure detected → TaskBreaker.replan_after_failure()
```

## Token Efficiency

### Without Skill Retrieval

```
Prompt Size for 100 Skills:
  - 100 skills × 50 lines/skill = 5000 lines
  - ≈ 5000 tokens
  - Cost: High, slow inference
```

### With Skill Retrieval

```
Prompt Size with Top-5 Retrieved:
  - 5 skills × 50 lines/skill = 250 lines
  - ≈ 150 tokens
  - Cost: 97% reduction ✓
```

**Retrieval Process**:

```
Query: "open web browser"
    ↓
sentence-transformers encode
    ↓
[-0.12, 0.45, 0.78, ...]
    ↓
Cosine similarity vs all skills
    ↓
[("open_web_browser", 0.92),
 ("launch_app", 0.85),
 ("start_chrome", 0.81),
 ...]
    ↓
Top 5 skill_ids returned
    ↓
Load only relevant skills into CodeAgent
```

## Testing Architecture

```
test/
├── test_without_llm.py    (No API calls)
│   ├── Memory modules
│   ├── UI automation
│   └── File structure
│
├── test_modules.py        (With LLM)
│   ├── Vision analysis
│   ├── Action planning
│   └── Skill extraction
│
└── executor.py            (Action execution)
    └── ActionExecutor for legacy format
```

## Configuration & Policies

```python
# Environment Variables
OPENAI_API_KEY           # Required
UI_MAX_SCROLL_CLICKS=10  # Safety limit
UI_MAX_WAIT_SECONDS=5    # Timeout
UI_MAX_TYPE_CHARS=512    # Input limit
UI_ALLOW_ALT_F4=1        # Dangerous keys

# Data Files
memory/data/
├── short_term_state.json      # Session ephemeral
├── skills.json                # Persistent skills
└── skill_embeddings.npy       # Cached vectors
```

## Error Handling & Recovery

```
Action Failed
    ↓
Reflector.judge_action_success() → False
    ↓
Check: Is agent stuck?
    ↓
Reflector.detect_stuck_state() → True/False
    │
    ├─→ True: TaskBreaker.replan_after_failure()
    │           ↓
    │   Generate new subtask plan
    │           ↓
    │   Reset subtask index
    │           ↓
    │   Continue with new plan
    │
    └─→ False: Retry current subtask
                ↓
        Increment retry counter
                ↓
        If retry < 3: continue
        If retry >= 3: replan
```

## Performance Characteristics

| Metric                   | Value                                   |
| ------------------------ | --------------------------------------- |
| Screenshot latency       | ~100ms                                  |
| Vision LLM latency       | ~1-2s                                   |
| Action planning latency  | ~2-3s (first time), ~0.5s (with skills) |
| Action execution latency | ~0.1-0.5s                               |
| Skill retrieval latency  | ~50ms                                   |
| Memory persistence       | Instant (JSON write)                    |
| Total cycle time         | ~3-5s per action                        |

## Scalability Considerations

- **Skill Database**: Linear growth, O(n) storage
- **Skill Retrieval**: O(n) similarity search (can optimize with FAISS)
- **Memory Footprint**: ~1MB per 1000 skills
- **Embedding Build**: O(n) one-time cost per skill addition
- **Context Window**: Limited by LLM (current: 128k tokens for GPT-4)

## Security & Safety

✅ **Implemented**:

- Coordinate validation (0-1 range)
- Action policy limits (scroll, wait, type)
- Optional dangerous key blocking (Alt+F4)
- Fail-safe mouse position (corners)

⚠️ **User Responsibility**:

- Agent has full system control
- Always supervise execution
- Test on non-critical applications
- Review generated skills before trusting

---
