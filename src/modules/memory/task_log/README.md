# Task Logs

This directory contains timestamped log files of all agent runs.

## Log File Format

Files are named: `task_YYYYMMDD_HHMMSS.log`

Example: `task_20251022_143052.log`

## Contents

Each log file contains:

- Full task description
- All LLM prompts sent (Vision API, Action Planner, Task Decomposition, etc.)
- All LLM responses received
- Token usage for each API call (Input/Output/Total)
- OCR and YOLO detection summaries
- Action execution details
- Self-reflection judgments
- Skill curation activities
- Session summary

## Log Retention

Logs are kept indefinitely for debugging and analysis. You can safely delete old logs if disk space is a concern.

## Usage

Logs are automatically created when running:

```bash
python -m src.modules.main "Your task description"
```

The log file path is displayed at the start and end of each run.

## Token Usage Analysis

Use the `analyze_tokens.py` script to calculate total token usage and costs:

```bash
# From this directory:
python3 analyze_tokens.py task_20251029_193919.log

# From project root:
python3 src/modules/memory/task_log/analyze_tokens.py src/modules/memory/task_log/task_20251029_193919.log
```

### Pricing (GPT-4.1)

- **Input:** $3.00 per 1M tokens
- **Output:** $12.00 per 1M tokens
- **Cached Input:** $0.75 per 1M tokens (not tracked separately)

The script provides:

- Token breakdown by module (vision, planning, task breaking, reflection)
- Total token counts (input/output/total)
- Estimated cost based on GPT-4.1 pricing
