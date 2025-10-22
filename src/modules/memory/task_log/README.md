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
