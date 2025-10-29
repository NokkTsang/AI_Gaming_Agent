#!/usr/bin/env python3
"""
Analyze token usage from agent log files.
Parses log files and calculates total token usage across all API calls.
"""

import re
import sys
from pathlib import Path


def parse_token_line(line: str) -> dict:
    """
    Parse a token usage line from the log.
    Format: "Tokens - Input: 1857, Output: 260, Total: 2117"

    Returns:
        dict with 'input', 'output', 'total' keys, or None if not a token line
    """
    pattern = r"Tokens - Input: (\d+), Output: (\d+), Total: (\d+)"
    match = re.search(pattern, line)
    if match:
        return {
            "input": int(match.group(1)),
            "output": int(match.group(2)),
            "total": int(match.group(3)),
        }
    return None


def analyze_log_file(log_path: str) -> dict:
    """
    Analyze a log file and extract token usage statistics.

    Args:
        log_path: Path to the log file

    Returns:
        Dictionary with token usage breakdown by module
    """
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Error: Log file not found: {log_path}")
        return None

    token_stats = {
        "vision": {"input": 0, "output": 0, "total": 0, "calls": 0},
        "planning": {"input": 0, "output": 0, "total": 0, "calls": 0},
        "task_breaking": {"input": 0, "output": 0, "total": 0, "calls": 0},
        "reflection": {"input": 0, "output": 0, "total": 0, "calls": 0},
        "completion_check": {"input": 0, "output": 0, "total": 0, "calls": 0},
        "overall": {"input": 0, "output": 0, "total": 0, "calls": 0},
    }

    current_section = None

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            # Detect which section we're in
            if "VISION API REQUEST" in line or "VISION API RESPONSE" in line:
                current_section = "vision"
            elif "ACTION PLANNER REQUEST" in line or "ACTION PLANNER RESPONSE" in line:
                current_section = "planning"
            elif (
                "TASK DECOMPOSITION REQUEST" in line
                or "TASK DECOMPOSITION RESPONSE" in line
            ):
                current_section = "task_breaking"
            elif (
                "SELF-REFLECTION REQUEST" in line or "SELF-REFLECTION RESPONSE" in line
            ):
                current_section = "reflection"
            elif "COMPLETION CHECK" in line:
                current_section = "completion_check"

            # Parse token usage lines
            tokens = parse_token_line(line)
            if tokens and current_section:
                token_stats[current_section]["input"] += tokens["input"]
                token_stats[current_section]["output"] += tokens["output"]
                token_stats[current_section]["total"] += tokens["total"]
                token_stats[current_section]["calls"] += 1

                token_stats["overall"]["input"] += tokens["input"]
                token_stats["overall"]["output"] += tokens["output"]
                token_stats["overall"]["total"] += tokens["total"]
                token_stats["overall"]["calls"] += 1

    return token_stats


def print_token_report(stats: dict, log_path: str):
    """Print a formatted token usage report."""
    print(f"\n{'='*80}")
    print(f"TOKEN USAGE REPORT: {Path(log_path).name}")
    print(f"{'='*80}\n")

    # Module breakdown
    print("By Module:")
    print(f"{'Module':<20} {'Calls':<8} {'Input':<12} {'Output':<12} {'Total':<12}")
    print("-" * 80)

    for module in [
        "vision",
        "planning",
        "task_breaking",
        "reflection",
        "completion_check",
    ]:
        s = stats[module]
        if s["calls"] > 0:
            print(
                f"{module.replace('_', ' ').title():<20} {s['calls']:<8} "
                f"{s['input']:<12,} {s['output']:<12,} {s['total']:<12,}"
            )

    print("-" * 80)
    s = stats["overall"]
    print(
        f"{'TOTAL':<20} {s['calls']:<8} {s['input']:<12,} {s['output']:<12,} {s['total']:<12,}"
    )

    # Cost estimate (GPT-4.1 pricing)
    # Input: $3.00 per 1M tokens, Output: $12.00 per 1M tokens
    # Cached input: $0.75 per 1M tokens (not calculated here)
    input_cost = (s["input"] / 1_000_000) * 3.00
    output_cost = (s["output"] / 1_000_000) * 12.00
    total_cost = input_cost + output_cost

    print(f"\nEstimated Cost (GPT-4.1 rates):")
    print(f"  Input tokens:  ${input_cost:.4f} ($3.00/1M)")
    print(f"  Output tokens: ${output_cost:.4f} ($12.00/1M)")
    print(f"  Total:         ${total_cost:.4f}")
    print(f"  Note: Cached input tokens ($0.75/1M) not tracked separately")
    print(f"\n{'='*80}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_tokens.py <log_file_path>")
        print("\nExample:")
        print("  python3 analyze_tokens.py task_20251029_193919.log")
        print("  python3 analyze_tokens.py /path/to/task_log.log")
        sys.exit(1)

    log_path = sys.argv[1]
    stats = analyze_log_file(log_path)

    if stats:
        print_token_report(stats, log_path)


if __name__ == "__main__":
    main()
