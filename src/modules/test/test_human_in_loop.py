"""
Human-in-the-loop planning test

Goal: Replicate the agent's end-to-end decision flow, but DO NOT execute actions.
Instead:
- The script proposes the next action(s)
- A human operator performs them manually
- The script captures a new screenshot and continues planning based on the updated UI

This isolates "agent judgment" from "execution reliability".

Run from repo root:
  python -m src.modules.test.test_human_in_loop "<your task>"

Optional env/config:
- AGENT_WINDOW_TITLE or WINDOW_TITLE: prefer window capture by (substring) title; falls back to fullscreen
- OPENAI_API_KEY (+ any provider-specific config you use in your environment)

Notes:
- Vision API must be available; otherwise the script will stop after printing a helpful message.
- Screenshots are saved under src/modules/test/ with timestamps for auditing.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Ensure repo root on path (so `modules.*` imports work with -m)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.modules.screen_input.screen_capture import take_screenshot
from src.modules.information_gathering.info_gather import (
    analyze_screenshot_with_detection,
)
from src.modules.action_planning.planner import ActionPlanner


def _compare_images(img1_path: str, img2_path: str, threshold: float = 0.002) -> bool:
    """Lightweight difference check: True if changed > threshold.

    threshold default (0.2%) matches a sensitive change detection to catch small updates.
    """
    try:
        from PIL import Image
        import numpy as np

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        arr1 = np.array(img1, dtype=float)
        arr2 = np.array(img2, dtype=float)
        diff = abs(arr1 - arr2)
        change_ratio = diff.mean() / 255.0
        return change_ratio > threshold
    except Exception as e:
        print(f"   Warning: compare_images failed, assuming changed. Details: {e}")
        return True


def _env_window_title() -> Optional[str]:
    return os.getenv("AGENT_WINDOW_TITLE") or os.getenv("WINDOW_TITLE")


def run_human_in_loop(task: str, max_steps: int = 20, model: str = "gpt-4o-mini", enable_ocr: bool = True) -> None:
    print("Human-in-the-loop Planning Test")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Model: {model}  |  Max steps: {max_steps}  |  OCR: {enable_ocr}")

    # Prepare output directory
    out_dir = os.path.join(repo_root, "src", "modules", "test")
    os.makedirs(out_dir, exist_ok=True)
    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Window preference
    preferred_window = _env_window_title()
    if preferred_window:
        print(f"Preferred window (from env): '{preferred_window}'")
    else:
        print("No preferred window set; defaulting to fullscreen capture.")

    # Initialize planner
    planner = ActionPlanner(model=model, api_key=os.getenv("OPENAI_API_KEY"))

    # History buffer (simple list of action dicts)
    action_history: List[Dict[str, Any]] = []

    # Step 0: initial screenshot and vision
    img_path, _ = take_screenshot(
        tid=f"{session_stamp}_step0",
        output_dir=out_dir,
        window_title=preferred_window,
        method="auto",
    )
    print(f"Saved initial screenshot: {img_path}")

    try:
        observation = analyze_screenshot_with_detection(
            img_path,
            question=(
                "Analyze screenshot for GUI automation. List actionable UI elements and obvious next steps."
            ),
            model=model,
            include_ocr=enable_ocr,
            enable_object_detection=True,
        )
    except Exception as e:
        print("\n⨯ Vision analysis failed.")
        print("  Hints:")
        print("  - Check OPENAI_API_KEY / provider access")
        print("  - If region-restricted, consider alternative endpoint or Azure OpenAI")
        print(f"  Error: {e}")
        return

    print("\nInitial observation summarized. Starting interactive loop…")

    for step in range(1, max_steps + 1):
        print("\n" + "=" * 80)
        print(f"Iteration {step}/{max_steps}")
        print("=" * 80)

        # Plan next action
        action_dict = planner.plan_next_action(task=task, observation=observation, action_history=action_history)
        if not action_dict:
            print("No action proposed. Stopping.")
            break

        # Support batched actions or single action
        actions: List[Dict[str, Any]]
        if "actions" in action_dict and isinstance(action_dict["actions"], list):
            thought = action_dict.get("thought", "")
            actions = action_dict["actions"]
            print(f"Thought: {thought}")
            print("Proposed actions (in order):")
            for i, a in enumerate(actions, 1):
                print(f"  {i}. {a}")
        else:
            thought = action_dict.get("thought", "")
            print(f"Thought: {thought}")
            print(f"Proposed action: {action_dict}")
            actions = [
                {k: v for k, v in action_dict.items() if k in ("action_type", "action_inputs")}
            ]

        # Append to history (as proposed)
        for a in actions:
            hist_item = {"action_type": a.get("action_type"), "action_inputs": a.get("action_inputs", {})}
            action_history.append(hist_item)

        # Ask human to perform
        print("\nPlease perform the proposed action(s) manually on your machine.")
        print("When done, press Enter to continue; or type 'skip' to skip; 'quit' to exit.")
        cmd = input("[enter/skip/quit] > ").strip().lower()
        if cmd == "quit":
            print("User requested quit.")
            break
        elif cmd == "skip":
            print("Skipping capture and re-planning.")
            continue

        # Capture new screen after human performed action(s)
        img_after, _ = take_screenshot(
            tid=f"{session_stamp}_step{step}",
            output_dir=out_dir,
            window_title=preferred_window,
            method="auto",
        )
        print(f"Saved screenshot after human action(s): {img_after}")

        # Quick change detection (informational)
        try:
            changed = _compare_images(img_path, img_after)
            print(f"Change detected: {'YES' if changed else 'NO'}")
        except Exception:
            pass

        # Update for next iteration
        img_path = img_after
        try:
            observation = analyze_screenshot_with_detection(
                img_path,
                question=(
                    "Analyze the updated screenshot. Based on the task, propose the next action(s)."
                ),
                model=model,
                include_ocr=enable_ocr,
                enable_object_detection=True,
            )
        except Exception as e:
            print("\n⨯ Vision analysis failed during loop. Stopping.")
            print(f"  Error: {e}")
            break

        # Optional end condition if the model says finished
        if isinstance(action_dict, dict) and action_dict.get("action_type") == "finished":
            print("Model indicated task finished.")
            break

    print("\nDone.")


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]).strip() or "Open Chrome and search for OpenAI"
    run_human_in_loop(task)
