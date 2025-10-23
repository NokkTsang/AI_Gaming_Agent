"""
AI Gaming Agent - Main Entry Point

Full AI Gaming Agent with memory, reflection, and skill learning.
Implements Cradle-inspired architecture.

Run from project root: python -m src.modules.main
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Add src directory to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from modules.screen_input.screen_capture import take_screenshot
from modules.information_gathering.info_gather import (
    analyze_screenshot,
    analyze_screenshot_with_detection,
)
from modules.action_planning.planner import ActionPlanner
from modules.ui_automation.executor import ActionExecutor
from modules.ui_automation.atomic_actions import UIAutomator

# Memory system
from modules.memory.short_term import TaskState
from modules.memory.long_term import SkillDatabase
from modules.memory.skill_retrieval import EmbeddingRetriever

# Reasoning modules
from modules.self_reflection.reflector import Reflector
from modules.task_inference.task_breaker import TaskBreaker
from modules.skill_curation.skill_manager import SkillManager

# Load environment variables
load_dotenv()


class TaskLogger:
    """Handles logging of terminal output to files."""

    def __init__(self, log_dir: str = "src/modules/memory/task_log"):
        """Initialize logger with timestamped log file."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"task_{timestamp}.log"

        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def start_logging(self):
        """Start capturing all print statements to file."""
        self.file_handle = open(self.log_file, "w", encoding="utf-8")
        sys.stdout = TeeOutput(self.original_stdout, self.file_handle)
        sys.stderr = TeeOutput(self.original_stderr, self.file_handle)

    def stop_logging(self):
        """Stop capturing and close log file."""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if hasattr(self, "file_handle"):
            self.file_handle.close()

    def get_log_path(self) -> str:
        """Return path to current log file."""
        return str(self.log_file)


class TeeOutput:
    """Redirect output to both original stream and file."""

    def __init__(self, original, file_handle):
        self.original = original
        self.file = file_handle

    def write(self, data):
        self.original.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()


class AIGamingAgent:
    """
    Full AI Gaming Agent with memory, reflection, and skill learning.

    This agent implements the complete Cradle-inspired architecture:
    1. Screen capture
    2. Information gathering (vision LLM)
    3. Task decomposition into subtasks
    4. Skill retrieval from long-term memory
    5. Action planning with retrieved skills
    6. Action execution
    7. Self-reflection on success/failure
    8. Skill curation and saving
    """

    def __init__(
        self,
        max_steps: int = 50,
        model: str = "gpt-4o-mini",
        enable_ocr: bool = True,
    ):
        """
        Initialize the full agent with all components.

        Args:
            max_steps: Maximum steps per task (default: 50)
            model: OpenAI model for ALL components (planning, reasoning, and vision)
            enable_ocr: Enable OCR text detection (default: True)
        """
        self.max_steps = max_steps
        self.max_subtask_attempts = 3  # Default, can be overridden
        self.model = model
        self.vision_model = model  # Use same model for consistency
        self.enable_ocr = enable_ocr
        api_key = os.getenv("OPENAI_API_KEY")
        # Optional: Prefer capturing a specific window first (then fallback to fullscreen)
        # Read from environment to avoid hard-coding in code.
        # Support WINDOW_TITLE for convenience.
        self.preferred_window_title = (
            os.getenv("WINDOW_TITLE") or None
        )

        # Vision system prompt (used for all screenshot analysis)
        self.vision_system_prompt = """Analyze screenshot for GUI automation.

FOCUS STATE: Any field focused? (cursor blinking, highlighted border)
→ If focused, TYPE not CLICK

UI ELEMENTS (interactive only):
List each: type, appearance, coordinates [x,y] from DETECTED TEXT OR visual estimate, text, brief note

OBJECT DETECTION (for visual elements without text):
If you see icons, markers, or game elements that need PRECISE coordinates, request detection:
REQUEST_DETECTION: <object name>

Examples:
- "red flag icon below START HERE text" → REQUEST_DETECTION: red flag
- "tower building sites on map" → REQUEST_DETECTION: tower site  
- "enemy units on path" → REQUEST_DETECTION: enemy
- "Search button (has text)" → Use OCR coordinates (no detection needed)

Use OCR coordinates for text. Request detection for visual elements. Be concise."""

        # Initialize logger
        self.logger = TaskLogger()

        print("Initializing AI Gaming Agent...")

        # Screen input and UI automation
        self.automator = UIAutomator()
        self.executor = ActionExecutor(
            screen_width=1920,
            screen_height=1080,
            automator=self.automator,
        )

        # Memory system
        print("  Loading memory system...")
        self.short_term = TaskState()
        self.long_term = SkillDatabase()
        self.skill_retrieval = EmbeddingRetriever()

        # Reasoning modules (all use the same model)
        print("  Initializing reasoning modules...")
        self.task_breaker = TaskBreaker(api_key=api_key, model=model)
        self.reflector = Reflector(api_key=api_key, model=model)
        self.skill_manager = SkillManager(api_key=api_key, model=model)

        # Action planner
        self.planner = ActionPlanner(model=model, api_key=api_key)

        print("✓ Agent initialized successfully!\n")

    def update_screen_size(self):
        """Dynamically update screen size for executor."""
        import pyautogui

        w, h = pyautogui.size()
        self.executor.screen_width = int(w)
        self.executor.screen_height = int(h)
        print(f"[DEBUG] Screen size updated: {w}x{h}")

    def compare_screenshots(
        self, img1_path: str, img2_path: str, threshold: float = 0.002
    ) -> bool:
        """Compare two screenshots to detect if a change occurred.

        Args:
            img1_path: Path to first screenshot
            img2_path: Path to second screenshot
            threshold: Minimum change ratio to consider different (default 0.2%)

        Returns:
            True if screenshots are significantly different, False if same
        """
        try:
            from PIL import Image
            import numpy as np

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # Resize to same size if different
            if img1.size != img2.size:
                img2 = img2.resize(img1.size)

            # Convert to numpy arrays
            arr1 = np.array(img1)
            arr2 = np.array(img2)

            # Calculate difference ratio
            diff = np.abs(arr1.astype(float) - arr2.astype(float))
            change_ratio = np.mean(diff) / 255.0

            return change_ratio > threshold

        except Exception as e:
            print(f"   Warning: Screenshot comparison failed: {e}")
            return True  # Assume change occurred if comparison fails

    def run(self, task: str):
        """
        Main agent loop - execute task with full memory and learning.

        Args:
            task: High-level task description
        """
        # Start logging all output
        self.logger.start_logging()

        print(f"\n{'='*80}")
        print(f"Task: {task}")
        print(f"Log file: {self.logger.get_log_path()}")
        print(f"{'='*80}\n")

        self.update_screen_size()

        try:
            # Step 1: Capture initial screen
            print("Step 1: Capturing initial screen...")
            # Prefer window capture if a target title is configured; will fallback to fullscreen automatically
            screenshot_path = take_screenshot(
                window_title=self.preferred_window_title, method="auto"
            )
            initial_observation = analyze_screenshot_with_detection(
                screenshot_path,
                self.vision_system_prompt,
                model=self.vision_model,
                include_ocr=self.enable_ocr,
                enable_object_detection=True,
            )
            print(
                f"  Initial observation received ({len(initial_observation)} chars)\n"
            )

            # Step 2: Decompose task into subtasks
            print("Step 2: Breaking down task into subtasks...")
            subtasks = self.task_breaker.decompose_task(task, initial_observation)
            print(f"  ✓ Generated {len(subtasks)} subtasks:")
            for i, st in enumerate(subtasks, 1):
                print(f"    {i}. {st}")
            print()

            # Step 3: Initialize short-term memory
            self.short_term.initialize_task(task, subtasks)
            self.short_term.add_observation(initial_observation)

            # Main execution loop
            step = 0
            consecutive_failures = 0
            max_failures = 3
            subtask_attempts = {}  # Track attempts per subtask

            # Timing tracking for performance monitoring
            step_timings = {
                "vision": [],
                "planning": [],
                "execution": [],
                "reflection": [],
                "completion_check": [],
            }
            task_start_time = time.time()

            while step < self.max_steps:
                step += 1
                print(f"\n{'='*80}")
                print(f"Iteration {step}/{self.max_steps}")
                print(f"{'='*80}")

                # Get current subtask
                current_subtask = self.short_term.get_current_subtask()
                if current_subtask is None:
                    print("\n✓ All subtasks completed!")
                    break

                print(f"Current subtask: {current_subtask}\n")

                # Track subtask attempts
                if current_subtask not in subtask_attempts:
                    subtask_attempts[current_subtask] = 0
                subtask_attempts[current_subtask] += 1

                # Skip subtask if exceeded max attempts
                if subtask_attempts[current_subtask] > self.max_subtask_attempts:
                    print(
                        f"  ⚠️ Subtask failed after {self.max_subtask_attempts} attempts, skipping to next subtask"
                    )
                    print(f"     Failed subtask: {current_subtask}\n")

                    # Log the skip
                    self.short_term.add_observation(
                        f"SUBTASK SKIPPED: '{current_subtask}' failed after {self.max_subtask_attempts} attempts. Moving to next subtask."
                    )

                    # Advance to next subtask
                    self.short_term.advance_subtask()
                    subtask_attempts[current_subtask] = 0  # Reset counter
                    continue

                # Warn if approaching limit
                if subtask_attempts[current_subtask] == self.max_subtask_attempts:
                    print(
                        f"  ⚠️ WARNING: Last attempt for this subtask (attempt {subtask_attempts[current_subtask]}/{self.max_subtask_attempts})"
                    )
                    print("  If this fails, subtask will be skipped\n")

                    self.short_term.add_observation(
                        f"FINAL ATTEMPT: This is the last try for '{current_subtask}'. "
                        + "Consider alternative actions: different action types, different coordinates, hotkeys, or different UI elements."
                    )

                # Step 4: Retrieve relevant skills
                print("  Retrieving relevant skills...")
                relevant_skill_ids = self.skill_retrieval.retrieve_relevant_skills(
                    current_subtask, top_k=3
                )
                print(f"  ✓ Retrieved {len(relevant_skill_ids)} skills\n")

                # Get recent context
                context = self.short_term.get_recent_context(n=3)
                recent_obs = context["recent_observations"]
                current_observation = (
                    recent_obs[-1]["content"] if recent_obs else initial_observation
                )

                # Step 5: Plan next action
                print("  Planning next action...")
                planning_start = time.time()
                action_dict = self.planner.plan_next_action(
                    task=task,
                    observation=current_observation,
                    action_history=context["recent_actions"],
                )
                planning_time = time.time() - planning_start
                step_timings["planning"].append(planning_time)

                if action_dict is None:
                    print("  ⨯ Failed to generate action")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("\n⨯ Too many failures. Stopping.")
                        break
                    continue

                print(f"  Action: {action_dict['action_type']}")
                print(f"  Inputs: {action_dict['action_inputs']}")
                print(f"  [Planning took {planning_time:.2f}s]\n")

                # Check if finished
                if action_dict["action_type"] == "finished":
                    print("✓ Task completed!")
                    break

                # Record observation and screenshot before action
                observation_before = current_observation
                screenshot_before = screenshot_path

                # Step 6: Execute action
                print("  Executing action...")
                execution_start = time.time()
                try:
                    self.executor.execute(action_dict)
                    self.short_term.add_action(action_dict)  # Store as dict, not string
                    execution_time = time.time() - execution_start
                    step_timings["execution"].append(execution_time)
                    print(f"  ✓ Action executed [took {execution_time:.2f}s]\n")
                except Exception as e:
                    execution_time = time.time() - execution_start
                    step_timings["execution"].append(execution_time)
                    print(f"  ⨯ Execution failed: {e} [took {execution_time:.2f}s]\n")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    continue

                # Wait for UI to update
                time.sleep(0.5)

                # Step 7: Capture screen after action
                print("  Capturing result...")
                screenshot_after = take_screenshot(
                    window_title=self.preferred_window_title, method="auto"
                )

                # Check if screen changed (self-correction mechanism)
                # Lower threshold (0.2%) catches subtle changes like focus borders
                screen_changed = self.compare_screenshots(
                    screenshot_before, screenshot_after
                )

                if not screen_changed and action_dict["action_type"] in [
                    "click",
                    "double_click",
                    "right_click",
                ]:
                    print(
                        "  Warning: Screen did not change after click. Retrying with correction..."
                    )

                    # Extract target object from thought or subtask
                    target_objects = []
                    thought = action_dict.get("thought", "")
                    if "flag" in thought.lower() or "flag" in current_subtask.lower():
                        target_objects.extend(["red flag", "flag marker", "flag icon"])
                    if (
                        "button" in thought.lower()
                        or "button" in current_subtask.lower()
                    ):
                        target_objects.extend(
                            [
                                "button",
                                "start button",
                                "game button",
                                "menu button",
                                "play button",
                            ]
                        )
                    if "icon" in thought.lower():
                        target_objects.extend(["icon", "game icon"])

                    # If START is mentioned, add specific terms
                    if "start" in thought.lower() or "start" in current_subtask.lower():
                        target_objects.extend(
                            ["start button", "start menu", "start icon"]
                        )

                    # Remove duplicates
                    target_objects = (
                        list(set(target_objects)) if target_objects else None
                    )

                    # Ask LLM to provide corrected coordinates
                    correction_prompt = (
                        self.vision_system_prompt
                        + f"\n\nPREVIOUS FAILED ATTEMPT:\n"
                        + f"- Clicked at: {action_dict['action_inputs'].get('start_box', 'unknown')}\n"
                        + f"- Goal: {action_dict.get('thought', 'click action')}\n"
                        + f"- Result: Nothing happened, screen unchanged\n\n"
                        + "CRITICAL ANALYSIS:\n"
                        + "1. Is target element focused/active?\n"
                        + "2. Should agent TYPE (if field focused) or CLICK (if not focused)?\n"
                        + "3. If clicking needed: exact coordinates [x, y] from OCR\n\n"
                        + "Format:\n"
                        + "STATE: [element focus state]\n"
                        + "RECOMMENDED ACTION: [type/click/hotkey]\n"
                        + "COORDINATES: [x, y] (only if clicking recommended)"
                    )

                    correction_observation = analyze_screenshot_with_detection(
                        screenshot_after,
                        correction_prompt,
                        model=self.vision_model,
                        include_ocr=self.enable_ocr,
                        enable_object_detection=True,
                    )
                    print(
                        f"  Correction suggestion:\n{correction_observation}\n"
                    )  # Let the agent retry in next iteration with this new information
                    self.short_term.add_observation(
                        f"CLICK FAILED at {action_dict['action_inputs'].get('start_box')}. "
                        + f"Correction analysis:\n{correction_observation}\n"
                        + "Use the EXACT coordinates provided above in your next attempt."
                    )
                    consecutive_failures += 1
                    continue

                # Normal observation after successful action
                vision_start = time.time()
                observation_prompt = (
                    self.vision_system_prompt
                    + f"\n\nPrevious state: {observation_before[:100]}...\n"
                    + "Describe what changed after the action."
                )
                observation_after = analyze_screenshot_with_detection(
                    screenshot_after,
                    observation_prompt,
                    model=self.vision_model,
                    include_ocr=self.enable_ocr,
                    enable_object_detection=True,
                )
                vision_time = time.time() - vision_start
                step_timings["vision"].append(vision_time)
                self.short_term.add_observation(observation_after)
                print(
                    f"  New state: {observation_after[:150]}... [vision took {vision_time:.2f}s]\n"
                )

                # Reset consecutive failures on successful action
                consecutive_failures = 0

                # Step 8: Self-reflection
                print("  Reflecting on outcome...")
                reflection_start = time.time()
                success, reasoning = self.reflector.judge_action_success(
                    task_goal=task,
                    current_subtask=current_subtask,
                    action_taken=f"{action_dict['action_type']}({action_dict['action_inputs']})",
                    observation_before=observation_before,
                    observation_after=observation_after,
                )
                reflection_time = time.time() - reflection_start
                step_timings["reflection"].append(reflection_time)

                if success:
                    print(
                        f"  ✓ Success: {reasoning} [reflection took {reflection_time:.2f}s]\n"
                    )
                    consecutive_failures = 0

                    # Advance to next subtask
                    self.short_term.advance_subtask()

                    # Check if task is complete after successful subtask
                    remaining_subtasks = self.short_term.state["subtasks"][
                        self.short_term.state["current_subtask_index"] :
                    ]

                    if not remaining_subtasks:
                        # All subtasks done - verify overall goal is achieved
                        print(
                            "  All subtasks completed. Verifying overall task goal..."
                        )
                        completion_start = time.time()
                        is_complete = self.task_breaker.check_task_completion(
                            task_goal=task, current_observation=observation_after
                        )
                        completion_time = time.time() - completion_start
                        step_timings["completion_check"].append(completion_time)

                        if is_complete:
                            print(
                                f"  ✓ Task goal verified as complete! [check took {completion_time:.2f}s]"
                            )
                            break
                        else:
                            print(
                                f"  Task goal not yet achieved despite completing all subtasks [check took {completion_time:.2f}s]"
                            )
                            print("  Generating additional subtasks...")
                            # Generate more subtasks to complete the goal
                            new_subtasks = self.task_breaker.decompose_task(
                                task, observation_after
                            )
                            self.short_term.update_subtasks(new_subtasks)

                    elif len(remaining_subtasks) <= 2:
                        # Near completion - proactively check if goal already achieved
                        print(
                            "  Near completion. Checking if task goal is already achieved..."
                        )
                        completion_start = time.time()
                        is_complete = self.task_breaker.check_task_completion(
                            task_goal=task, current_observation=observation_after
                        )
                        completion_time = time.time() - completion_start
                        step_timings["completion_check"].append(completion_time)

                        if is_complete:
                            print(
                                f"  ✓ Task goal achieved early! [check took {completion_time:.2f}s]"
                            )
                            break
                        else:
                            print(
                                f"  Goal not yet achieved. Continuing... [check took {completion_time:.2f}s]"
                            )

                    # Step 9: Skill curation (save successful patterns)
                    recent_actions = [
                        a["action"] for a in context["recent_actions"][-5:]
                    ]
                    if len(recent_actions) >= 2:
                        if self.skill_manager.should_save_as_skill(
                            recent_actions, current_subtask
                        ):
                            print("  Curating new skill...")
                            skill_name, skill_desc, skill_code = (
                                self.skill_manager.extract_skill(
                                    recent_actions, current_subtask, task
                                )
                            )
                            skill_id = self.long_term.add_skill(
                                skill_name=skill_name,
                                skill_description=skill_desc,
                                skill_code=skill_code,
                                task_context=task,
                            )
                            print(f"  ✓ Saved skill: {skill_name} (ID: {skill_id})")
                            self.skill_retrieval.rebuild_embeddings()
                else:
                    print(f"  ⨯ Failed: {reasoning}\n")
                    consecutive_failures += 1

                    if consecutive_failures >= max_failures:
                        print("  ! Too many failures. Replanning...")
                        completed = self.short_term.state["subtasks"][
                            : self.short_term.state["current_subtask_index"]
                        ]
                        new_subtasks = self.task_breaker.replan_after_failure(
                            task_goal=task,
                            failed_subtask=current_subtask,
                            failure_reason=reasoning,
                            current_observation=observation_after,
                            completed_subtasks=completed,
                        )
                        print(f"  ✓ New plan with {len(new_subtasks)} subtasks")
                        self.short_term.update_subtasks(new_subtasks)
                        consecutive_failures = 0

            # Final summary
            total_time = time.time() - task_start_time
            print(f"\n{'='*80}")
            print("Session Summary")
            print(f"{'='*80}")
            print(f"Total iterations: {step}")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            print(f"Skills in database: {self.long_term.get_skill_count()}")
            print(f"Actions taken: {len(self.short_term.state['actions'])}")

            # Performance breakdown
            if step_timings["vision"]:
                avg_vision = sum(step_timings["vision"]) / len(step_timings["vision"])
                print(f"\nPerformance Breakdown:")
                print(
                    f"  Vision analysis: {len(step_timings['vision'])} calls, avg {avg_vision:.2f}s, total {sum(step_timings['vision']):.2f}s"
                )
            if step_timings["planning"]:
                avg_planning = sum(step_timings["planning"]) / len(
                    step_timings["planning"]
                )
                print(
                    f"  Action planning: {len(step_timings['planning'])} calls, avg {avg_planning:.2f}s, total {sum(step_timings['planning']):.2f}s"
                )
            if step_timings["execution"]:
                avg_execution = sum(step_timings["execution"]) / len(
                    step_timings["execution"]
                )
                print(
                    f"  Action execution: {len(step_timings['execution'])} calls, avg {avg_execution:.2f}s, total {sum(step_timings['execution']):.2f}s"
                )
            if step_timings["reflection"]:
                avg_reflection = sum(step_timings["reflection"]) / len(
                    step_timings["reflection"]
                )
                print(
                    f"  Self-reflection: {len(step_timings['reflection'])} calls, avg {avg_reflection:.2f}s, total {sum(step_timings['reflection']):.2f}s"
                )
            if step_timings["completion_check"]:
                avg_completion = sum(step_timings["completion_check"]) / len(
                    step_timings["completion_check"]
                )
                print(
                    f"  Completion checks: {len(step_timings['completion_check'])} calls, avg {avg_completion:.2f}s, total {sum(step_timings['completion_check']):.2f}s"
                )

            print(f"\nLog saved to: {self.logger.get_log_path()}")
            print(f"{'='*80}\n")

        except KeyboardInterrupt:
            print("\n\n! Agent interrupted by user")
            print(f"Log saved to: {self.logger.get_log_path()}")
        except Exception as e:
            print(f"\n\n⨯ Agent error: {e}")
            print(f"Log saved to: {self.logger.get_log_path()}")
            import traceback

            traceback.print_exc()
        finally:
            # Always stop logging and close file
            self.logger.stop_logging()


def main():
    """Main entry point."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("⨯ Error: OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or export it")
        sys.exit(1)

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        print("Usage: python -m src.modules.main '<task description>'")
        print("\nExample:")
        print("  python -m src.modules.main 'Open Chrome and search for OpenAI'")
        print("\nRunning with default task...")
        task = "Open a web browser, go to Google, and search for 'openai'"

    # ============================================================
    # CONFIGURATION: Edit these values to customize the agent
    # ============================================================
    MODEL = "gpt-4.1"
    MAX_STEPS = 50
    MAX_SUBTASK_ATTEMPTS = 3  # Max attempts per subtask before skipping
    ENABLE_OCR = True  # OCR provides precise coordinates for text elements
    # ============================================================

    print(f"Using model: {MODEL}")
    print(f"Max steps per task: {MAX_STEPS}")
    print(f"Max attempts per subtask: {MAX_SUBTASK_ATTEMPTS}")
    print(f"OCR enabled: {ENABLE_OCR}\n")

    agent = AIGamingAgent(max_steps=MAX_STEPS, model=MODEL, enable_ocr=ENABLE_OCR)
    agent.max_subtask_attempts = MAX_SUBTASK_ATTEMPTS  # Pass config to agent
    agent.run(task)


if __name__ == "__main__":
    main()
