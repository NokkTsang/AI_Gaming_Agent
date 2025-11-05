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
from typing import Tuple

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

# Game-TARS enhancements
from modules.action_planning.complexity_detector import ComplexityDetector
from modules.task_inference.task_clarifier import TaskClarifier
from modules.task_inference.completion_detector import CompletionDetector

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
        enable_grounding_dino: bool = True,
    ):
        """
        Initialize the full agent with all components.

        Args:
            max_steps: Maximum steps per task (default: 50)
            model: OpenAI model for ALL components (planning, reasoning, and vision)
            enable_ocr: Enable OCR text detection (default: True)
            enable_grounding_dino: Enable GroundingDINO object detection (default: True)
        """
        self.max_steps = max_steps
        self.max_subtask_attempts = 3  # Default, can be overridden
        self.model = model
        self.vision_model = model  # Use same model for consistency
        self.enable_ocr = enable_ocr
        self.enable_grounding_dino = enable_grounding_dino
        api_key = os.getenv("OPENAI_API_KEY")
        # Optional: Prefer capturing a specific window first (then fallback to fullscreen)
        # Read from environment to avoid hard-coding in code.
        # Support WINDOW_TITLE for convenience.
        self.preferred_window_title = os.getenv("WINDOW_TITLE") or None

        # Monitor selection: 0=all, 1=primary (default), 2=secondary, etc.
        self.monitor_index = int(os.getenv("MONITOR_INDEX", "1"))

        # Vision system prompt (used for all screenshot analysis)
        self.vision_system_prompt = """Analyze screenshot for GUI automation.

FOCUS STATE: Any field focused? (cursor blinking, highlighted border)
→ If focused, TYPE not CLICK

UI ELEMENTS (interactive only):
List each: type, appearance, NORMALIZED coordinates [x,y] from DETECTED TEXT, text, brief note
IMPORTANT: Use ONLY normalized [0,1] coordinates from OCR list. Do NOT estimate pixel coordinates.

SPATIAL CONSTRAINTS: If obstacles/walls present, note safe zones.

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

        # Game-TARS improvements
        print("  Initializing Game-TARS enhancements...")
        self.complexity_detector = ComplexityDetector(model=model, api_key=api_key)
        self.task_clarifier = TaskClarifier(model=model, api_key=api_key)
        self.completion_detector = CompletionDetector(model=model, api_key=api_key)

        # Action planner
        self.planner = ActionPlanner(model=model, api_key=api_key)

        print("✓ Agent initialized successfully!")
        print("  - Two-tier memory: 80 full + 2400 compressed steps")
        print("  - Sparse thinking: Reactive mode for simple actions")
        print("  - Task clarification: Structured instructions")
        print("  - Completion detection: Explicit validation")
        print()

    def update_screen_size(self, screen_region: Tuple[int, int, int, int]):
        """Update screen size for executor based on actual captured region.

        Args:
            screen_region: Tuple of (left, top, width, height) of captured screen
        """
        # Update executor with actual captured screen dimensions and offset
        self.executor.screen_left = int(screen_region[0])
        self.executor.screen_top = int(screen_region[1])
        self.executor.screen_width = int(screen_region[2])
        self.executor.screen_height = int(screen_region[3])
        print(
            f"[DEBUG] Screen region updated: offset=({screen_region[0]}, {screen_region[1]}), size={screen_region[2]}x{screen_region[3]}"
        )

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

    def run(
        self,
        task: str,
        enable_sparse_thinking: bool = True,
        enable_clarification: bool = True,
    ):
        """
        Main agent loop - execute task with full memory and learning.

        NOW WITH GAME-TARS IMPROVEMENTS:
        - Task clarification with structured instructions
        - Two-tier memory (80 full + 2400 compressed)
        - Sparse thinking (reactive vs deliberative)
        - Explicit completion detection
        - Enhanced stuck detection with recovery

        Args:
            task: High-level task description
            enable_sparse_thinking: Use complexity detection for speed (default: True)
            enable_clarification: Clarify ambiguous tasks upfront (default: True)
        """
        # Start logging all output
        self.logger.start_logging()

        print(f"\n{'='*80}")
        print(f"Task: {task}")
        print(f"Log file: {self.logger.get_log_path()}")
        print(
            f"Enhancements: Sparse={enable_sparse_thinking}, Clarify={enable_clarification}"
        )
        print(f"{'='*80}\n")

        # Structured instruction (for completion detection)
        instruction = None

        try:
            # Step 0: Task Clarification (NEW - Game-TARS §3.1)
            if enable_clarification:
                print("Step 0: Task Clarification (Game-TARS Instruction Following)")
                print("-" * 80)
                # Note: We'll clarify before initial screenshot for now
                # In production, you might want initial screenshot first
                instruction = self.task_clarifier.clarify_task(
                    task, "", auto_accept_unambiguous=True
                )
                # Update task with clarified goal
                task = instruction.get("goal", task)
                # CRITICAL: Pass structured instruction to planner
                self.planner.structured_instruction = instruction

            # Step 1: Capture initial screen
            print("Step 1: Capturing initial screen...")
            # Capture screenshot and get the actual screen region for coordinate mapping
            screenshot_path, screen_region = take_screenshot(
                window_title=self.preferred_window_title,
                method="auto",
                focus_window=True,
                monitor_index=self.monitor_index,
            )

            # Update executor with actual captured screen dimensions
            self.update_screen_size(screen_region)

            initial_observation = analyze_screenshot_with_detection(
                screenshot_path,
                self.vision_system_prompt,
                model=self.vision_model,
                include_ocr=self.enable_ocr,
                enable_object_detection=self.enable_grounding_dino,
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

            # Step 3: Initialize short-term memory (NEW - Two-tier system)
            print("Step 3: Initializing two-tier memory...")
            self.short_term.initialize_task(task, subtasks)
            # Use new add_step instead of add_observation
            self.short_term.add_step(
                observation=initial_observation,
                thought="Initial task analysis",
                action={"action_type": "observe", "action_inputs": {}},
            )
            print(
                f"  ✓ Memory: {self.short_term.max_context_steps} full + {self.short_term.max_summary_steps} compressed"
            )
            print()

            # Main execution loop
            step = 0
            consecutive_failures = 0
            max_failures = 3
            subtask_attempts = {}  # Track attempts per subtask

            # Timing tracking for performance monitoring
            step_timings = {
                "complexity_check": [],  # NEW
                "vision": [],
                "planning": [],
                "execution": [],
                "reflection": [],
                "completion_check": [],
            }

            # Track sparse thinking stats
            thinking_count = 0
            reactive_count = 0

            task_start_time = time.time()

            # For stuck detection
            observation_history = [initial_observation]

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

                # NEW: Step 4.5 - Complexity Detection (Game-TARS Sparse Thinking)
                if enable_sparse_thinking:
                    print("  Step 4.5: Complexity check (sparse thinking)...")
                    complexity_start = time.time()

                    last_obs = (
                        observation_history[-1] if len(observation_history) > 0 else ""
                    )
                    last_action = (
                        context["recent_actions"][-1]
                        if context["recent_actions"]
                        else None
                    )

                    needs_thinking, reason = self.complexity_detector.needs_thinking(
                        current_observation=current_observation,
                        last_observation=last_obs,
                        current_subtask=current_subtask,
                        last_action=last_action,
                        action_history=context["recent_actions"],
                        force_thinking=(step == 1),  # Always think on first step
                    )

                    complexity_time = time.time() - complexity_start
                    step_timings["complexity_check"].append(complexity_time)
                    print(f"  → {reason} [{complexity_time:.2f}s]")
                else:
                    needs_thinking = True  # Default to always thinking
                    reason = "Sparse thinking disabled"

                # Step 5: Plan next action (Deep or Reactive)
                print(
                    f"  Planning next action ({'DEEP' if needs_thinking else 'REACTIVE'})..."
                )
                planning_start = time.time()

                if needs_thinking:
                    # Deep planning: Full reasoning + action
                    action_dict = self.planner.plan_next_action(
                        task=current_subtask,
                        observation=current_observation,
                        action_history=context["recent_actions"],
                    )
                    thinking_count += 1
                else:
                    # Reactive planning: Action only (fast)
                    action_dict = self.planner.plan_reactive_action(
                        task=current_subtask,
                        observation=current_observation,
                        action_history=context["recent_actions"],
                    )
                    reactive_count += 1

                planning_time = time.time() - planning_start
                step_timings["planning"].append(planning_time)

                if action_dict is None:
                    print("  ⨯ Failed to generate action")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("\n⨯ Too many failures. Stopping.")
                        break
                    continue

                # Handle action sequences (batch execution)
                actions_to_execute = []
                if "actions" in action_dict:
                    # Multi-action sequence
                    actions_to_execute = action_dict["actions"]
                    print(f"  Action sequence: {len(actions_to_execute)} actions")
                    for i, action in enumerate(actions_to_execute, 1):
                        print(
                            f"    {i}. {action['action_type']}({action['action_inputs']})"
                        )
                    print(f"  [Planning took {planning_time:.2f}s]\n")
                else:
                    # Single action
                    actions_to_execute = [action_dict]
                    print(f"  Action: {action_dict['action_type']}")
                    print(f"  Inputs: {action_dict['action_inputs']}")
                    print(f"  [Planning took {planning_time:.2f}s]\n")

                # NEW: Check if finished (with completion validation if instruction available)
                if action_dict.get("action_type") == "finished" or (
                    len(actions_to_execute) == 1
                    and actions_to_execute[0].get("action_type") == "finished"
                ):
                    if instruction:
                        # Validate completion against success criteria
                        print("\n  Validating task completion...")
                        completion_start = time.time()
                        allow_finish, completion_reason = (
                            self.completion_detector.validate_completion_action(
                                instruction=instruction,
                                observation=current_observation,
                                action_history=self.short_term.state["actions"],
                            )
                        )
                        completion_time = time.time() - completion_start
                        step_timings["completion_check"].append(completion_time)

                        if allow_finish:
                            print(f"  ✓ {completion_reason}")
                            print("✓ Task completed!")
                            break
                        else:
                            print(f"  ✗ {completion_reason}")
                            print(
                                "  ⚠️  Rejected premature completion - continuing task"
                            )
                            # Don't execute the finished action, continue instead
                            consecutive_failures += 1
                            continue
                    else:
                        # No instruction available, trust agent's judgment
                        print("✓ Task completed!")
                        break

                # Record observation and screenshot before actions
                observation_before = current_observation
                screenshot_before = screenshot_path

                # Step 6: Execute actions
                print("  Executing action(s)...")
                execution_start = time.time()
                executed_count = 0
                try:
                    for action in actions_to_execute:
                        self.executor.execute(action)
                        # Note: Actions will be stored with add_step() after observation
                        executed_count += 1
                        # Small delay between batched actions
                        if len(actions_to_execute) > 1:
                            time.sleep(0.1)

                    execution_time = time.time() - execution_start
                    step_timings["execution"].append(execution_time)
                    if len(actions_to_execute) > 1:
                        print(
                            f"  ✓ {executed_count} actions executed [took {execution_time:.2f}s]\n"
                        )
                    else:
                        print(f"  ✓ Action executed [took {execution_time:.2f}s]\n")
                except Exception as e:
                    execution_time = time.time() - execution_start
                    step_timings["execution"].append(execution_time)
                    print(
                        f"  ⨯ Execution failed after {executed_count} actions: {e} [took {execution_time:.2f}s]\n"
                    )
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    continue

                # Wait for UI to update
                time.sleep(0.5)

                # Step 7: Capture screen after action
                print("  Capturing result...")
                screenshot_after, screen_region_after = take_screenshot(
                    window_title=self.preferred_window_title,
                    method="auto",
                    focus_window=True,
                    monitor_index=self.monitor_index,
                )

                # Update screen size if region changed (shouldn't happen, but defensive)
                if screen_region_after != screen_region:
                    self.update_screen_size(screen_region_after)
                    screen_region = screen_region_after

                # Check if screen changed (self-correction mechanism)
                # Lower threshold (0.2%) catches subtle changes like focus borders
                screen_changed = self.compare_screenshots(
                    screenshot_before, screenshot_after
                )

                # Check if last executed action was a click (for failure detection)
                last_action = actions_to_execute[-1]
                if not screen_changed and last_action.get("action_type") in [
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
                        enable_object_detection=self.enable_grounding_dino,
                    )
                    print(
                        f"  Correction suggestion:\n{correction_observation}\n"
                    )  # Let the agent retry in next iteration with this new information

                    # Store failed attempt with correction using add_step
                    failed_observation = (
                        f"CLICK FAILED at {action_dict['action_inputs'].get('start_box')}. "
                        + f"Correction analysis:\n{correction_observation}\n"
                        + "Use the EXACT coordinates provided above in your next attempt."
                    )
                    thought = action_dict.get("thought", "Failed click attempt")
                    self.short_term.add_step(failed_observation, thought, action_dict)
                    observation_history.append(failed_observation)

                    consecutive_failures += 1
                    continue

                # Normal observation after successful action
                # Performance optimization: Skip vision analysis if screen unchanged
                vision_start = time.time()

                # Check if screen changed using pixel diff
                # Use different thresholds: 0.001 (0.1%) for keyboard actions, 0.002 (0.2%) for clicks
                last_action_type = (
                    actions_to_execute[-1].get("action_type")
                    if actions_to_execute
                    else None
                )
                threshold = 0.001 if last_action_type in ["hotkey", "type"] else 0.002
                screen_changed_final = self.compare_screenshots(
                    screenshot_before, screenshot_after, threshold=threshold
                )

                if not screen_changed_final and step > 0:
                    # Screen unchanged - reuse previous observation, skip expensive vision call
                    observation_after = observation_before
                    vision_time = time.time() - vision_start
                    step_timings["vision"].append(vision_time)

                    # Store with add_step (observation, thought, action)
                    thought = action_dict.get("thought", "Action continuation")
                    self.short_term.add_step(observation_after, thought, action_dict)
                    observation_history.append(observation_after)

                    print(
                        f"  Screen unchanged [threshold={threshold*100:.1f}%], reusing observation [saved ~40s]\n"
                    )
                else:
                    # Screen changed or first iteration - run full vision analysis
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
                        enable_object_detection=self.enable_grounding_dino,
                    )
                    vision_time = time.time() - vision_start
                    step_timings["vision"].append(vision_time)

                    # Store with add_step (observation, thought, action)
                    thought = action_dict.get("thought", "Action with new observation")
                    self.short_term.add_step(observation_after, thought, action_dict)
                    observation_history.append(observation_after)

                    print(
                        f"  New state: {observation_after[:150]}... [vision took {vision_time:.2f}s]\n"
                    )

                # Reset consecutive failures on successful action
                consecutive_failures = 0

                # Step 8: Self-reflection
                print("  Reflecting on outcome...")
                reflection_start = time.time()

                # Format action_taken string for reflection
                if len(actions_to_execute) > 1:
                    action_strs = [
                        f"{a['action_type']}({a['action_inputs']})"
                        for a in actions_to_execute
                    ]
                    action_taken = f"Action sequence: {', '.join(action_strs)}"
                else:
                    action = actions_to_execute[0]
                    action_taken = f"{action['action_type']}({action['action_inputs']})"

                success, reasoning = self.reflector.judge_action_success(
                    task_goal=task,
                    current_subtask=current_subtask,
                    action_taken=action_taken,
                    observation_before=observation_before,
                    observation_after=observation_after,
                )
                reflection_time = time.time() - reflection_start
                step_timings["reflection"].append(reflection_time)

                # Step 8.5: Check for stuck state with recovery (Game-TARS enhancement)
                if len(self.short_term.state["actions"]) >= 5:
                    print("  Checking for stuck patterns...")
                    stuck_start = time.time()
                    is_stuck, stuck_reason, recovery_actions = (
                        self.reflector.detect_stuck_with_recovery(
                            action_history=self.short_term.state["actions"],
                            observation_history=observation_history,
                            window_size=5,
                        )
                    )
                    stuck_time = time.time() - stuck_start

                    if is_stuck:
                        print(f"  ⚠️  STUCK DETECTED: {stuck_reason}")
                        print(
                            f"  Attempting recovery: {len(recovery_actions)} action(s)"
                        )

                        # Execute recovery actions
                        for i, recovery_action in enumerate(recovery_actions, 1):
                            print(
                                f"    Recovery {i}: {recovery_action['action_type']}({recovery_action['action_inputs']})"
                            )
                            try:
                                self.executor.execute(recovery_action)
                                time.sleep(0.5)
                            except Exception as e:
                                print(f"    ⨯ Recovery action failed: {e}")

                        print(
                            f"  ✓ Recovery attempted [check took {stuck_time:.2f}s]\n"
                        )
                        consecutive_failures = 0  # Reset after recovery attempt
                    else:
                        print(
                            f"  ✓ No stuck pattern detected [check took {stuck_time:.2f}s]"
                        )

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

            # Game-TARS Enhancement Statistics
            print(f"\n{'='*80}")
            print("GAME-TARS ENHANCEMENT STATISTICS")
            print(f"{'='*80}")

            # Two-tier memory statistics
            mem_stats = self.short_term.get_memory_stats()
            print(f"Two-Tier Memory:")
            print(
                f"  Context memory: {mem_stats['context_steps']}/{mem_stats['context_limit']} steps"
            )
            print(
                f"  Summary memory: {mem_stats['summary_steps']}/{mem_stats['summary_limit']} steps"
            )
            print(
                f"  Total capacity: {mem_stats['context_steps'] + mem_stats['summary_steps']}/{mem_stats['context_limit'] + mem_stats['summary_limit']} steps (vs 20 baseline)"
            )

            # Sparse thinking statistics (if enabled)
            if enable_sparse_thinking and (thinking_count + reactive_count) > 0:
                total_actions = thinking_count + reactive_count
                thinking_rate = thinking_count / total_actions * 100
                reactive_rate = reactive_count / total_actions * 100
                print(f"\nSparse Thinking:")
                print(
                    f"  Deep reasoning: {thinking_count} actions ({thinking_rate:.1f}%)"
                )
                print(
                    f"  Reactive (fast): {reactive_count} actions ({reactive_rate:.1f}%)"
                )
                print(
                    f"  Expected speedup: ~{1 / (thinking_rate/100 + reactive_rate/100 * 0.15):.1f}x"
                )

                # Complexity check performance
                if step_timings.get("complexity_check"):
                    avg_complexity = sum(step_timings["complexity_check"]) / len(
                        step_timings["complexity_check"]
                    )
                    print(
                        f"  Complexity checks: {len(step_timings['complexity_check'])} calls, avg {avg_complexity:.2f}s"
                    )

            # Task clarification (if enabled)
            if enable_clarification and instruction:
                print(f"\nTask Clarification:")
                print(f"  Structured instruction generated: Yes")
                print(
                    f"  Success criteria defined: {len(instruction.get('success_criteria', []))} criteria"
                )
                print(
                    f"  Constraints specified: {len(instruction.get('constraints', []))} constraints"
                )

            print(f"{'='*80}")

            print(f"\nLog saved to: {self.logger.get_log_path()}")
            print(
                f"Note: Token usage details are logged for each API call in the log file"
            )
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
    WINDOW_TITLE = None  # Set to None to list windows, or "Your Game" to auto-select
    MODEL = "gpt-4.1-nano"
    MAX_STEPS = 50
    MAX_SUBTASK_ATTEMPTS = 3
    ENABLE_OCR = True
    ENABLE_GROUNDING_DINO = True
    ENABLE_SPARSE_THINKING = True  # 3-4x speedup (recommended)
    ENABLE_CLARIFICATION = True  # Task Q&A (optional)
    # ============================================================

    # Set window title in environment (for window selection)
    if WINDOW_TITLE:
        os.environ["WINDOW_TITLE"] = WINDOW_TITLE

    print(f"Using model: {MODEL}")
    print(f"Max steps per task: {MAX_STEPS}")
    print(f"Max attempts per subtask: {MAX_SUBTASK_ATTEMPTS}")
    print(f"OCR enabled: {ENABLE_OCR}")
    print(f"GroundingDINO enabled: {ENABLE_GROUNDING_DINO}")
    print(f"Sparse thinking: {ENABLE_SPARSE_THINKING}")
    print(f"Task clarification: {ENABLE_CLARIFICATION}\n")

    agent = AIGamingAgent(
        max_steps=MAX_STEPS,
        model=MODEL,
        enable_ocr=ENABLE_OCR,
        enable_grounding_dino=ENABLE_GROUNDING_DINO,
    )
    agent.max_subtask_attempts = MAX_SUBTASK_ATTEMPTS
    agent.run(
        task=task,
        enable_sparse_thinking=ENABLE_SPARSE_THINKING,
        enable_clarification=ENABLE_CLARIFICATION,
    )


if __name__ == "__main__":
    main()
