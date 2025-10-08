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

# Add src directory to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from modules.screen_input.screen_capture import take_screenshot
from modules.information_gathering.info_gather import analyze_screenshot
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

    def __init__(self, max_steps: int = 50, model: str = "gpt-4o-mini"):
        """
        Initialize the full agent with all components.

        Args:
            max_steps: Maximum steps per task (default: 50)
            model: OpenAI model for planning and reasoning
        """
        self.max_steps = max_steps
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")

        # Vision system prompt (used for all screenshot analysis)
        self.vision_system_prompt = """You are a GUI automation agent. Analyze this screenshot carefully.

Use the 3x3 GRID REFERENCE provided to describe UI element locations.
Reference grid cells (A1, A2, A3, B1, B2, B3, C1, C2, C3) when describing positions.

Use DETECTED TEXT ELEMENTS (if provided) for precise coordinates of text-based UI.

IMPORTANT: In games and applications, clickable elements are often VISUAL (icons, buttons, flags, markers).
Text labels may appear ABOVE or NEAR the clickable element. Look for:
- Buttons with icons or graphics
- Flags, markers, or visual indicators
- The clickable area may be BELOW or ADJACENT to text labels

For each UI element, note:
- Type (button/text field/link/icon/flag/marker)
- Visual appearance (color, shape, icon type)
- Grid cell location (e.g., "in A2" or "at B3")
- Approximate coordinates [x, y] in range [0,1] of the CENTER of the clickable area
- Text content if visible and its position relative to the clickable element

Be precise about locations for accurate clicking."""

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

        # Reasoning modules
        print("  Initializing reasoning modules...")
        self.task_breaker = TaskBreaker(api_key=api_key)
        self.reflector = Reflector(api_key=api_key)
        self.skill_manager = SkillManager(api_key=api_key)

        # Action planner
        self.planner = ActionPlanner(model=model, api_key=api_key)

        print("✓ Agent initialized successfully!\n")

    def update_screen_size(self):
        """Dynamically update screen size for executor."""
        import pyautogui

        w, h = pyautogui.size()
        self.executor.screen_width = int(w)
        self.executor.screen_height = int(h)

    def compare_screenshots(
        self, img1_path: str, img2_path: str, threshold: float = 0.005
    ) -> bool:
        """Compare two screenshots to detect if a change occurred.

        Args:
            img1_path: Path to first screenshot
            img2_path: Path to second screenshot
            threshold: Minimum change ratio to consider different (default 2%)

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
        print(f"\n{'='*80}")
        print(f"Task: {task}")
        print(f"{'='*80}\n")

        self.update_screen_size()

        try:
            # Step 1: Capture initial screen
            print("Step 1: Capturing initial screen...")
            screenshot_path = take_screenshot()
            initial_observation = analyze_screenshot(
                screenshot_path, self.vision_system_prompt
            )
            print(f"  Initial state: {initial_observation[:150]}...\n")

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

                # Force skip if stuck too long on same subtask
                if subtask_attempts[current_subtask] > 5:
                    print(
                        f"  Warning: Stuck on subtask for {subtask_attempts[current_subtask]} iterations"
                    )
                    print("  Suggesting to skip or try alternative approach...\n")

                    skip_prompt = (
                        f"You've attempted this subtask {subtask_attempts[current_subtask]} times without success: {current_subtask}\n\n"
                        + "Options:\n"
                        + "1. Try a completely different approach (different location, different action type)\n"
                        + "2. Skip this subtask if it's blocking progress\n"
                        + "3. Break it into smaller sub-steps\n\n"
                        + "What should we do? Provide a new strategy."
                    )

                    # Add this feedback to observation
                    self.short_term.add_observation(
                        f"STUCK ALERT: Attempted '{current_subtask}' {subtask_attempts[current_subtask]} times. "
                        + "Need alternative strategy or should skip this step."
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
                action_dict = self.planner.plan_next_action(
                    task=task,
                    observation=current_observation,
                    action_history=context["recent_actions"],
                )

                if action_dict is None:
                    print("  ⨯ Failed to generate action")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("\n⨯ Too many failures. Stopping.")
                        break
                    continue

                print(f"  Action: {action_dict['action_type']}")
                print(f"  Inputs: {action_dict['action_inputs']}\n")

                # Check if finished
                if action_dict["action_type"] == "finished":
                    print("✓ Task completed!")
                    break

                # Record observation and screenshot before action
                observation_before = current_observation
                screenshot_before = screenshot_path

                # Step 6: Execute action
                print("  Executing action...")
                try:
                    self.executor.execute(action_dict)
                    self.short_term.add_action(action_dict)  # Store as dict, not string
                    print("  ✓ Action executed\n")
                except Exception as e:
                    print(f"  ⨯ Execution failed: {e}\n")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        break
                    continue

                # Wait for UI to update
                time.sleep(0.5)

                # Step 7: Capture screen after action
                print("  Capturing result...")
                screenshot_after = take_screenshot()

                # Check if screen changed (self-correction mechanism)
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

                    # Ask LLM to provide corrected coordinates with object detection
                    correction_prompt = (
                        self.vision_system_prompt
                        + f"\n\nPREVIOUS FAILED ATTEMPT:\n"
                        + f"- Clicked at: {action_dict['action_inputs'].get('start_box', 'unknown')}\n"
                        + f"- Goal: {action_dict.get('thought', 'click action')}\n"
                        + f"- Result: Nothing happened, screen unchanged\n\n"
                        + "CRITICAL: The click target may be a VISUAL ELEMENT (icon/button/flag) NEAR the text, not the text itself.\n"
                        + "Look for clickable UI elements like buttons, icons, or markers that are ADJACENT to or BELOW relevant text.\n\n"
                        + "TASK: Find the EXACT clickable element:\n"
                        + "1. Identify if there's a visual button/icon/flag near the text (describe color, shape, position relative to text)\n"
                        + "2. Which grid cell is the CLICKABLE element in (A1, A2, A3, B1, B2, B3, C1, C2, C3)\n"
                        + "3. EXACT coordinates of the CENTER of the clickable element as [x, y] where x and y are between 0 and 1\n\n"
                        + "Format your response as:\n"
                        + "ELEMENT: [visual description of clickable item]\n"
                        + "GRID: [cell]\n"
                        + "COORDINATES: [x, y]"
                    )

                    # Use object detection if we know what to look for
                    correction_observation = analyze_screenshot(
                        screenshot_after,
                        correction_prompt,
                        detect_objects=target_objects,
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
                observation_prompt = (
                    self.vision_system_prompt
                    + f"\n\nPrevious state: {observation_before[:100]}...\n"
                    + "Describe what changed after the action."
                )
                observation_after = analyze_screenshot(
                    screenshot_after, observation_prompt
                )
                self.short_term.add_observation(observation_after)
                print(f"  New state: {observation_after[:150]}...\n")

                # Reset consecutive failures on successful action
                consecutive_failures = 0

                # Step 8: Self-reflection
                print("  Reflecting on outcome...")
                success, reasoning = self.reflector.judge_action_success(
                    task_goal=task,
                    current_subtask=current_subtask,
                    action_taken=f"{action_dict['action_type']}({action_dict['action_inputs']})",
                    observation_before=observation_before,
                    observation_after=observation_after,
                )

                if success:
                    print(f"  ✓ Success: {reasoning}\n")
                    consecutive_failures = 0

                    # Advance to next subtask
                    self.short_term.advance_subtask()

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
            print(f"\n{'='*80}")
            print("Session Summary")
            print(f"{'='*80}")
            print(f"Total iterations: {step}")
            print(f"Skills in database: {self.long_term.get_skill_count()}")
            print(f"Actions taken: {len(self.short_term.state['actions'])}")
            print(f"{'='*80}\n")

        except KeyboardInterrupt:
            print("\n\n! Agent interrupted by user")
        except Exception as e:
            print(f"\n\n⨯ Agent error: {e}")
            import traceback

            traceback.print_exc()


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
    # ============================================================

    print(f"Using model: {MODEL}")
    print(f"Max steps per task: {MAX_STEPS}\n")

    agent = AIGamingAgent(max_steps=MAX_STEPS, model=MODEL)
    agent.run(task)


if __name__ == "__main__":
    main()
