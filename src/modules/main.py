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

Describe the screen in quadrants:
- Top-left [0-0.5, 0-0.5]
- Top-right [0.5-1, 0-0.5]  
- Bottom-left [0-0.5, 0.5-1]
- Bottom-right [0.5-1, 0.5-1]

For each UI element, note:
- Type (button/text field/link/icon)
- Position (top/middle/bottom, left/center/right)
- Approximate coordinates [x, y] in range [0,1]

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

                # Record observation before action
                observation_before = current_observation

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
                screenshot_path = take_screenshot()
                observation_prompt = (
                    self.vision_system_prompt
                    + f"\n\nPrevious state: {observation_before[:100]}...\n"
                    + "Describe what changed after the action."
                )
                observation_after = analyze_screenshot(
                    screenshot_path, observation_prompt
                )
                self.short_term.add_observation(observation_after)
                print(f"  New state: {observation_after[:150]}...\n")

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
    print(f"Max steps per task: {MAX_STEPS}")

    agent = AIGamingAgent(max_steps=MAX_STEPS, model=MODEL)
    agent.run(task)


if __name__ == "__main__":
    main()
