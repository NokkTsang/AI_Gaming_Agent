"""
AI Gaming Agent - Main Entry Point

A minimal agent that demonstrates the core loop:
Screen Input → Information Gathering → Action Planning → UI Automation

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
from modules.ui_automation.automator import UIAutomator

# Load environment variables
load_dotenv()


class SimpleAgent:
    """A simple agent that can execute tasks by observing and acting on the screen."""

    def __init__(self, task: str, max_steps: int = 10):
        self.task = task
        self.max_steps = max_steps
        self.action_history = []
        self.observation_history = []

        # Initialize modules
        self.automator = UIAutomator()
        self.executor = ActionExecutor(
            screen_width=1920,  # Will be updated dynamically
            screen_height=1080,
            automator=self.automator,
        )
        self.planner = ActionPlanner(
            model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")
        )

    def update_screen_size(self):
        """Dynamically update screen size for executor."""
        import pyautogui

        w, h = pyautogui.size()
        self.executor.screen_width = int(w)
        self.executor.screen_height = int(h)

    def run(self):
        """Execute the agent loop."""
        print(f"🤖 Starting Agent")
        print(f"📋 Task: {self.task}")
        print(f"🔄 Max steps: {self.max_steps}\n")

        self.update_screen_size()

        for step in range(self.max_steps):
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{self.max_steps}")
            print(f"{'='*60}\n")

            # 1. Screen Input - Capture screenshot
            print("📸 Capturing screenshot...")
            screenshot_path = take_screenshot()
            print(f"   Saved to: {screenshot_path}")

            # 2. Information Gathering - Analyze screen
            print("\n🔍 Analyzing screen...")
            observation_prompt = (
                f"Current task: {self.task}\n"
                f"Action history: {self._format_history()}\n\n"
                "Describe what you see on the screen. "
                "Focus on elements relevant to the task (search bars, buttons, text fields, etc.)."
            )
            observation = analyze_screenshot(screenshot_path, observation_prompt)
            print(f"   Observation: {observation[:200]}...")
            self.observation_history.append(observation)

            # 3. Action Planning - Decide next action
            print("\n🧠 Planning next action...")
            action_dict = self.planner.plan_next_action(
                task=self.task,
                observation=observation,
                action_history=self.action_history,
            )

            if action_dict is None:
                print("   ❌ Failed to generate action. Stopping.")
                break

            print(f"   Action: {action_dict['action_type']}")
            print(f"   Inputs: {action_dict['action_inputs']}")

            # Check if task is complete
            if action_dict["action_type"] == "finished":
                print("\n✅ Task completed!")
                print(
                    f"   Final message: {action_dict['action_inputs'].get('content', 'Done')}"
                )
                break

            # 4. UI Automation - Execute action
            print("\n⚡ Executing action...")
            try:
                self.executor.execute(action_dict)
                self.action_history.append(action_dict)
                print("   ✓ Action executed successfully")
            except Exception as e:
                print(f"   ❌ Action execution failed: {e}")
                # Continue anyway to avoid getting stuck

            # Wait before next observation
            time.sleep(2)

        print("\n" + "=" * 60)
        print("🏁 Agent finished")
        print(f"   Total steps: {len(self.action_history)}")
        print("=" * 60)

    def _format_history(self) -> str:
        """Format action history for context."""
        if not self.action_history:
            return "No actions yet"

        recent = self.action_history[-3:]  # Last 3 actions
        formatted = []
        for i, action in enumerate(recent, 1):
            formatted.append(f"{i}. {action['action_type']}({action['action_inputs']})")
        return "; ".join(formatted)


def main():
    """Main entry point."""
    # Example task: Search Google for "openai"
    task = "Open a web browser, go to Google, and search for 'openai'"

    agent = SimpleAgent(task=task, max_steps=10)
    agent.run()


if __name__ == "__main__":
    main()
