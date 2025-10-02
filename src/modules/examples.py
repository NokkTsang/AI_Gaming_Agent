"""
Example: Custom Tasks

This file shows how to create custom tasks for the agent.
Run from project root: python -m src.modules.examples
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from modules.main import SimpleAgent

load_dotenv()


def example_google_search():
    """Example: Search Google for a topic."""
    task = "Open a web browser, go to Google, and search for 'openai'"
    agent = SimpleAgent(task=task, max_steps=10)
    agent.run()


def example_open_application():
    """Example: Open a specific application."""
    task = "Open the Calculator application on macOS using Spotlight (Cmd+Space)"
    agent = SimpleAgent(task=task, max_steps=5)
    agent.run()


def example_file_operations():
    """Example: Create a new text file."""
    task = "Open TextEdit, create a new document, and type 'Hello World'"
    agent = SimpleAgent(task=task, max_steps=8)
    agent.run()


def example_web_navigation():
    """Example: Navigate to a specific website."""
    task = "Open a web browser and navigate to https://github.com"
    agent = SimpleAgent(task=task, max_steps=6)
    agent.run()


def custom_task(task_description: str, max_steps: int = 10):
    """Run a custom task."""
    agent = SimpleAgent(task=task_description, max_steps=max_steps)
    agent.run()


if __name__ == "__main__":
    print("AI Gaming Agent - Task Examples")
    print("=" * 60)
    print("Available examples:")
    print("1. Google search (default)")
    print("2. Open Calculator")
    print("3. TextEdit file creation")
    print("4. Web navigation")
    print("5. Custom task")
    print("=" * 60)

    choice = input("Select example (1-5): ").strip()

    if choice == "1":
        example_google_search()
    elif choice == "2":
        example_open_application()
    elif choice == "3":
        example_file_operations()
    elif choice == "4":
        example_web_navigation()
    elif choice == "5":
        custom = input("Enter your custom task: ").strip()
        steps = input("Max steps (default 10): ").strip()
        steps = int(steps) if steps else 10
        custom_task(custom, steps)
    else:
        print("Invalid choice. Running default (Google search)...")
        example_google_search()
