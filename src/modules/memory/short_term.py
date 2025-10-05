"""
Short-term memory module for tracking current task state.
Stores task goal, current subtask, observation history, and action history.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class TaskState:
    """Manages short-term memory for current task execution."""

    def __init__(self, state_file: str = "short_term_state.json"):
        """
        Initialize task state manager.

        Args:
            state_file: Path to JSON file for persisting state
        """
        self.state_file = os.path.join(os.path.dirname(__file__), "data", state_file)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from JSON file or create empty state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return self._create_empty_state()

    def _create_empty_state(self) -> Dict:
        """Create empty state structure."""
        return {
            "task_goal": "",
            "subtasks": [],
            "current_subtask_index": 0,
            "observations": [],
            "actions": [],
            "start_time": None,
            "last_update": None,
        }

    def _save_state(self):
        """Persist current state to JSON file."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def initialize_task(self, task_goal: str, subtasks: List[str]):
        """
        Initialize new task with goal and subtask breakdown.

        Args:
            task_goal: High-level task description
            subtasks: List of subtask descriptions
        """
        self.state = self._create_empty_state()
        self.state["task_goal"] = task_goal
        self.state["subtasks"] = subtasks
        self.state["start_time"] = datetime.now().isoformat()
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def add_observation(self, observation: str):
        """
        Add new observation from screen analysis.

        Args:
            observation: Text description of current screen state
        """
        self.state["observations"].append(
            {"timestamp": datetime.now().isoformat(), "content": observation}
        )
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def add_action(self, action: Dict):
        """
        Add executed action to history.

        Args:
            action: Action dictionary with tool name and parameters
        """
        self.state["actions"].append(
            {"timestamp": datetime.now().isoformat(), "action": action}
        )
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def get_current_subtask(self) -> Optional[str]:
        """
        Get current subtask description.

        Returns:
            Current subtask string or None if task complete
        """
        idx = self.state["current_subtask_index"]
        if idx < len(self.state["subtasks"]):
            return self.state["subtasks"][idx]
        return None

    def advance_subtask(self):
        """Move to next subtask."""
        self.state["current_subtask_index"] += 1
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def update_subtasks(self, new_subtasks: List[str]):
        """
        Replace remaining subtasks (used when task decomposition adjusts).

        Args:
            new_subtasks: New list of subtasks
        """
        self.state["subtasks"] = new_subtasks
        self.state["current_subtask_index"] = 0
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def get_recent_context(self, n: int = 3) -> Dict:
        """
        Get recent observations and actions for context.

        Args:
            n: Number of recent items to retrieve

        Returns:
            Dict with recent observations and actions
        """
        return {
            "recent_observations": self.state["observations"][-n:],
            "recent_actions": self.state["actions"][-n:],
            "current_subtask": self.get_current_subtask(),
        }

    def clear_state(self):
        """Clear all state (for new task)."""
        self.state = self._create_empty_state()
        self._save_state()

    def get_full_state(self) -> Dict:
        """Get complete state dictionary."""
        return self.state.copy()
