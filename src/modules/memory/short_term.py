"""
Short-term memory module for tracking current task state.
Implements Game-TARS two-tier memory architecture:
- Tier 1: Short-term working memory (last 80 steps, full detail)
- Tier 2: Long-term summary memory (2400+ steps, compressed thoughts only)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class TaskState:
    """
    Manages short-term memory for current task execution.

    Implements two-tier memory system from Game-TARS:
    - Context memory: 80 recent steps with full observations, thoughts, actions
    - Summary memory: 2400+ compressed steps with thoughts only (observations dropped)
    """

    def __init__(
        self,
        state_file: str = "short_term_state.json",
        max_context_steps: int = 80,
        max_summary_steps: int = 2400,
    ):
        """
        Initialize task state manager with two-tier memory.

        Args:
            state_file: Path to JSON file for persisting state
            max_context_steps: Maximum steps in Tier 1 (full detail)
            max_summary_steps: Maximum steps in Tier 2 (compressed)
        """
        self.state_file = os.path.join(os.path.dirname(__file__), "data", state_file)
        self.max_context_steps = max_context_steps
        self.max_summary_steps = max_summary_steps
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from JSON file or create empty state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return self._create_empty_state()

    def _create_empty_state(self) -> Dict:
        """Create empty state structure with two-tier memory."""
        return {
            "task_goal": "",
            "subtasks": [],
            "current_subtask_index": 0,
            # Two-tier memory system
            "context_memory": [],  # Tier 1: Last N steps (full detail)
            "summary_memory": [],  # Tier 2: Compressed history (thoughts only)
            # Legacy fields (kept for compatibility)
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

    def add_step(
        self,
        observation: str,
        thought: Optional[str],
        action: Dict,
        compress: bool = True,
    ):
        """
        Add new step to two-tier memory with automatic compression.

        This is the main method for Game-TARS style memory management.

        Args:
            observation: Screen observation (full detail)
            thought: LLM's reasoning (None for reactive actions)
            action: Action dictionary
            compress: Whether to compress old steps (default True)
        """
        step = {
            "observation": observation,
            "thought": thought if thought else "",
            "action": action,
            "timestamp": datetime.now().isoformat(),
        }

        # Add to Tier 1: Context memory
        self.state["context_memory"].append(step)

        # If context exceeds limit, compress oldest step to Tier 2
        if compress and len(self.state["context_memory"]) > self.max_context_steps:
            evicted = self.state["context_memory"].pop(0)  # Remove oldest

            # Compress: Keep only thought (drop observation and action for token savings)
            if evicted["thought"]:  # Only store if there was reasoning
                compressed = {
                    "thought": evicted["thought"],
                    "timestamp": evicted["timestamp"],
                }
                self.state["summary_memory"].append(compressed)

                # Limit summary memory too
                if len(self.state["summary_memory"]) > self.max_summary_steps:
                    self.state["summary_memory"].pop(0)

        # Also update legacy fields for backward compatibility
        self.state["observations"].append(
            {"timestamp": datetime.now().isoformat(), "content": observation}
        )
        self.state["actions"].append(
            {"timestamp": datetime.now().isoformat(), "action": action}
        )

        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def add_observation(self, observation: str):
        """
        Add new observation (legacy method for compatibility).

        Note: For Game-TARS memory, use add_step() instead.
        """
        self.state["observations"].append(
            {"timestamp": datetime.now().isoformat(), "content": observation}
        )
        self.state["last_update"] = datetime.now().isoformat()
        self._save_state()

    def add_action(self, action: Dict):
        """
        Add executed action (legacy method for compatibility).

        Note: For Game-TARS memory, use add_step() instead.
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

    def get_memory_for_context(
        self, recent_limit: int = 10, summary_limit: int = 50
    ) -> str:
        """
        Format two-tier memory for LLM context (Game-TARS style).

        Args:
            recent_limit: Number of recent context steps to include in full detail
            summary_limit: Number of summary steps to include

        Returns:
            Formatted string with compressed history + recent context
        """
        parts = []

        # Tier 2: Historical summary (if any)
        if self.state["summary_memory"]:
            summary_steps = self.state["summary_memory"][-summary_limit:]
            if summary_steps:
                parts.append("=== HISTORICAL CONTEXT (Compressed) ===")
                parts.append(f"[{len(summary_steps)} earlier steps summarized]")
                for i, step in enumerate(
                    summary_steps[-10:], 1
                ):  # Show last 10 summaries
                    parts.append(f"{i}. {step['thought']}")
                parts.append("")

        # Tier 1: Recent context (full detail)
        if self.state["context_memory"]:
            recent_steps = self.state["context_memory"][-recent_limit:]
            parts.append("=== RECENT MEMORY (Full Detail) ===")
            for i, step in enumerate(recent_steps, 1):
                parts.append(f"\n--- Step {i} ---")
                parts.append(f"Action: {step['action'].get('action_type', 'unknown')}")
                if step["thought"]:
                    parts.append(f"Thought: {step['thought']}")
                parts.append(f"Result: {step['observation'][:200]}...")

        return "\n".join(parts) if parts else "No memory yet."

    def get_memory_stats(self) -> Dict:
        """Get statistics about memory usage."""
        return {
            "context_steps": len(self.state.get("context_memory", [])),
            "summary_steps": len(self.state.get("summary_memory", [])),
            "total_steps": len(self.state.get("context_memory", []))
            + len(self.state.get("summary_memory", [])),
            "context_limit": self.max_context_steps,
            "summary_limit": self.max_summary_steps,
            "context_usage_pct": len(self.state.get("context_memory", []))
            / self.max_context_steps
            * 100,
        }

    def get_full_state(self) -> Dict:
        """Get complete state dictionary."""
        return self.state.copy()
