"""
Self-reflection module for judging action success/failure.
Uses LLM to compare before/after observations and determine outcome.
"""

import os
from typing import Dict, Tuple, List
from openai import OpenAI


class Reflector:
    """Judges action success by analyzing observation changes."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize reflector with OpenAI API.

        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: OpenAI model to use for reflection
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def judge_action_success(
        self,
        task_goal: str,
        current_subtask: str,
        action_taken: Dict,
        observation_before: str,
        observation_after: str,
    ) -> Tuple[bool, str]:
        """
        Judge if action successfully progressed toward subtask goal.

        Args:
            task_goal: Overall task description
            current_subtask: Current subtask being attempted
            action_taken: Action dict that was executed
            observation_before: Screen state before action
            observation_after: Screen state after action

        Returns:
            Tuple of (success: bool, reasoning: str)
        """
        prompt = self._create_judgment_prompt(
            task_goal,
            current_subtask,
            action_taken,
            observation_before,
            observation_after,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at judging whether UI actions successfully progressed toward a goal. Respond with 'SUCCESS' or 'FAILURE' followed by brief reasoning.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )

            result_text = response.choices[0].message.content.strip()

            # Log tokens
            if hasattr(response, "usage") and response.usage:
                print(f"     [Reflection] Tokens: {response.usage.total_tokens}")

            # Parse response
            if result_text.startswith("SUCCESS"):
                return True, result_text.replace("SUCCESS", "").strip(": ")
            else:
                return False, result_text.replace("FAILURE", "").strip(": ")

        except Exception as e:
            # On error, assume failure and return error message
            return False, f"Reflection error: {str(e)}"

    def _create_judgment_prompt(
        self,
        task_goal: str,
        current_subtask: str,
        action_taken: Dict,
        observation_before: str,
        observation_after: str,
    ) -> str:
        """Create prompt for LLM judgment."""
        return f"""Task Goal: {task_goal}
Current Subtask: {current_subtask}

Action Taken:
{action_taken}

Observation BEFORE action:
{observation_before}

Observation AFTER action:
{observation_after}

Did the action successfully progress toward completing the subtask?
Consider:
- Did the UI change as expected?
- Is the system state closer to the subtask goal?
- Are there error messages or unexpected states?
- Did screen reset to start/menu? (constraint violation)

Respond with 'SUCCESS: [reason]' or 'FAILURE: [reason]'."""

    def detect_stuck_with_recovery(
        self,
        action_history: List[Dict],
        observation_history: List[str],
        window_size: int = 5,
    ) -> Tuple[bool, str, List[Dict]]:
        """
        Stuck detection with recovery action suggestions using heuristics.

        Detects:
        1. Same action repeated multiple times
        2. Screen unchanged for multiple actions
        3. Oscillating between two actions

        Returns:
            (is_stuck: bool, reason: str, recovery_actions: List[Dict])
        """
        if len(action_history) < window_size:
            return False, "Not enough history", []

        recent_actions = action_history[-window_size:]
        recent_obs = (
            observation_history[-window_size:]
            if len(observation_history) >= window_size
            else observation_history
        )

        # Check 1: Same action repeated
        action_strs = [str(a) for a in recent_actions]
        if len(set(action_strs)) == 1:
            recovery = [
                {"action_type": "wait", "action_inputs": {}},
                {"action_type": "hotkey", "action_inputs": {"key": "esc"}},
            ]
            return True, f"Repeated same action {window_size}x", recovery

        # Check 2: Screen unchanged (compare first 200 chars)
        if recent_obs:
            obs_samples = [obs[:200] for obs in recent_obs]
            if len(set(obs_samples)) == 1:
                # Recovery: Try OPPOSITE direction to backtrack (maze strategy)
                # Look at recent actions to find the most common direction
                recent_keys = [
                    a.get("action_inputs", {}).get("key")
                    for a in recent_actions
                    if a.get("action_type") == "hotkey"
                ]

                # Count direction frequencies in recent actions
                direction_counts = {"up": 0, "down": 0, "left": 0, "right": 0}
                for key in recent_keys:
                    if key in direction_counts:
                        direction_counts[key] += 1

                # Find most used direction and suggest its OPPOSITE (backtracking)
                opposite_map = {
                    "up": "down",
                    "down": "up",
                    "left": "right",
                    "right": "left",
                }
                most_used = max(direction_counts, key=direction_counts.get)

                # If a direction was clearly dominant, try opposite (backtrack)
                if direction_counts[most_used] >= 2:
                    backtrack_dir = opposite_map[most_used]
                    recovery = [
                        {"action_type": "wait", "action_inputs": {"duration": 0.3}},
                        {
                            "action_type": "hotkey",
                            "action_inputs": {"key": backtrack_dir},
                        },
                        {
                            "action_type": "hotkey",
                            "action_inputs": {"key": backtrack_dir},
                        },
                    ]
                    return (
                        True,
                        f"Screen unchanged - backtracking (opposite of {most_used})",
                        recovery,
                    )
                else:
                    # No dominant direction, try all unused directions
                    all_directions = [
                        "down",
                        "left",
                        "up",
                        "right",
                    ]  # Reordered for maze priority
                    tried_dirs = set(recent_keys)
                    untried = [d for d in all_directions if d not in tried_dirs]

                    if untried:
                        recovery = [
                            {"action_type": "wait", "action_inputs": {"duration": 0.3}},
                            {
                                "action_type": "hotkey",
                                "action_inputs": {"key": untried[0]},
                            },
                        ]
                    else:
                        # All tried, do a longer backtrack sequence
                        recovery = [
                            {"action_type": "wait", "action_inputs": {"duration": 0.5}},
                            {"action_type": "hotkey", "action_inputs": {"key": "left"}},
                            {"action_type": "hotkey", "action_inputs": {"key": "down"}},
                        ]
                    return True, "Screen unchanged - trying alternate path", recovery

        # Check 3: Oscillating between two actions
        if window_size >= 4:
            if action_strs[0] == action_strs[2] and action_strs[1] == action_strs[3]:
                recovery = [
                    {"action_type": "wait", "action_inputs": {}},
                    {"action_type": "hotkey", "action_inputs": {"key": "esc"}},
                ]
                return True, "Oscillating between 2 actions", recovery

        return False, "Agent progressing normally", []
