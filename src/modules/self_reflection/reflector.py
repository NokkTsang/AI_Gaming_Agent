"""
Self-reflection module for judging action success/failure.
Uses LLM to compare before/after observations and determine outcome.
Enhanced with stuck detection and recovery suggestions.
"""

import os
import json
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
            print("\n" + "=" * 80)
            print("SELF-REFLECTION REQUEST")
            print("=" * 80)
            print(f"Model: {self.model}")
            print(f"\nPrompt ({len(prompt)} chars):")
            print("-" * 80)
            print(prompt)
            print("-" * 80)

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

            print("\nSELF-REFLECTION RESPONSE")
            print("=" * 80)
            print(result_text)
            print("=" * 80)
            if hasattr(response, "usage") and response.usage:
                print(
                    f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}"
                )
            print("=" * 80 + "\n")

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

    def detect_stuck_state(
        self, recent_actions: list, window_size: int = 3
    ) -> Tuple[bool, str]:
        """
        Detect if agent is stuck (repeating similar actions).

        Args:
            recent_actions: List of recent action dicts
            window_size: Number of recent actions to check

        Returns:
            Tuple of (is_stuck: bool, reason: str)
        """
        if len(recent_actions) < window_size:
            return False, "Not enough actions to determine"

        # Check last N actions
        recent = recent_actions[-window_size:]

        # Simple heuristic: if all actions are identical, agent is stuck
        if len(set(str(a) for a in recent)) == 1:
            return True, f"Repeated same action {window_size} times"

        # Check for alternating between two actions
        if window_size >= 4:
            if str(recent[-1]) == str(recent[-3]) and str(recent[-2]) == str(
                recent[-4]
            ):
                return True, "Alternating between two actions"

        return False, "Actions appear varied"

    def detect_stuck_with_recovery(
        self,
        action_history: List[Dict],
        observation_history: List[str],
        window_size: int = 5,
    ) -> Tuple[bool, str, List[Dict]]:
        """
        Enhanced stuck detection with recovery action suggestions.

        Detects:
        1. Same action repeated multiple times
        2. Screen unchanged for multiple actions
        3. Oscillating between two actions
        4. Semantic stuck patterns (via LLM)

        Args:
            action_history: Full action history
            observation_history: Full observation history
            window_size: Number of recent steps to check

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
                recovery = [
                    {
                        "action_type": "click",
                        "action_inputs": {"start_box": [0.5, 0.5]},
                    },
                    {"action_type": "hotkey", "action_inputs": {"key": "enter"}},
                ]
                return True, "Screen unchanged for 5 actions", recovery

        # Check 3: Oscillating between two actions
        if window_size >= 4:
            if action_strs[0] == action_strs[2] and action_strs[1] == action_strs[3]:
                recovery = [
                    {"action_type": "wait", "action_inputs": {}},
                    {"action_type": "hotkey", "action_inputs": {"key": "esc"}},
                ]
                return True, "Oscillating between 2 actions", recovery

        # Check 4: Use LLM for semantic stuck detection
        if len(recent_actions) >= 5:
            stuck, reason, recovery = self._llm_stuck_detection(
                recent_actions, recent_obs
            )
            if stuck:
                return stuck, reason, recovery

        return False, "Agent progressing normally", []

    def _llm_stuck_detection(
        self, recent_actions: List[Dict], recent_obs: List[str]
    ) -> Tuple[bool, str, List[Dict]]:
        """
        Use LLM to detect semantic stuck patterns.
        """
        actions_text = "\n".join(
            [
                f"{i+1}. {a.get('action_type')}({a.get('action_inputs', {})})"
                for i, a in enumerate(recent_actions)
            ]
        )

        obs_text = "\n".join(
            [
                f"{i+1}. {obs[:150]}..."
                for i, obs in enumerate(recent_obs[:3])  # Show first 3
            ]
        )

        prompt = f"""Recent actions (last 5):
{actions_text}

Recent observations (first 3):
{obs_text}

Is the agent STUCK in an ineffective loop?

STUCK indicators:
- Repeating same failed action
- Not making progress toward goal
- Clicking same unresponsive element
- Going in circles

PROGRESSING indicators:
- Actions are varied
- Observations are changing
- Moving toward goal

Respond in JSON:
{{
  "stuck": true/false,
  "reason": "brief explanation",
  "recovery_action": "suggested action type (wait/esc/enter/click_different)"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )

            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(result_text)

            if result.get("stuck", False):
                recovery_action_type = result.get("recovery_action", "wait")

                # Map recovery action to actual action dict
                recovery_map = {
                    "wait": {"action_type": "wait", "action_inputs": {}},
                    "esc": {"action_type": "hotkey", "action_inputs": {"key": "esc"}},
                    "enter": {
                        "action_type": "hotkey",
                        "action_inputs": {"key": "enter"},
                    },
                    "click_different": {
                        "action_type": "click",
                        "action_inputs": {"start_box": [0.3, 0.3]},
                    },
                }

                recovery = [
                    recovery_map.get(recovery_action_type, recovery_map["wait"])
                ]
                return (
                    True,
                    result.get("reason", "LLM detected stuck pattern"),
                    recovery,
                )

            return False, "LLM: Agent progressing", []

        except Exception as e:
            # On error, assume not stuck
            return False, f"LLM check failed: {e}", []
