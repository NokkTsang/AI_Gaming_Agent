"""
Self-reflection module for judging action success/failure.
Uses LLM to compare before/after observations and determine outcome.
"""

import os
from typing import Dict, Tuple
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
