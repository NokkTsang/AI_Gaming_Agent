"""
Task Completion Detector - Validates if task is actually complete.

Prevents false completion signals by checking explicit success criteria
against current observations.
"""

import os
import json
from typing import Dict, List, Tuple
from openai import OpenAI


class CompletionDetector:
    """
    Validates if task is actually complete based on success criteria.

    Addresses Problem #5: Agent says "completed" but task not actually done.
    Uses explicit criteria checking + confidence scoring.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize completion detector.

        Args:
            model: LLM model for validation
            api_key: OpenAI API key
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def is_task_complete(
        self,
        instruction: Dict,
        current_observation: str,
        action_history: List[Dict],
        confidence_threshold: float = 0.8,
    ) -> Tuple[bool, str, float]:
        """
        Check if all success criteria are met.

        Args:
            instruction: Structured instruction with success criteria
            current_observation: Current screen description
            action_history: Full action history
            confidence_threshold: Minimum confidence to auto-accept (default 0.8)

        Returns:
            (is_complete: bool, reasoning: str, confidence: float)
        """
        # Extract success criteria
        success_criteria = instruction.get("success_criteria", [])
        if not success_criteria:
            # No explicit criteria, use heuristic
            return self._heuristic_completion_check(
                instruction.get("goal", ""), current_observation, action_history
            )

        # Build validation prompt
        criteria_text = "\n".join(
            [f"  {i+1}. {c}" for i, c in enumerate(success_criteria)]
        )
        recent_actions = (
            action_history[-10:] if len(action_history) > 10 else action_history
        )
        actions_text = "\n".join(
            [
                f"  {i+1}. {a.get('action_type', 'unknown')}({a.get('action_inputs', {})})"
                for i, a in enumerate(recent_actions)
            ]
        )

        prompt = f"""TASK GOAL:
{instruction.get('goal', 'Unknown goal')}

SUCCESS CRITERIA (ALL must be met):
{criteria_text}

CURRENT SCREEN:
{current_observation}

RECENT ACTIONS (last 10):
{actions_text}

Question: Are ALL success criteria met?

Analyze each criterion:
1. Check if criterion is satisfied by current observation
2. Look for explicit indicators (text, visual state, etc.)
3. Consider if actions suggest task completion

Respond in JSON:
{{
  "complete": true/false,
  "reasoning": "Brief explanation",
  "confidence": 0.0-1.0,
  "met_criteria": ["criterion 1", "criterion 2", ...],
  "unmet_criteria": ["criterion X", ...]
}}

Be CONSERVATIVE - only mark complete if you're confident ALL criteria are met."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )

            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            result = json.loads(result_text)

            is_complete = result.get("complete", False)
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "No reasoning provided")

            # If low confidence, ask user for confirmation
            if is_complete and confidence < confidence_threshold:
                print(f"\nUncertain if task complete (confidence: {confidence:.0%})")
                print(f"   Reasoning: {reasoning}")
                print(f"   Met criteria: {result.get('met_criteria', [])}")
                print(f"   Unmet criteria: {result.get('unmet_criteria', [])}")

                user_confirm = (
                    input("   Is task actually complete? (y/n): ").strip().lower()
                )
                is_complete = user_confirm == "y"

                if is_complete:
                    reasoning = f"User confirmed completion (AI confidence was {confidence:.0%})"
                else:
                    reasoning = (
                        f"User rejected completion (AI confidence was {confidence:.0%})"
                    )

            return is_complete, reasoning, confidence

        except Exception as e:
            print(f"   Error checking completion: {e}")
            # On error, be conservative - assume not complete
            return False, f"Error in completion check: {e}", 0.0

    def _heuristic_completion_check(
        self, goal: str, observation: str, action_history: List[Dict]
    ) -> Tuple[bool, str, float]:
        """
        Fallback heuristic when no explicit success criteria provided.
        """
        # Check for common completion indicators
        completion_keywords = [
            "completed",
            "success",
            "finished",
            "done",
            "won",
            "victory",
            "congratulations",
            "well done",
            "level complete",
            "mission accomplished",
        ]

        obs_lower = observation.lower()
        if any(kw in obs_lower for kw in completion_keywords):
            keyword = next(kw for kw in completion_keywords if kw in obs_lower)
            return True, f"Completion keyword detected: '{keyword}'", 0.8

        # Check if last action was "finished"
        if action_history and action_history[-1].get("action_type") == "finished":
            return True, "Agent explicitly called 'finished' action", 0.9

        # Default: not complete
        return False, "No explicit completion indicators found", 0.3

    def validate_completion_action(
        self, instruction: Dict, observation: str, action_history: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Validate if agent should be allowed to call 'finished' action.

        This is called BEFORE executing a 'finished' action to prevent
        premature completion.

        Returns:
            (allow_finish: bool, reason: str)
        """
        is_complete, reasoning, confidence = self.is_task_complete(
            instruction,
            observation,
            action_history,
            confidence_threshold=0.7,  # Slightly lower threshold for validation
        )

        if is_complete:
            return (
                True,
                f"Completion validated: {reasoning} (confidence: {confidence:.0%})",
            )
        else:
            return (
                False,
                f"Task not complete: {reasoning} (confidence: {confidence:.0%})",
            )
