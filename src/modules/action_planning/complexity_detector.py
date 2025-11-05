"""
Complexity Detector - Determines if deep reasoning is needed (Game-TARS Sparse Thinking).

Implements Game-TARS sparse thinking strategy (§2.2, §3.3):
- Identifies when deep reasoning is critical vs when reactive action suffices
- Uses heuristics + lightweight LLM for fast classification
- Enables 3-4x speedup by skipping unnecessary vision/planning calls
"""

import os
from typing import Dict, List, Tuple, Optional
from openai import OpenAI


class ComplexityDetector:
    """
    Determines if current state needs deep reasoning or reactive action.

    Based on Game-TARS sparse thinking: only trigger expensive reasoning
    at critical decision points (new areas, forks, failures, strategic moments).
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize complexity detector.

        Args:
            model: Lightweight model for fast checks (default: gpt-4o-mini)
            api_key: OpenAI API key
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def needs_thinking(
        self,
        current_observation: str,
        last_observation: str,
        current_subtask: str,
        last_action: Optional[Dict],
        action_history: List[Dict],
        force_thinking: bool = False,
    ) -> Tuple[bool, str]:
        """
        Decide if deep reasoning is needed for current state.

        Args:
            current_observation: Current screen description
            last_observation: Previous screen description
            current_subtask: Current subtask being attempted
            last_action: Last action taken (or None)
            action_history: Full action history
            force_thinking: Override heuristics and force deep thinking

        Returns:
            (needs_thinking: bool, reason: str)
        """
        if force_thinking:
            return True, "Forced deep thinking"

        # Heuristic 1: First action always needs thinking
        if len(action_history) == 0:
            return True, "First action of task"

        # Heuristic 2: Last action failed → need thinking
        if last_action and last_action.get("failed", False):
            return True, "Previous action failed, need new strategy"

        # Heuristic 3: Screen significantly changed (new elements)
        if self._detect_significant_change(current_observation, last_observation):
            return True, "Significant screen change detected (new UI elements)"

        # Heuristic 4: Keywords indicating complexity
        complex_keywords = [
            "multiple options",
            "choose",
            "decide",
            "fork",
            "junction",
            "new area",
            "different screen",
            "menu",
            "selection",
            "dialog",
            "which",
            "or",
            "strategy",
            "plan",
            "puzzle",
            "error",
            "warning",
        ]
        if any(kw in current_observation.lower() for kw in complex_keywords):
            keyword = next(
                kw for kw in complex_keywords if kw in current_observation.lower()
            )
            return True, f"Complex decision keyword detected: '{keyword}'"

        # Heuristic 5: Repetitive action pattern (can continue reactively)
        if self._is_repetitive_pattern(action_history, window=5):
            return False, "Continuing repetitive action sequence (reactive mode)"

        # Heuristic 6: Simple continuation actions
        if last_action and last_action.get("action_type") in ["hotkey", "type", "wait"]:
            if len(action_history) >= 2 and action_history[-2].get(
                "action_type"
            ) == last_action.get("action_type"):
                return False, "Simple continuation of keyboard/typing actions"

        # Default: Use lightweight LLM check (fast, ~1s)
        return self._llm_complexity_check(current_observation, current_subtask)

    def _detect_significant_change(self, current: str, previous: str) -> bool:
        """
        Check if significant new content appeared (indicating new UI state).

        Uses simple word-level diff heuristic.
        """
        if not previous:
            return False

        curr_words = set(current.lower().split())
        prev_words = set(previous.lower().split())
        new_words = curr_words - prev_words

        # If >30% words are new, consider it significant change
        if len(curr_words) == 0:
            return False

        change_ratio = len(new_words) / len(curr_words)
        return change_ratio > 0.3

    def _is_repetitive_pattern(self, history: List[Dict], window: int = 5) -> bool:
        """
        Check if last N actions are repetitive (e.g., moving in same direction).

        Repetitive patterns can use reactive policy without deep thinking.
        """
        if len(history) < window:
            return False

        recent = history[-window:]
        action_types = [a.get("action_type") for a in recent]

        # If all same action type and it's a simple action, it's repetitive
        if len(set(action_types)) == 1:
            if action_types[0] in ["hotkey", "move", "type", "wait"]:
                return True

        # If alternating between two simple actions (e.g., up-right-up-right)
        if len(set(action_types)) == 2:
            if all(at in ["hotkey", "move"] for at in action_types):
                return True

        return False

    def _llm_complexity_check(self, observation: str, subtask: str) -> Tuple[bool, str]:
        """
        Use lightweight LLM to judge complexity.

        Fast check (~1-2s) using gpt-4o-mini.
        """
        prompt = f"""Task: {subtask}
Current state: {observation[:400]}

Is this a SIMPLE/REPETITIVE action or COMPLEX/STRATEGIC decision?

SIMPLE examples:
- Continue moving in same direction
- Repeat same action
- Obvious next step (no choices)
- Familiar screen state

COMPLEX examples:
- Choose between multiple options
- New situation requiring analysis
- Strategic decision point
- Puzzle or problem to solve
- Multiple valid paths

Answer with ONE word: SIMPLE or COMPLEX"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip().upper()

            if "SIMPLE" in result:
                return False, "LLM judged as simple/repetitive"
            else:
                return True, "LLM judged as complex/strategic"

        except Exception as e:
            # On error, default to thinking (safe choice)
            return True, f"LLM check failed: {e}, defaulting to deep thinking"

    def get_thinking_stats(self, action_history: List[Dict]) -> Dict:
        """
        Get statistics about thinking vs reactive actions.

        Useful for analyzing sparse thinking effectiveness.
        """
        if not action_history:
            return {"total": 0, "thinking": 0, "reactive": 0, "thinking_rate": 0.0}

        thinking_count = sum(1 for a in action_history if a.get("thought"))
        reactive_count = len(action_history) - thinking_count

        return {
            "total": len(action_history),
            "thinking": thinking_count,
            "reactive": reactive_count,
            "thinking_rate": (
                thinking_count / len(action_history) if action_history else 0.0
            ),
        }
