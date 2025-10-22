"""
Action Planner - Uses LLM to decide next action based on task and observation.
"""

import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI


class ActionPlanner:
    """Plans actions using a vision-capable LLM based on task and screen observations."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines the agent\'s action space."""
        return """You are a GUI automation agent. Decide the next action based on task and screen observation.

## Actions
1. click(start_box=[x, y]) - Click at normalized [0,1] coordinates
2. double_click(start_box=[x, y])
3. right_click(start_box=[x, y])
4. drag(start_box=[x1, y1], end_box=[x2, y2])
5. type(content="text") - Use \\n for Enter
6. hotkey(key="cmd c") - Space-separated keys
7. scroll(direction="up"/"down", start_box=[x, y])
8. wait() - Wait 5 seconds
9. finished(content="summary") - Task complete

## Coordinates
Normalized [0,1]: [0,0]=top-left, [1,1]=bottom-right, [0.5,0.5]=center

## Correction Feedback
If "CLICK FAILED" with "COORDINATES: [x, y]" → USE those EXACT coordinates (vision analysis correction)

## Failure Recovery (after 2+ failed clicks)
1. Try typing (element may be focused)
2. Try hotkeys (cmd+l, tab, enter, etc.)
3. Try different interaction (double-click, different position)
4. Skip after 3 failed attempts

NEVER repeat same failed action >2 times.

## Output
Valid JSON only:
```json
{"thought": "reasoning", "action_type": "click", "action_inputs": {"start_box": [0.5, 0.8]}}
```

## Examples
Click: {"thought": "Clicking START button", "action_type": "click", "action_inputs": {"start_box": [0.5, 0.72]}}
Finished: {"thought": "Game loaded successfully", "action_type": "finished", "action_inputs": {"content": "Started Kingdom Rush"}}

Return JSON only, no markdown.
"""

    def plan_next_action(
        self, task: str, observation: str, action_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Plan the next action based on task, current observation, and history.

        Args:
            task: The overall goal/task to accomplish
            observation: Current screen description from vision LLM
            action_history: List of previous actions taken

        Returns:
            Action dictionary with \'action_type\' and \'action_inputs\', or None if failed
        """
        # Format action history for context
        history_str = self._format_history(action_history)

        user_prompt = f"""**Task**: {task}

**Current Screen Observation**:
{observation}

**Action History**:
{history_str}

Based on the current screen and task, what should be the next action? Return JSON only."""

        try:
            # Log the planning request
            print("\n" + "=" * 80)
            print("ACTION PLANNER REQUEST")
            print("=" * 80)
            print(f"Model: {self.model}")
            print(f"\nSystem Prompt ({len(self.system_prompt)} chars):")
            print("-" * 80)
            print(self.system_prompt)
            print("-" * 80)
            print(f"\nUser Prompt ({len(user_prompt)} chars):")
            print("-" * 80)
            print(user_prompt)
            print("-" * 80)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            response_text = response.choices[0].message.content.strip()

            # Log the response and token usage
            print("\nACTION PLANNER RESPONSE")
            print("=" * 80)
            print(response_text)
            print("=" * 80)
            if hasattr(response, "usage") and response.usage:
                print(
                    f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}"
                )
            print("=" * 80 + "\n")

            # Parse the JSON response
            action_dict = self._parse_response(response_text)

            if action_dict:
                return action_dict
            else:
                print(
                    f"   Warning: Failed to parse action from response: {response_text[:500]}"
                )
                return None

        except Exception as e:
            print(f"   Error planning action: {e}")
            return None

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into action dictionary."""
        # Remove markdown code blocks if present
        response_text = re.sub(r"```json\s*", "", response_text)
        response_text = re.sub(r"```\s*", "", response_text)
        response_text = response_text.strip()

        try:
            data = json.loads(response_text)

            # Validate required fields
            if "action_type" not in data:
                return None

            # Ensure action_inputs exists
            if "action_inputs" not in data:
                data["action_inputs"] = {}

            return data

        except json.JSONDecodeError as e:
            print(f"   Warning: JSON decode error: {e}")
            return None

    def _format_history(self, action_history: List[Dict[str, Any]]) -> str:
        """Format action history into readable string."""
        if not action_history:
            return "No previous actions"

        # Show last 5 actions
        recent = action_history[-5:]
        lines = []
        for i, action in enumerate(recent, 1):
            action_type = action.get("action_type", "unknown")
            inputs = action.get("action_inputs", {})
            lines.append(f"{i}. {action_type}({inputs})")

        return "\n".join(lines)
