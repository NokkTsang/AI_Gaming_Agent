"""
Action Planner - Uses LLM to decide next action based on task and observation.
"""

import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI


class ActionPlanner:
    """Plans actions using a vision-capable LLM based on task and screen observations."""

    def __init__(self, model: str = "gpt-4.1-nano", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines the agent's action space."""
        return """You are a GUI automation agent. Given a task and screen observation, you decide the next action.

## Available Actions

1. **click(start_box=[x, y])** - Click at normalized coordinates [x, y] where 0 ≤ x, y ≤ 1
2. **double_click(start_box=[x, y])** - Double-click at position
3. **right_click(start_box=[x, y])** - Right-click at position
4. **drag(start_box=[x1, y1], end_box=[x2, y2])** - Drag from start to end
5. **type(content="text")** - Type text (use \\n at end to press Enter)
6. **hotkey(key="ctrl c")** - Press hotkey combo (space-separated, lowercase)
7. **scroll(direction="up" or "down", start_box=[x, y])** - Scroll at position (optional)
8. **wait()** - Wait 5 seconds
9. **finished(content="summary")** - Mark task as complete

## Coordinate System
- All coordinates are **normalized** in range [0, 1]
- [0, 0] = top-left corner
- [1, 1] = bottom-right corner
- [0.5, 0.5] = center of screen

## Output Format
Return ONLY a valid JSON object:
```json
{
  "thought": "Brief reasoning about what to do next",
  "action_type": "click",
  "action_inputs": {
    "start_box": [0.5, 0.1]
  }
}
```

## Examples

**Example 1: Click address bar**
```json
{
  "thought": "I need to open a browser first. I'll click on the Chrome icon in the taskbar.",
  "action_type": "click",
  "action_inputs": {"start_box": [0.5, 0.95]}
}
```

**Example 2: Type in search**
```json
{
  "thought": "The search box is visible. I'll type the search query.",
  "action_type": "type",
  "action_inputs": {"content": "openai\\n"}
}
```

**Example 3: Open browser with hotkey**
```json
{
  "thought": "I'll use Cmd+Space to open Spotlight and launch a browser.",
  "action_type": "hotkey",
  "action_inputs": {"key": "cmd space"}
}
```

**Example 4: Task complete**
```json
{
  "thought": "The search results for 'openai' are now displayed. Task complete.",
  "action_type": "finished",
  "action_inputs": {"content": "Successfully searched for openai on Google"}
}
```

## Important Rules
1. Think step-by-step
2. Only use actions from the list above
3. Coordinates must be in [0, 1] range
4. Return valid JSON only (no markdown, no extra text)
5. Be specific about which UI element you're targeting
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
            Action dictionary with 'action_type' and 'action_inputs', or None if failed
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )

            response_text = response.choices[0].message.content.strip()

            # Parse the JSON response
            action_dict = self._parse_response(response_text)

            if action_dict:
                return action_dict
            else:
                print(
                    f"   ⚠️ Failed to parse action from response: {response_text[:200]}"
                )
                return None

        except Exception as e:
            print(f"   ❌ Error planning action: {e}")
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
            print(f"   ⚠️ JSON decode error: {e}")
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
