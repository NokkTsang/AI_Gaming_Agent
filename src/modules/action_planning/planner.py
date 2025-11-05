"""
Action Planner - Uses LLM to decide next action based on task and observation.
Supports both deep reasoning and reactive (fast) action planning for sparse thinking.
"""

import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI


class ActionPlanner:
    """
    Plans actions using a vision-capable LLM based on task and screen observations.

    Supports two modes:
    - Deep planning: Full reasoning + action (for complex decisions)
    - Reactive planning: Action-only (for simple continuations)
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = self._build_system_prompt()
        self.reactive_system_prompt = self._build_reactive_system_prompt()
        self.structured_instruction = None  # Will be set by main.py after clarification

    def _build_system_prompt(self) -> str:
        """Build the system prompt that defines the agent\'s action space."""
        return """You are a GUI automation agent. Decide the next action based on task and screen observation.

## Actions
1. click(start_box=[x, y]) - Click at normalized [0,1] coordinates
2. double_click(start_box=[x, y])
3. right_click(start_box=[x, y])
4. drag(start_box=[x1, y1], end_box=[x2, y2])
5. type(content="text") - Use \\n for Enter
6. hotkey(key="cmd c") - Space-separated keys. Arrow keys: "up", "down", "left", "right"
7. scroll(direction="up"/"down", start_box=[x, y])
8. wait() - Wait 5 seconds
9. move(start_box=[x, y]) - Smooth cursor movement (for spatial constraints)
10. finished(content="summary") - Task complete

## Coordinates
Normalized [0,1]: [0,0]=top-left, [1,1]=bottom-right, [0.5,0.5]=center

## Spatial Navigation
For obstacles/walls: Use move (not click) with small steps through safe zones.

## Action Batching for Games
For keyboard-based games (arrow keys, WASD): Return 3-5 actions in sequence to reduce delays.
Example: Moving through maze → plan multiple arrow key presses toward goal.

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
Single action: {"thought": "reasoning", "action_type": "click", "action_inputs": {"start_box": [0.5, 0.8]}}
Multiple actions: {"thought": "reasoning", "actions": [{"action_type": "hotkey", "action_inputs": {"key": "up"}}, {"action_type": "hotkey", "action_inputs": {"key": "right"}}]}

## Examples
Click: {"thought": "Clicking START button", "action_type": "click", "action_inputs": {"start_box": [0.5, 0.72]}}
Arrow key: {"thought": "Moving player up", "action_type": "hotkey", "action_inputs": {"key": "up"}}
Arrow sequence: {"thought": "Moving toward goal: up 3x then right 2x", "actions": [{"action_type": "hotkey", "action_inputs": {"key": "up"}}, {"action_type": "hotkey", "action_inputs": {"key": "up"}}, {"action_type": "hotkey", "action_inputs": {"key": "up"}}, {"action_type": "hotkey", "action_inputs": {"key": "right"}}, {"action_type": "hotkey", "action_inputs": {"key": "right"}}]}
Hotkey combo: {"thought": "Copy text", "action_type": "hotkey", "action_inputs": {"key": "cmd c"}}
Move: {"thought": "Moving cursor through maze path", "action_type": "move", "action_inputs": {"start_box": [0.3, 0.5]}}
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

        # Build system prompt with structured instruction if available
        system_prompt = self._get_augmented_system_prompt()

        try:
            # Log the planning request
            print("\n" + "=" * 80)
            print("ACTION PLANNER REQUEST")
            print("=" * 80)
            print(f"Model: {self.model}")
            print(f"\nSystem Prompt ({len(system_prompt)} chars):")
            print("-" * 80)
            print(system_prompt)
            print("-" * 80)
            print(f"\nUser Prompt ({len(user_prompt)} chars):")
            print("-" * 80)
            print(user_prompt)
            print("-" * 80)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
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

    def plan_reactive_action(
        self, task: str, observation: str, action_history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Plan action WITHOUT deep reasoning (reactive/fast mode).

        For simple/repetitive actions, skip expensive reasoning generation.
        This is Game-TARS sparse thinking: action-only policy for obvious moves.

        Args:
            task: The overall goal/task to accomplish
            observation: Current screen description (reused from last step)
            action_history: List of previous actions taken

        Returns:
            Action dictionary WITHOUT 'thought' field (reactive action)
        """
        # Format action history for context
        history_str = self._format_history(action_history)

        user_prompt = f"""**Task**: {task}

**Current State**:
{observation[:300]}...

**Recent Actions**:
{history_str}

Continue the task with the NEXT OBVIOUS ACTION. This is a simple continuation - no deep reasoning needed.

Return JSON action only (NO thought field):
{{"action_type": "...", "action_inputs": {{...}}}}"""

        try:
            # Log the planning request (abbreviated)
            print("\n" + "=" * 80)
            print("REACTIVE PLANNER (Fast Mode)")
            print("=" * 80)
            print(f"Model: {self.model} | Mode: Action-only (no reasoning)")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.reactive_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temp for predictable continuation
                max_tokens=200,  # Shorter (no reasoning text)
            )

            response_text = response.choices[0].message.content.strip()

            # Log the response
            print(f"Response: {response_text[:150]}")
            if hasattr(response, "usage") and response.usage:
                print(f"Tokens: {response.usage.total_tokens}")
            print("=" * 80 + "\n")

            # Parse the JSON response
            action_dict = self._parse_response(response_text)

            if action_dict:
                # Ensure no thought field (reactive mode)
                if "thought" in action_dict:
                    del action_dict["thought"]
                return action_dict
            else:
                print(f"   Warning: Failed to parse reactive action")
                return None

        except Exception as e:
            print(f"   Error in reactive planning: {e}")
            return None

    def _get_augmented_system_prompt(self) -> str:
        """
        Get system prompt augmented with structured instruction if available.
        This allows the agent to understand task-specific rules and action spaces.
        """
        base_prompt = self.system_prompt

        if self.structured_instruction:
            # Prepend the structured instruction to override generic rules
            instruction_text = "\n=== TASK-SPECIFIC RULES (HIGHEST PRIORITY) ===\n"
            instruction_text += (
                f"\nGOAL: {self.structured_instruction.get('goal', 'N/A')}\n"
            )

            instruction_text += "\nACTION SPACE (what actions are ALLOWED):\n"
            for action in self.structured_instruction.get("action_space", []):
                instruction_text += f"  • {action}\n"

            instruction_text += "\nCONSTRAINTS (what is FORBIDDEN):\n"
            for constraint in self.structured_instruction.get("constraints", []):
                instruction_text += f"  • {constraint}\n"

            instruction_text += "\nSUCCESS CRITERIA:\n"
            for criterion in self.structured_instruction.get("success_criteria", []):
                instruction_text += f"  • {criterion}\n"

            instruction_text += "\n=== GENERAL GUI AUTOMATION RULES (use if task rules don't apply) ===\n"

            return instruction_text + base_prompt

        return base_prompt

    def _build_reactive_system_prompt(self) -> str:
        """Build system prompt for reactive (action-only) planning."""
        return """You are a fast reactive agent. Generate the NEXT OBVIOUS ACTION without reasoning.

## Actions (same as normal mode)
1. click(start_box=[x, y])
2. double_click(start_box=[x, y])
3. right_click(start_box=[x, y])
4. drag(start_box=[x1, y1], end_box=[x2, y2])
5. type(content="text")
6. hotkey(key="...") - Arrow keys: "up", "down", "left", "right"
7. scroll(direction="up"/"down", start_box=[x, y])
8. wait()
9. move(start_box=[x, y])
10. finished(content="summary")

## Reactive Mode Rules
- Continue current pattern/direction
- NO reasoning or explanation
- Fast, predictable actions
- Trust recent action history

## Output Format
Single action (NO thought): {"action_type": "hotkey", "action_inputs": {"key": "up"}}
Action sequence (NO thought): {"actions": [{"action_type": "hotkey", "action_inputs": {"key": "up"}}, {"action_type": "hotkey", "action_inputs": {"key": "right"}}]}

Return JSON only, no markdown, NO thought field."""

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into action dictionary or action sequence."""
        # Remove markdown code blocks if present
        response_text = re.sub(r"```json\s*", "", response_text)
        response_text = re.sub(r"```\s*", "", response_text)
        response_text = response_text.strip()

        try:
            data = json.loads(response_text)

            # Check if this is a multi-action response
            if "actions" in data:
                # Validate it's a list of actions
                if not isinstance(data["actions"], list):
                    print("   Warning: 'actions' field must be a list")
                    return None

                # Validate each action has required fields
                for action in data["actions"]:
                    if "action_type" not in action:
                        print("   Warning: Each action must have 'action_type'")
                        return None
                    if "action_inputs" not in action:
                        action["action_inputs"] = {}

                return data  # Return the full dict with "actions" list

            # Single action response
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
