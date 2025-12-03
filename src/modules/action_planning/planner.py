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
        """Build the system prompt that defines the agent's action space (compressed for token efficiency)."""
        return """GUI automation agent. Actions (normalized [0,1] coords, [0,0]=top-left):
- click/double_click/right_click(start_box=[x,y])
- drag(start_box=[x1,y1], end_box=[x2,y2])
- type(content="text") - \\n for Enter
- hotkey(key="cmd c") - space-separated, arrows: "up"/"down"/"left"/"right"
- scroll(direction="up"/"down", start_box=[x,y])
- move(start_box=[x,y]) - smooth cursor movement
- wait() - 5 second pause
- finished(content="summary") - task complete

Rules:
- Use OCR coordinates for text elements
- For games: batch 3-5 arrow keys in sequence
- If "CLICK FAILED" with coordinates → use those exact coords
- After 2+ failed clicks: try type/hotkey/different position
- Never repeat same failed action >2 times

Output JSON only:
{"thought":"reason", "action_type":"click", "action_inputs":{"start_box":[0.5,0.8]}}
Or multi: {"thought":"reason", "actions":[{"action_type":"hotkey", "action_inputs":{"key":"up"}}, ...]}
"""

    def plan_next_action(
        self,
        task: str,
        observation: str,
        action_history: List[Dict[str, Any]],
        relevant_skills: List[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Plan the next action based on task, current observation, and history.

        Args:
            task: The overall goal/task to accomplish
            observation: Current screen description from vision LLM
            action_history: List of previous actions taken
            relevant_skills: List of relevant skill dicts from memory (optional)

        Returns:
            Action dictionary with 'action_type' and 'action_inputs', or None if failed
        """
        # Format action history for context
        history_str = self._format_history(action_history)

        # Format relevant skills if provided
        skills_str = self._format_skills(relevant_skills) if relevant_skills else ""

        user_prompt = f"""**Task**: {task}

**Current Screen Observation**:
{observation}

**Action History**:
{history_str}
{skills_str}
Based on the current screen and task, what should be the next action? Return JSON only."""

        # Build system prompt with structured instruction if available
        system_prompt = self._get_augmented_system_prompt()

        try:
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

            # Log tokens
            if hasattr(response, "usage") and response.usage:
                print(
                    f"     [Planner] Tokens: {response.usage.prompt_tokens}→{response.usage.completion_tokens} ({response.usage.total_tokens} total)"
                )

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
        self,
        task: str,
        observation: str,
        action_history: List[Dict[str, Any]],
        relevant_skills: List[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Plan action WITHOUT deep reasoning (reactive/fast mode).

        For simple/repetitive actions, skip expensive reasoning generation.
        This is Game-TARS sparse thinking: action-only policy for obvious moves.

        Args:
            task: The overall goal/task to accomplish
            observation: Current screen description (reused from last step)
            action_history: List of previous actions taken
            relevant_skills: List of relevant skill dicts from memory (optional)

        Returns:
            Action dictionary WITHOUT 'thought' field (reactive action)
        """
        # Format action history for context
        history_str = self._format_history(action_history)

        # Format skills if provided (brief version for reactive mode)
        skills_hint = ""
        if relevant_skills:
            skill_names = [s.get("skill_name", "?") for s in relevant_skills[:2]]
            skills_hint = f"\n(Available skills: {', '.join(skill_names)})"

        user_prompt = f"""**Task**: {task}

**Current State**:
{observation[:300]}...

**Recent Actions**:
{history_str}{skills_hint}

Continue the task with the NEXT OBVIOUS ACTION. This is a simple continuation - no deep reasoning needed.

Return JSON action only (NO thought field):
{{"action_type": "...", "action_inputs": {{...}}}}"""

        try:
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

            # Log tokens
            if hasattr(response, "usage") and response.usage:
                print(f"     [Reactive] Tokens: {response.usage.total_tokens}")

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
        return """Fast reactive agent. Generate NEXT OBVIOUS ACTION without reasoning.
Continue current pattern. NO thought field. Trust recent history.

Output JSON only (NO thought):
{"action_type": "hotkey", "action_inputs": {"key": "up"}}
Or sequence: {"actions": [{"action_type": "hotkey", "action_inputs": {"key": "up"}}]}"""

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

    def _format_skills(self, skills: List[Dict]) -> str:
        """Format relevant skills into prompt context."""
        if not skills:
            return ""

        lines = ["\n**Relevant Skills from Memory** (reuse if applicable):"]
        for skill in skills[:3]:  # Limit to top 3 to save tokens
            name = skill.get("skill_name", "unknown")
            desc = skill.get("description", "")
            # Extract action pattern from code (simplified)
            code = skill.get("code", "")
            # Find action types in code
            import re

            actions = re.findall(r"action_type='(\w+)'", code)
            action_pattern = ", ".join(actions[:5]) if actions else "N/A"
            lines.append(f"  - {name}: {desc[:80]}... [Actions: {action_pattern}]")

        return "\n".join(lines)
