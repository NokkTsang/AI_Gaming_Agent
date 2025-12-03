"""
Skill curation module for extracting and saving reusable action sequences.
Converts successful action sequences into callable skills.
"""

import os
from typing import List, Dict, Tuple
from openai import OpenAI


class SkillManager:
    """Extracts and manages reusable skills from successful actions."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize skill manager with OpenAI API.

        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: OpenAI model to use for skill curation
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def should_save_as_skill(
        self, action_sequence: List[Dict], subtask_completed: str
    ) -> bool:
        """
        Determine if action sequence is worth saving as skill.

        Args:
            action_sequence: List of actions that led to success
            subtask_completed: Subtask description

        Returns:
            True if sequence should be saved
        """
        # Heuristics for skill-worthiness
        if len(action_sequence) < 3:
            # Too simple - single actions or pairs not worth saving
            # This prevents saving trivial skills like "press right arrow"
            return False

        if len(action_sequence) > 15:
            # Too complex, might be too specific
            return False

        # Check if all actions are the same type (not useful as a skill)
        action_types = [a.get("action_type") or a.get("tool") for a in action_sequence]
        if len(set(action_types)) == 1:
            # All same action type (e.g., all hotkeys) - not a useful skill
            return False

        # Check for variety in the sequence (indicates reusability)
        return True

    def extract_skill(
        self, action_sequence: List[Dict], subtask_description: str, task_context: str
    ) -> Tuple[str, str, str]:
        """
        Extract skill from successful action sequence.

        Args:
            action_sequence: List of action dicts
            subtask_description: What the sequence accomplished
            task_context: Original task context

        Returns:
            Tuple of (skill_name, skill_description, skill_code)
        """
        # Generate skill name from subtask
        skill_name = self._generate_skill_name(subtask_description)

        # Create skill description
        skill_description = f"Accomplishes: {subtask_description}"

        # Convert actions to Python code
        skill_code = self._actions_to_code(action_sequence, skill_name)

        return skill_name, skill_description, skill_code

    def _generate_skill_name(self, subtask_description: str) -> str:
        """
        Generate concise skill name from subtask.

        Args:
            subtask_description: Natural language subtask

        Returns:
            Snake_case skill name
        """
        prompt = f"""Convert this task description into a concise snake_case function name (2-4 words):

Task: {subtask_description}

Respond with only the function name, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate concise snake_case function names.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=20,
            )

            skill_name = response.choices[0].message.content.strip()
            tokens = (
                response.usage.total_tokens
                if hasattr(response, "usage") and response.usage
                else "?"
            )
            print(f"  [SkillManager] Generated name: {skill_name} | Tokens: {tokens}")

            # Clean up name
            skill_name = skill_name.replace(" ", "_").lower()
            skill_name = "".join(c for c in skill_name if c.isalnum() or c == "_")

            return skill_name if skill_name else "new_skill"

        except Exception as e:
            print(f"Skill name generation error: {e}")
            # Fallback: simple cleanup
            return subtask_description.lower().replace(" ", "_")[:30]

    def _actions_to_code(self, action_sequence: List[Dict], skill_name: str) -> str:
        """
        Convert action sequence to executable Python code.

        Args:
            action_sequence: List of action dicts
            skill_name: Name for the skill function

        Returns:
            Python code string
        """
        code_lines = [f"def {skill_name}():"]
        code_lines.append('    """Generated skill from successful action sequence."""')

        for action in action_sequence:
            # Convert action dict to function call
            tool_name = action.get("tool", "wait")
            params = {k: v for k, v in action.items() if k != "tool"}

            # Format parameters
            param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())

            code_lines.append(f"    {tool_name}({param_str})")

        return "\n".join(code_lines)
