"""
Task inference module for breaking down tasks and adjusting on failure.
Decomposes high-level tasks into subtasks and re-plans when needed.
"""

import os
from typing import List
from openai import OpenAI


class TaskBreaker:
    """Decomposes tasks into subtasks and adjusts plans."""

    def __init__(self, api_key: str = None):
        """
        Initialize task breaker with OpenAI API.

        Args:
            api_key: OpenAI API key (reads from env if not provided)
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def decompose_task(self, task_goal: str, initial_observation: str) -> List[str]:
        """
        Break down high-level task into ordered subtasks.

        Args:
            task_goal: High-level task description
            initial_observation: Current screen state

        Returns:
            List of subtask descriptions
        """
        prompt = f"""You are a task planning expert. Break down the following task into a sequence of smaller subtasks.

Task: {task_goal}

Current Screen State:
{initial_observation}

Requirements:
- Each subtask should be concrete and achievable
- Subtasks should be ordered logically
- Keep subtasks high-level (don't specify exact clicks)
- Aim for 3-7 subtasks

Respond with a numbered list of subtasks, one per line."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a task decomposition expert. Break tasks into clear, ordered subtasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse numbered list
            subtasks = []
            for line in result_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove numbering
                    subtask = line.lstrip("0123456789.-) ").strip()
                    if subtask:
                        subtasks.append(subtask)

            return subtasks if subtasks else [task_goal]

        except Exception as e:
            print(f"Task decomposition error: {e}")
            # Fallback: return original task as single subtask
            return [task_goal]

    def replan_after_failure(
        self,
        task_goal: str,
        failed_subtask: str,
        failure_reason: str,
        current_observation: str,
        completed_subtasks: List[str],
    ) -> List[str]:
        """
        Generate new subtask plan after failure.

        Args:
            task_goal: Overall task goal
            failed_subtask: Subtask that failed
            failure_reason: Why it failed
            current_observation: Current screen state
            completed_subtasks: Subtasks completed so far

        Returns:
            New list of remaining subtasks
        """
        completed_str = "\n".join(f"✓ {s}" for s in completed_subtasks)

        prompt = f"""You are a task planning expert. A subtask has failed and you need to adjust the plan.

Overall Task: {task_goal}

Completed Subtasks:
{completed_str if completed_str else "None yet"}

Failed Subtask: {failed_subtask}
Failure Reason: {failure_reason}

Current Screen State:
{current_observation}

Generate a new plan for the remaining subtasks. Consider:
- Why did the subtask fail?
- Do we need a different approach?
- Should we break it into smaller steps?
- Are there prerequisites we missed?

Respond with a numbered list of NEW subtasks to complete the task."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a task replanning expert. Adjust plans when failures occur.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse numbered list
            subtasks = []
            for line in result_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    subtask = line.lstrip("0123456789.-) ").strip()
                    if subtask:
                        subtasks.append(subtask)

            return subtasks if subtasks else [failed_subtask]

        except Exception as e:
            print(f"Replanning error: {e}")
            # Fallback: retry the failed subtask
            return [failed_subtask]

    def check_task_completion(self, task_goal: str, current_observation: str) -> bool:
        """
        Check if overall task goal is achieved.

        Args:
            task_goal: Original task goal
            current_observation: Current screen state

        Returns:
            True if task appears complete
        """
        prompt = f"""Task Goal: {task_goal}

Current Screen State:
{current_observation}

Has the task goal been achieved? Respond with only 'YES' or 'NO'."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You judge if a task goal has been achieved based on screen state.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=10,
            )

            result_text = response.choices[0].message.content.strip().upper()
            return "YES" in result_text

        except Exception as e:
            print(f"Completion check error: {e}")
            return False
