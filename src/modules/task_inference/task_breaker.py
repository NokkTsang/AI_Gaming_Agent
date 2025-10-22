"""
Task inference module for breaking down tasks and adjusting on failure.
Decomposes high-level tasks into subtasks and re-plans when needed.
"""

import os
from typing import List
from openai import OpenAI


class TaskBreaker:
    """Decomposes tasks into subtasks and adjusts plans."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize task breaker with OpenAI API.

        Args:
            api_key: OpenAI API key (reads from env if not provided)
            model: OpenAI model to use for task planning
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

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
- Focus on ACTIONS, not observations (avoid "review", "check", "verify", "observe")
- Each subtask should change the system state or interact with UI
- Final subtask should be the completion of the goal, not a verification step
- Aim for 3-6 subtasks

GOOD examples:
- "Click the Start button to launch the game"
- "Navigate to settings menu"
- "Enter username in login field"
- "Submit the search query"

BAD examples (too vague):
- "Review the search results" (what does review mean?)
- "Check if game loaded" (no action specified)
- "Verify the page" (passive observation)

Respond with a numbered list of ACTION-ORIENTED subtasks, one per line."""

        try:
            print("\n" + "=" * 80)
            print("TASK DECOMPOSITION REQUEST")
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
                        "content": "You are a task decomposition expert. Break tasks into clear, ordered subtasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            print("\nTASK DECOMPOSITION RESPONSE")
            print("=" * 80)
            print(result_text)
            print("=" * 80)
            if hasattr(response, "usage") and response.usage:
                print(
                    f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}"
                )
            print("=" * 80 + "\n")

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
            print("\n" + "=" * 80)
            print("TASK REPLANNING REQUEST")
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
                        "content": "You are a task replanning expert. Adjust plans when failures occur.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            print("\nTASK REPLANNING RESPONSE")
            print("=" * 80)
            print(result_text)
            print("=" * 80)
            if hasattr(response, "usage") and response.usage:
                print(
                    f"Tokens - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}"
                )
            print("=" * 80 + "\n")

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

Analyze if the task goal has been FULLY ACHIEVED based on the current screen state.

Consider:
- Is the desired end state visible on screen?
- Are we at the expected final destination (e.g., game started, search results shown, form submitted)?
- Has the action been completed (not just started)?

For searches: Are search results displayed?
For games: Has the game launched/started?
For forms: Has submission completed?
For navigation: Are we at the target page/screen?

Respond with only 'YES' if the task goal is FULLY achieved, or 'NO' if more work is needed."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You judge if a task goal has been fully achieved based on screen state. Be precise and look for concrete evidence of completion.",
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
