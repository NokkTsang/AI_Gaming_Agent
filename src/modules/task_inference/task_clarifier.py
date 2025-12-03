"""
Task Clarifier - Converts ambiguous tasks into structured instructions (Game-TARS §3.1).

Implements Game-TARS instruction following approach:
- Defines explicit goal, action space, constraints, success criteria
- Detects ambiguities and asks user for clarification
- Prevents agent from confidently doing wrong thing
"""

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI


class TaskClarifier:
    """
    Transforms ambiguous task descriptions into structured instructions.

    Based on Game-TARS Instruction Following (§3.1): explicit contracts
    prevent behavioral inertia and ensure task understanding.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize task clarifier.

        Args:
            model: LLM model for analysis
            api_key: OpenAI API key
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def clarify_task(
        self,
        user_task: str,
        initial_observation: str = "",
        auto_accept_unambiguous: bool = True,
    ) -> Dict:
        """
        Transform ambiguous task into structured instruction.

        The clarification process supports BOTH:
        1. Quick selection: Type 'a', 'b', 'c' to choose from suggestions
        2. Free-text input: Type anything else to provide detailed explanation

        Examples:
          Q: "How should the agent handle dead ends in the maze?"
             a) Backtrack to last junction (default)
             b) Reset to start
             Or type your own answer...

          User can type: "a" → uses suggestion
                    OR: "Try backtracking 3 times, then use A* pathfinding" → uses this

        Args:
            user_task: User's original task (e.g., "play the maze game")
            initial_observation: Initial screenshot description (optional)
            auto_accept_unambiguous: Skip Q&A if no ambiguities detected

        Returns:
            Structured instruction dict with goal, constraints, success criteria
        """
        print(f'\n  [TaskClarifier] Processing task: "{user_task[:50]}..."')

        # Step 1: Analyze task and create structured instruction
        instruction = self._analyze_task(user_task, initial_observation)

        # Step 2: If ambiguities exist, ask user
        if instruction.get("ambiguities") and instruction["ambiguities"]:
            print("\nTask has ambiguities that need clarification:")
            print("-" * 80)

            clarifications = {}
            for i, question in enumerate(instruction["ambiguities"], 1):
                print(f"\n{i}. {question['question']}")

                # Show options as suggestions, but allow free-text
                if question.get("options"):
                    for opt_idx, option in enumerate(question["options"], 1):
                        default_marker = " (default)" if opt_idx == 1 else ""
                        print(f"   {chr(96+opt_idx)}) {option}{default_marker}")
                    print(f"   Or type your own answer...")

                    answer = input(
                        f"   Your answer (a-{chr(96+len(question['options']))} or type freely, Enter for default): "
                    ).strip()

                    # Parse answer: single letter = choice, anything else = free text
                    if not answer:  # Default to first option
                        answer = question["options"][0]
                    elif (
                        len(answer) == 1
                        and ord(answer.lower()) >= 97
                        and ord(answer.lower()) < 97 + len(question["options"])
                    ):
                        # User chose a letter option
                        answer = question["options"][ord(answer.lower()) - 97]
                    # else: keep the typed answer as-is (free text)
                else:
                    # Open-ended question
                    answer = input(f"   Your answer: ").strip()
                    if not answer:
                        answer = question.get("default", "No answer provided")

                clarifications[f"q{i}"] = answer
                print(f"   Recorded: {answer}")

            # Step 3: Refine instruction with answers
            instruction = self._refine_with_answers(
                user_task, instruction, clarifications
            )
            print("\n" + "-" * 80)
        else:
            if auto_accept_unambiguous:
                print("Task is unambiguous, proceeding with generated instruction")
            else:
                print("\nNo ambiguities detected")

        # Display final instruction summary
        print(f"  Goal: {instruction['goal'][:60]}...")
        print(
            f"    Actions: {len(instruction['action_space'])} | Constraints: {len(instruction['constraints'])} | Success criteria: {len(instruction['success_criteria'])}"
        )

        return instruction

    def _analyze_task(self, user_task: str, initial_observation: str) -> Dict:
        """
        Analyze task and create structured instruction with ambiguity detection.
        """
        context = (
            f"\n\nInitial screen: {initial_observation[:500]}"
            if initial_observation
            else ""
        )

        prompt = f"""User wants to: "{user_task}"{context}

Create a STRUCTURED INSTRUCTION for an AI agent:

1. GOAL: One clear sentence describing what to achieve
2. ACTION SPACE: List available actions and their semantics (e.g., "UP: Move character up")
3. CONSTRAINTS: What NOT to do or limitations (e.g., "Cannot touch walls")
4. SUCCESS CRITERIA: Specific conditions indicating completion (e.g., "Player reaches goal flag")
5. FAILURE CONDITIONS: Conditions indicating failure/reset (e.g., "Screen resets to start")
6. AMBIGUITIES: Questions needing user clarification (empty if none)

For AMBIGUITIES, provide:
- "question": Clear, open-ended question asking user to describe/explain
- "options": 2-4 common choices as SUGGESTIONS (user can still type freely)
- "default": Recommended default choice

Questions should invite detailed answers, not force multiple choice. Examples:
- GOOD: "How should the agent handle dead ends in the maze?" → options as suggestions
- GOOD: "What strategy should be used to prioritize targets?" → user can explain freely
- BAD: "Should it go left or right?" → too restrictive

Format as JSON:
{{
  "goal": "...",
  "action_space": ["action1: description", "action2: description", ...],
  "constraints": ["constraint1", "constraint2", ...],
  "success_criteria": ["criterion1", "criterion2", ...],
  "failure_conditions": ["condition1", "condition2", ...],
  "ambiguities": [
    {{"question": "...", "options": ["suggestion1", "suggestion2"], "default": "suggestion1"}},
    ...
  ]
}}

Common ambiguities to detect:
- Game-specific strategies (pathfinding, target prioritization, resource management)
- Behavior on obstacles/failure (retry, backtrack, skip)
- One-time vs repeated execution
- Speed vs accuracy tradeoffs
- Boundary conditions (what's allowed/forbidden)
- Domain-specific knowledge (game mechanics, controls, win conditions)

Return JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )

            result_text = response.choices[0].message.content.strip()
            # Remove markdown if present
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            instruction = json.loads(result_text)
            return instruction

        except Exception as e:
            print(f"   Error analyzing task: {e}")
            # Return minimal instruction on error
            return {
                "goal": user_task,
                "action_space": ["keyboard and mouse actions"],
                "constraints": ["Follow task requirements"],
                "success_criteria": ["Task appears complete"],
                "failure_conditions": ["Error messages or stuck state"],
                "ambiguities": [],
            }

    def _refine_with_answers(
        self, original_task: str, instruction: Dict, clarifications: Dict
    ) -> Dict:
        """
        Incorporate user's clarification answers into instruction.
        """
        # Format clarifications
        clarification_text = "\n".join(
            [
                f"Q{i+1}: {q['question']} → A: {clarifications.get(f'q{i+1}', 'no answer')}"
                for i, q in enumerate(instruction.get("ambiguities", []))
            ]
        )

        prompt = f"""Original task: "{original_task}"

Ambiguities and user answers:
{clarification_text}

Refine the instruction to incorporate these clarifications:

{{
  "goal": "...",  // Updated with clarifications
  "constraints": [...],  // Updated with clarifications
  "success_criteria": [...],  // Updated with clarifications
  "failure_conditions": [...]  // Updated if relevant
}}

Make the instruction UNAMBIGUOUS based on user's answers.
Return JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )

            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            refined = json.loads(result_text)

            # Merge with original instruction
            instruction.update(refined)
            # Remove ambiguities (now resolved)
            if "ambiguities" in instruction:
                del instruction["ambiguities"]
            # Add clarification history
            instruction["clarifications"] = clarifications

            return instruction

        except Exception as e:
            print(f"   Error refining instruction: {e}")
            # Remove ambiguities and return as-is
            if "ambiguities" in instruction:
                del instruction["ambiguities"]
            return instruction
