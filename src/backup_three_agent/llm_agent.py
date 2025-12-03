"""
LLM Agent (Agent 3) - Action Decision Making

This agent receives:
- Game context from VLM Agent (rules, objectives, visual semantics)
- Spatial data from GroundingDINO Agent (object positions, relationships)

This agent outputs:
- Executable action instruction in JSON format
- Reasoning explanation for the decision

The output is consumed by the UI Automation Executor.
"""

from typing import Dict, Optional
import os
import json
from openai import RateLimitError, APIError, APIConnectionError


class LLMAgent:
    """
    Agent 3: LLM-based decision maker for action planning.

    Combines high-level game understanding from VLM with precise spatial
    data from DINO to make informed action decisions.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Initialize LLM Agent.

        Args:
            model: OpenAI model to use for decision making
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("[LLM Agent] OpenAI API key not found")
                return

            self.client = OpenAI(api_key=api_key)
            print(f"[LLM Agent] Initialized with {self.model}")

        except Exception as e:
            print(f"[LLM Agent] FAILED to initialize: {e}")
            self.client = None

    def decide(
        self,
        game_context: str,
        spatial_data: Dict,
        task_description: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Make action decision based on game context and spatial data.

        Args:
            game_context: Game understanding from VLM Agent
            spatial_data: Detection results from GroundingDINO Agent
            task_description: Optional current task/goal

        Returns:
            Dictionary containing:
            - action: Executable action instruction (JSON)
            - reasoning: Explanation of the decision
            - confidence: Confidence score (0-1)
        """
        print(f"\n[LLM Agent] Making decision with {self.model}...")

        # Validate inputs
        if not game_context or not game_context.strip():
            raise ValueError("Game context cannot be empty")

        if not spatial_data:
            raise ValueError("Spatial data cannot be empty")

        # Check client availability
        if not self.client:
            print("[LLM Agent] OpenAI client not available")
            return {
                "action": None,
                "reasoning": "LLM client not initialized",
                "confidence": 0.0,
                "error": "OpenAI client not available",
            }

        # Build decision prompt
        prompt = self._build_decision_prompt(
            game_context, spatial_data, task_description
        )

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI gaming agent that makes precise action decisions. "
                            "Based on game rules and spatial data, decide the next action. "
                            "Output must be valid JSON with: action, reasoning, confidence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=500,
            )

            # Parse response
            result = self._parse_llm_response(response)

            print(f"[LLM Agent] Decision made")
            print(f"  → Action: {result.get('action', {})}")
            print(f"  → Confidence: {result.get('confidence', 0):.2f}")
            print(f"  → Reasoning: {result.get('reasoning', '')[:100]}...")

            return result

        except RateLimitError as e:
            error_msg = (
                "\n[ERROR] OpenAI API Quota Exceeded\n"
                "─" * 60 + "\n"
                "Your OpenAI API account has exceeded its quota.\n\n"
                "Possible solutions:\n"
                "  1. Check your billing details at: https://platform.openai.com/account/billing\n"
                "  2. Add payment method or upgrade your plan\n"
                "  3. Wait for quota reset (if on free tier)\n"
                "  4. Use a different API key with available quota\n\n"
                f"Error details: {str(e)}\n"
                "─" * 60
            )
            print(error_msg)
            return {
                "action": None,
                "reasoning": "OpenAI API quota exceeded",
                "confidence": 0.0,
                "error": "rate_limit_exceeded",
            }

        except APIConnectionError as e:
            error_msg = (
                "\n[ERROR] OpenAI API Connection Error\n"
                "─" * 60 + "\n"
                "Failed to connect to OpenAI API.\n\n"
                "Possible solutions:\n"
                "  1. Check your internet connection\n"
                "  2. Verify OpenAI API is accessible in your region\n"
                "  3. Check if you're behind a proxy/firewall\n"
                "  4. Try again in a few moments\n\n"
                f"Error details: {str(e)}\n"
                "─" * 60
            )
            print(error_msg)
            return {
                "action": None,
                "reasoning": "Failed to connect to OpenAI API",
                "confidence": 0.0,
                "error": "connection_error",
            }

        except APIError as e:
            error_msg = (
                "\n[ERROR] OpenAI API Error\n"
                "─" * 60 + "\n"
                f"API Error: {str(e)}\n\n"
                "Possible solutions:\n"
                "  1. Verify your API key is valid\n"
                "  2. Check if the model is available\n"
                "  3. Review the error message above\n"
                "  4. Try again in a few moments\n"
                "─" * 60
            )
            print(error_msg)
            return {
                "action": None,
                "reasoning": f"OpenAI API error: {str(e)}",
                "confidence": 0.0,
                "error": "api_error",
            }

        except Exception as e:
            print(f"[LLM Agent] FAILED: Decision failed: {e}")
            return {
                "action": None,
                "reasoning": f"LLM call failed: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
            }

    def _build_decision_prompt(
        self, game_context: str, spatial_data: Dict, task_description: Optional[str]
    ) -> str:
        """Build prompt for LLM decision making."""

        # Format spatial data
        spatial_summary = self._format_spatial_data(spatial_data)

        prompt = f"""# Game Context
{game_context}

# Spatial Information
{spatial_summary}
"""

        if task_description:
            prompt += f"""
# Current Task
{task_description}
"""

        prompt += """
# Decision Required
Based on the game context and spatial information, decide the next action.

Output format (valid JSON):
{
  "action": {
    "action_type": "hotkey" | "click" | "type" | "wait",
    "action_inputs": {
      // For hotkey: {"key": "up|down|left|right|w|a|s|d|..."}
      // For click: {"x": pixel_x, "y": pixel_y}
      // For type: {"text": "string to type"}
      // For wait: {"duration": seconds}
    }
  },
  "reasoning": "Brief explanation of why this action was chosen",
  "confidence": 0.0-1.0
}

Provide your decision:"""

        return prompt

    def _format_spatial_data(self, spatial_data: Dict) -> str:
        """Format spatial data into readable text."""

        lines = []

        # Screenshot size
        if "screenshot_size" in spatial_data:
            size = spatial_data["screenshot_size"]
            lines.append(f"Screenshot: {size.get('width', 0)}x{size.get('height', 0)}")

        # Detection prompt
        if "detection_prompt" in spatial_data:
            lines.append(f"Objects to detect: {spatial_data['detection_prompt']}")

        # Detected objects
        if "detected_objects" in spatial_data:
            objects = spatial_data["detected_objects"]
            lines.append(f"\nDetected {len(objects)} objects:")

            for idx, obj in enumerate(objects, 1):
                obj_name = obj.get("object", "unknown")
                pixel_coords = obj.get("pixel_coords", {})
                norm_coords = obj.get("normalized_coords", {})
                confidence = obj.get("confidence", 0)

                x = pixel_coords.get("x_center", 0)
                y = pixel_coords.get("y_center", 0)
                norm_x = norm_coords.get("x_center", 0)
                norm_y = norm_coords.get("y_center", 0)

                lines.append(
                    f"  {idx}. {obj_name}: "
                    f"pixel({x}, {y}) "
                    f"normalized({norm_x:.2f}, {norm_y:.2f}) "
                    f"confidence={confidence:.2f}"
                )
        else:
            lines.append("\nNo objects detected")

        return "\n".join(lines)

    def _parse_llm_response(self, response) -> Dict:
        """Parse LLM response and extract decision."""

        content = response.choices[0].message.content.strip()

        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()

        try:
            result = json.loads(content)

            # Validate structure
            if "action" not in result:
                result["action"] = None
            if "reasoning" not in result:
                result["reasoning"] = "No reasoning provided"
            if "confidence" not in result:
                result["confidence"] = 0.5

            return result

        except json.JSONDecodeError as e:
            print(f"[LLM Agent] Failed to parse JSON: {e}")
            print(f"[LLM Agent] Raw content: {content[:200]}")

            return {
                "action": None,
                "reasoning": f"Failed to parse LLM response: {content[:100]}",
                "confidence": 0.0,
                "raw_response": content,
            }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.modules.agents.llm_agent <game_context> <spatial_data_json>"
        )
        print("\nExample:")
        print(
            '  python -m src.modules.agents.llm_agent "Maze game" \'{"detected_objects":[]}\''
        )
        sys.exit(1)

    game_context = sys.argv[1]
    spatial_data = json.loads(sys.argv[2])

    # Run decision
    agent = LLMAgent()
    result = agent.decide(game_context, spatial_data)

    # Print results
    print("\n" + "=" * 80)
    print("DECISION RESULTS")
    print("=" * 80)
    print(f"\nAction: {json.dumps(result.get('action'), indent=2)}")
    print(f"\nReasoning: {result.get('reasoning')}")
    print(f"Confidence: {result.get('confidence'):.2%}")
