"""
VLM Agent - Vision-Language Model Agent for game understanding.

Agent 1 in the three-agent architecture:
- Input: Screenshot
- Output 1 (to GroundingDINO): Detection prompt specifying objects to locate
- Output 2 (to LLM): Game understanding context (rules, objectives, visual semantics)
"""

import os
from typing import Dict
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from PIL import Image
import base64
import io


class VLMAgent:
    """
    Vision-Language Model Agent for high-level game understanding.
    
    Analyzes screenshots to understand game rules and generates:
    1. Detection prompts for spatial localization
    2. Game context for decision making
    """
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """
        Initialize VLM Agent.
        
        Args:
            model: OpenAI vision model to use
            api_key: OpenAI API key (defaults to env var)
        """
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key)
        
    def analyze(self, screenshot_path: str, task_description: str = "") -> Dict[str, str]:
        """
        Analyze screenshot and generate dual outputs.
        
        Args:
            screenshot_path: Path to screenshot image
            task_description: Optional task context (e.g., "play maze game")
            
        Returns:
            Dict with keys:
            - 'detection_prompt': What objects GroundingDINO should detect
            - 'game_context': Game rules and understanding for LLM
            - 'raw_analysis': Full VLM response for debugging
        """
        # Load and encode image
        img = Image.open(screenshot_path)
        img = self._resize_for_vision(img)
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_img}"
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(task_description)
        
        # Call vision model with error handling
        print(f"\n[VLM Agent] Analyzing screenshot with {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                        ],
                    }
                ],
                max_tokens=600,
                temperature=0.3,
            )
        except RateLimitError as e:
            error_msg = (
                "\n❌ OpenAI API Quota Exceeded\n"
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
            raise RuntimeError(
                "OpenAI API quota exceeded. Please check your billing settings."
            ) from e
        except APIConnectionError as e:
            error_msg = (
                "\n❌ OpenAI API Connection Error\n"
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
            raise RuntimeError(
                "Failed to connect to OpenAI API. Check your connection."
            ) from e
        except APIError as e:
            error_msg = (
                "\n❌ OpenAI API Error\n"
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
            raise RuntimeError(
                f"OpenAI API error: {str(e)}"
            ) from e
        except Exception as e:
            error_msg = (
                "\n❌ Unexpected Error\n"
                "─" * 60 + "\n"
                f"Error: {str(e)}\n"
                f"Type: {type(e).__name__}\n"
                "─" * 60
            )
            print(error_msg)
            raise
        
        raw_response = response.choices[0].message.content.strip()
        
        # Log token usage
        if hasattr(response, "usage") and response.usage:
            print(f"[VLM Agent] Tokens: Input={response.usage.prompt_tokens}, "
                  f"Output={response.usage.completion_tokens}, "
                  f"Total={response.usage.total_tokens}")
        
        # Parse the dual outputs
        parsed = self._parse_dual_output(raw_response)
        parsed['raw_analysis'] = raw_response
        
        print(f"[VLM Agent] ✓ Analysis complete")
        print(f"  → Detection prompt length: {len(parsed['detection_prompt'])} chars")
        print(f"  → Game context length: {len(parsed['game_context'])} chars")
        
        return parsed
    
    def _resize_for_vision(self, image: Image.Image, max_edge: int = 1024) -> Image.Image:
        """Resize image to reduce tokens while preserving aspect ratio."""
        width, height = image.size
        ratio = max_edge / max(width, height)
        
        if ratio < 1:
            new_size = (int(width * ratio), int(height * ratio))
            return image.resize(new_size, Image.LANCZOS)
        return image
    
    def _build_analysis_prompt(self, task_description: str) -> str:
        """Build the analysis prompt for VLM."""
        base_prompt = """You are a game understanding agent. Analyze this screenshot and provide TWO outputs:

1. DETECTION_PROMPT: What specific objects should be detected for spatial localization?
   - List key visual elements (player, goals, obstacles, interactive objects)
   - Specify colors/shapes for identification
   - Format: "Detect: <object1> (<description>), <object2> (<description>), ..."

2. GAME_CONTEXT: What are the game rules and mechanics?
   - What type of game is this?
   - What do different colors/symbols represent?
   - What are the controls/actions available?
   - What is the objective?
   - Format: Clear, concise description of game rules

Format your response EXACTLY as:
```
DETECTION_PROMPT:
<your detection prompt here>

GAME_CONTEXT:
<your game context here>
```"""
        
        if task_description:
            base_prompt += f"\n\nTask context: {task_description}"
        
        return base_prompt
    
    def _parse_dual_output(self, response: str) -> Dict[str, str]:
        """
        Parse VLM response into detection_prompt and game_context.
        
        Args:
            response: Raw VLM response text
            
        Returns:
            Dict with 'detection_prompt' and 'game_context' keys
        """
        detection_prompt = ""
        game_context = ""
        
        # Remove markdown code blocks if present
        response = response.replace("```", "")
        
        # Split by the two sections
        if "DETECTION_PROMPT:" in response and "GAME_CONTEXT:" in response:
            parts = response.split("GAME_CONTEXT:")
            
            # Extract detection prompt
            detection_part = parts[0].split("DETECTION_PROMPT:")[1].strip()
            detection_prompt = detection_part.strip()
            
            # Extract game context
            game_context = parts[1].strip()
        else:
            # Fallback: treat entire response as game context
            print("[VLM Agent] Warning: Could not parse dual output format, using fallback")
            game_context = response
            detection_prompt = "Detect: all interactive game elements"
        
        return {
            'detection_prompt': detection_prompt,
            'game_context': game_context
        }


if __name__ == "__main__":
    import sys
    
    # Quick test
    if len(sys.argv) < 2:
        print("Usage: python -m src.modules.agents.vlm_agent <screenshot_path> [task]")
        sys.exit(1)
    
    screenshot_path = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else ""
    
    agent = VLMAgent()
    result = agent.analyze(screenshot_path, task)
    
    print("\n" + "=" * 80)
    print("VLM AGENT OUTPUT")
    print("=" * 80)
    print("\nDETECTION PROMPT (for GroundingDINO):")
    print("-" * 80)
    print(result['detection_prompt'])
    print("\nGAME CONTEXT (for LLM):")
    print("-" * 80)
    print(result['game_context'])
    print("=" * 80)
