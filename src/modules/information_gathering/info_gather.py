from smolagents import CodeAgent, LiteLLMModel, tool
from dotenv import load_dotenv
from typing import Optional
import base64
import os
import glob
import sys

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()


# ---- Minimal helper to send an image to an LLM and get feedback ----
def analyze_screenshot(
    image_path: str,
    question: str = "Describe the screenshot briefly and list 3-5 salient UI elements.",
    model: str = "gpt-4.1-nano",
) -> str:
    """Send an image to a vision-capable OpenAI model and return its feedback.

    Args:
        image_path: Local path to the image (jpg/png).
        question: Instruction for the LLM about how to analyze the image.
        model: OpenAI model that supports vision (e.g., gpt-4o, gpt-4o-mini).
    Returns:
        The model's text response.
    """
    from openai import OpenAI  # lightweight import to keep module load fast

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")

    # Read and base64-encode the image as a data URL
    with open(image_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


# ---- Optional: expose the analyzer as a smolagents Tool ----
@tool
def analyze_image_tool(image_path: str, prompt: Optional[str] = None) -> str:
    """Analyze a screenshot with a vision LLM and return concise feedback.

    Args:
        image_path (str): Absolute or relative path to a local JPG/PNG image.
        prompt (str, optional): Custom instruction for the analysis. If None, a default prompt is used.

    Returns:
        str: Concise textual feedback from the model.
    """
    question = (
        prompt or "Describe the screenshot briefly and list 3-5 salient UI elements."
    )
    return analyze_screenshot(image_path=image_path, question=question)


# ---- Create a smolagents CodeAgent that can use the tool if desired ----
def build_agent() -> CodeAgent:
    api_key = os.getenv("OPENAI_API_KEY")
    # Use a lightweight text model for tool orchestration; the tool itself calls the vision model.
    model = LiteLLMModel(model_id="openai/gpt-4.1-nano", api_key=api_key)
    return CodeAgent(tools=[analyze_image_tool], model=model, add_base_tools=True)


def _latest_screenshot(
    default_dir: str = "src/modules/screen_input/screenshots",
) -> Optional[str]:
    """Return the newest jpg/png in default_dir, if any.

    Args:
        default_dir (str): Directory to search for screenshots.

    Returns:
        Optional[str]: Path to newest screenshot or None if none found.
    """
    paths = glob.glob(os.path.join(default_dir, "*.jpg")) + glob.glob(
        os.path.join(default_dir, "*.png")
    )
    return max(paths, key=os.path.getmtime) if paths else None


if __name__ == "__main__":
    # Usage:
    # python -m src.modules.information_gathering.info_gather /path/to/image.jpg "optional prompt"
    # or just run without args to auto-pick the latest screenshot from src/screen_input/screenshots

    img_path = sys.argv[1] if len(sys.argv) >= 2 else _latest_screenshot()
    prompt = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else "Describe the screenshot briefly and list 3-5 salient UI elements."
    )

    if not img_path or not os.path.exists(img_path):
        raise SystemExit(
            "No image provided and no screenshots found. Pass an image path as the first argument, "
            "or save one under src/modules/screen_input/screenshots."
        )

    # Option A: Call the helper directly (fast path)
    feedback = analyze_screenshot(image_path=img_path, question=prompt)
    print("Feedback (direct):\n", feedback)

    # Option B: Let smolagents orchestrate tool usage (optional)
    # agent = build_agent()
    # result = agent.run(f"Use analyze_image_tool on '{img_path}'. {prompt}")
    # print("\nFeedback (via agent):\n", result)
