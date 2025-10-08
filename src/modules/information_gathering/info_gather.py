from smolagents import CodeAgent, LiteLLMModel, tool
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple
import base64
import os
import glob
import sys

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()


def extract_text_with_ocr(image_path: str) -> List[Dict[str, any]]:
    """Extract text and bounding boxes from screenshot using OCR.

    Args:
        image_path: Path to screenshot

    Returns:
        List of dicts with 'text', 'x', 'y', 'width', 'height' (normalized 0-1)
    """
    try:
        import easyocr
        from PIL import Image
        import ssl

        # Disable SSL verification for model download (macOS fix)
        ssl._create_default_https_context = ssl._create_unverified_context

        # Initialize reader (cached after first call)
        if not hasattr(extract_text_with_ocr, "reader"):
            print("   Initializing OCR (first-time setup, may take a moment)...")
            extract_text_with_ocr.reader = easyocr.Reader(
                ["en"], gpu=False, verbose=False
            )

        reader = extract_text_with_ocr.reader

        # Get image dimensions for normalization
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Run OCR
        results = reader.readtext(image_path)

        # Format results with normalized coordinates
        text_boxes = []
        for bbox, text, confidence in results:
            if confidence < 0.5:  # Skip low-confidence detections
                continue

            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Normalize to [0,1]
            text_boxes.append(
                {
                    "text": text.strip(),
                    "x": (x_min + x_max) / 2 / img_width,  # Center x
                    "y": (y_min + y_max) / 2 / img_height,  # Center y
                    "width": (x_max - x_min) / img_width,
                    "height": (y_max - y_min) / img_height,
                    "confidence": confidence,
                }
            )

        return text_boxes

    except ImportError:
        # EasyOCR not installed, return empty list
        return []
    except Exception as e:
        # OCR failed, silently return empty list (agent will work without OCR)
        return []


def create_grid_overlay(image_width: int, image_height: int) -> str:
    """Create a 3x3 grid description for spatial reference.

    Args:
        image_width: Screen width in pixels
        image_height: Screen height in pixels

    Returns:
        Formatted grid description string
    """
    grid_desc = "\n3x3 GRID REFERENCE:\n"
    grid_labels = [
        [
            "A1 [0.0-0.33, 0.0-0.33]",
            "A2 [0.33-0.67, 0.0-0.33]",
            "A3 [0.67-1.0, 0.0-0.33]",
        ],
        [
            "B1 [0.0-0.33, 0.33-0.67]",
            "B2 [0.33-0.67, 0.33-0.67]",
            "B3 [0.67-1.0, 0.33-0.67]",
        ],
        [
            "C1 [0.0-0.33, 0.67-1.0]",
            "C2 [0.33-0.67, 0.67-1.0]",
            "C3 [0.67-1.0, 0.67-1.0]",
        ],
    ]

    for row in grid_labels:
        grid_desc += " | ".join(row) + "\n"

    return grid_desc


# ---- Minimal helper to send an image to an LLM and get feedback ----
def analyze_screenshot(
    image_path: str,
    question: str = "Describe the screenshot briefly and list 3-5 salient UI elements.",
    model: str = "gpt-4.1-nano",
    include_ocr: bool = False,
    include_grid: bool = True,
) -> str:
    """Send an image to a vision-capable OpenAI model and return its feedback.

    Args:
        image_path: Local path to the image (jpg/png).
        question: Instruction for the LLM about how to analyze the image.
        model: OpenAI model that supports vision (e.g., gpt-4o, gpt-4o-mini).
        include_ocr: Whether to include OCR text detection results.
        include_grid: Whether to include 3x3 grid reference.
    Returns:
        The model's text response.
    """
    from openai import OpenAI  # lightweight import to keep module load fast
    from PIL import Image

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")

    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Build enhanced question with OCR and grid data
    enhanced_question = question

    if include_grid:
        enhanced_question = (
            create_grid_overlay(img_width, img_height) + "\n" + enhanced_question
        )

    if include_ocr:
        text_boxes = extract_text_with_ocr(image_path)
        if text_boxes:
            ocr_info = "\n\nDETECTED TEXT ELEMENTS:\n"
            for idx, box in enumerate(text_boxes[:20], 1):  # Limit to top 20
                ocr_info += (
                    f"{idx}. '{box['text']}' at [{box['x']:.2f}, {box['y']:.2f}]\n"
                )
            enhanced_question = enhanced_question + ocr_info

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
                    {"type": "text", "text": enhanced_question},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ],
        max_tokens=800,
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
