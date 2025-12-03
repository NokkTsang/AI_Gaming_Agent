from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple
import base64
import os
import io
import glob
import sys
import warnings
from PIL import Image

# Suppress verbose warnings from torch/transformers (they clutter logs)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")
warnings.filterwarnings("ignore", message=".*device.*argument is deprecated.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv()


def resize_for_vision(image: Image.Image) -> Image.Image:
    """Resize image to reduce tokens while preserving aspect ratio.

    Scales down large images to max 1024px on longest edge.
    Works on any screen resolution/ratio.

    Args:
        image: PIL Image object

    Returns:
        Resized PIL Image (or original if already small)
    """
    max_edge = 1024
    width, height = image.size
    ratio = max_edge / max(width, height)

    if ratio < 1:  # Only downscale, never upscale
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image


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
        print(f"   OCR detected {len(results)} text elements")

        # Format results with normalized coordinates
        text_boxes = []
        for bbox, text, confidence in results:
            if (
                confidence < 0.3
            ):  # Skip very low-confidence detections (lowered from 0.5 for stylized game fonts)
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

        print(
            f"   OCR found {len(text_boxes)} high-confidence text elements (>0.5 confidence)"
        )
        for box in text_boxes[:10]:  # Show first 10
            # Print with 3 significant figures for compactness
            print(
                f"      '{box['text']}' at [{box['x']:.3g}, {box['y']:.3g}] (conf: {box['confidence']:.3g})"
            )

        return text_boxes

    except ImportError:
        # EasyOCR not installed, return empty list
        return []
    except Exception as e:
        # OCR failed, silently return empty list (agent will work without OCR)
        return []


def extract_detection_requests(text: str) -> List[str]:
    """Extract object detection requests from vision model response.

    Looks for patterns like:
    - REQUEST_DETECTION: red flag
    - REQUEST_DETECTION: enemy soldier

    Args:
        text: Vision model response text

    Returns:
        List of object names to detect
    """
    import re

    requests = []
    pattern = r"REQUEST_DETECTION:\s*([^\n]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)

    for match in matches:
        # Clean up the object name
        obj_name = match.strip()
        if obj_name and obj_name.lower() != "none":
            requests.append(obj_name)

    return requests


def find_text_in_ocr(
    target_text: str, ocr_results: List[Dict], confidence_threshold: float = 0.8
) -> Optional[Dict]:
    """Find target text in OCR results with fuzzy matching.

    Useful for direct OCR-based clicking without LLM.

    Args:
        target_text: Text to find (e.g., "Sora 2", "Start", "Submit")
        ocr_results: List of OCR boxes from extract_text_with_ocr()
        confidence_threshold: Minimum confidence for match (0.0-1.0)

    Returns:
        Best matching OCR box dict with 'x', 'y', 'text', 'confidence', or None
    """
    if not ocr_results or not target_text:
        return None

    target_lower = target_text.lower().strip()
    best_match = None
    best_score = 0.0

    for box in ocr_results:
        if box["confidence"] < confidence_threshold:
            continue

        ocr_text_lower = box["text"].lower().strip()

        # Exact match (highest priority)
        if ocr_text_lower == target_lower:
            return box

        # Contained match (e.g., target="Start" matches "Start Game")
        if target_lower in ocr_text_lower or ocr_text_lower in target_lower:
            score = box["confidence"] * (
                len(target_lower) / max(len(ocr_text_lower), len(target_lower))
            )
            if score > best_score:
                best_score = score
                best_match = box

    return best_match


def check_ocr_success(
    target_text: str, ocr_before: List[Dict], ocr_after: List[Dict]
) -> Tuple[bool, str]:
    """Check if a click action succeeded by comparing OCR results.

    Success indicators:
    - Target text position changed significantly
    - Target text disappeared (navigated away)
    - New page content appeared

    Args:
        target_text: The text that was clicked
        ocr_before: OCR results before click
        ocr_after: OCR results after click

    Returns:
        Tuple of (success: bool, reason: str)
    """
    if not ocr_before or not ocr_after:
        return False, "OCR data unavailable"

    # Find target in before/after
    target_before = find_text_in_ocr(target_text, ocr_before, confidence_threshold=0.5)
    target_after = find_text_in_ocr(target_text, ocr_after, confidence_threshold=0.5)

    # Case 1: Target disappeared (navigated to new page)
    if target_before and not target_after:
        return (
            True,
            f"Target '{target_text}' no longer visible (likely navigated to new page)",
        )

    # Case 2: Target position changed significantly (UI responded)
    if target_before and target_after:
        x_diff = abs(target_before["x"] - target_after["x"])
        y_diff = abs(target_before["y"] - target_after["y"])
        if x_diff > 0.1 or y_diff > 0.1:
            return True, f"Target '{target_text}' position changed significantly"

    # Case 3: Check if screen content changed significantly
    texts_before = set(
        box["text"].lower() for box in ocr_before if box["confidence"] > 0.5
    )
    texts_after = set(
        box["text"].lower() for box in ocr_after if box["confidence"] > 0.5
    )

    new_texts = texts_after - texts_before
    removed_texts = texts_before - texts_after

    if len(new_texts) > 3 or len(removed_texts) > 3:
        return (
            True,
            f"Screen content changed significantly ({len(new_texts)} new, {len(removed_texts)} removed text elements)",
        )

    return False, "No significant OCR changes detected"


# ---- Minimal helper to send an image to an LLM and get feedback ----
def analyze_screenshot(
    image_path: str,
    question: str = "Describe the screenshot briefly and list 3-5 salient UI elements.",
    model: str = "gpt-4o",
    include_ocr: bool = False,
) -> str:
    """Send an image to a vision-capable OpenAI model and return its feedback.

    Args:
        image_path: Local path to the image (jpg/png).
        question: Instruction for the LLM about how to analyze the image.
        model: OpenAI model that supports vision (e.g., gpt-4o, gpt-4.1).
        include_ocr: Whether to include OCR text detection results.
    Returns:
        The model's text response.
    """
    from openai import OpenAI  # lightweight import to keep module load fast

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")

    # Load and resize image
    img = Image.open(image_path)
    original_size = img.size
    img = resize_for_vision(img)
    resized_size = img.size

    if resized_size != original_size:
        print(
            f"   Image resized: {original_size} → {resized_size} (token optimization)"
        )

    # Build enhanced question with OCR data
    enhanced_question = question

    if include_ocr:
        text_boxes = extract_text_with_ocr(image_path)
        if text_boxes:
            ocr_info = "\n\nDETECTED TEXT ELEMENTS:\n"
            for idx, box in enumerate(text_boxes[:20], 1):  # Limit to top 20
                # Use 3 significant figures for coordinates and confidence
                ocr_info += f"{idx}. '{box['text']}' at [{box['x']:.3g}, {box['y']:.3g}] (conf: {box['confidence']:.3g})\n"
            enhanced_question = enhanced_question + ocr_info
            print(f"   Added {len(text_boxes[:20])} OCR text elements to vision prompt")
        else:
            print("   OCR returned no text elements")

    # Encode the resized image
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
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
        max_tokens=400,  # Reduced from 800 - be concise
    )

    response_text = response.choices[0].message.content

    # Log tokens and brief summary
    if hasattr(response, "usage") and response.usage:
        print(
            f"     [Vision] Tokens: {response.usage.prompt_tokens}→{response.usage.completion_tokens} ({response.usage.total_tokens} total)"
        )
    print(f"     [Vision] Response: {response_text[:100]}...")

    return response_text


def analyze_screenshot_with_detection(
    image_path: str,
    question: str,
    model: str = "gpt-4o",
    include_ocr: bool = False,
    enable_object_detection: bool = True,
) -> str:
    """
    Hybrid analysis: Vision model can request object detection for precise coordinates.

    Flow:
    1. Vision model analyzes screenshot (with OCR if enabled)
    2. If response contains "REQUEST_DETECTION: <object>", run object detector
    3. Annotate image with detection bounding boxes
    4. Re-analyze annotated image for final decision

    Args:
        image_path: Path to screenshot
        question: Vision prompt
        model: OpenAI vision model
        include_ocr: Include OCR text detection
        enable_object_detection: Allow vision to request object detection

    Returns:
        Final vision analysis (with detection coordinates if requested)
    """
    from .object_detector import detect_objects_smart, annotate_detections

    # Phase 1: Initial vision analysis
    initial_response = analyze_screenshot(image_path, question, model, include_ocr)

    # Check if object detection requested
    if not enable_object_detection:
        return initial_response

    detection_requests = extract_detection_requests(initial_response)

    if not detection_requests:
        # No detection needed
        return initial_response

    # Phase 2: Run object detection for requested objects
    print(f"   [Detection] Requested: {detection_requests}")

    all_detections = []
    for obj_name in detection_requests:
        detections = detect_objects_smart(image_path, obj_name)
        if detections:
            print(f"      Found {len(detections)} '{obj_name}' objects")
            all_detections.extend(detections)
        else:
            print(f"      No '{obj_name}' objects found")

    if not all_detections:
        print("   WARNING: No objects detected, using initial analysis")
        return initial_response

    # Phase 3: Annotate image with detections
    annotated_path = annotate_detections(image_path, all_detections)
    if not annotated_path:
        print("   WARNING: Annotation failed, using initial analysis")
        return initial_response

    print(f"   Annotated image with {len(all_detections)} detections")

    # Phase 4: Re-analyze with annotated image
    detection_summary = "\n\nDETECTED OBJECTS:\n"
    for idx, det in enumerate(all_detections, 1):
        # Use 3 significant figures for compact numeric output
        detection_summary += (
            f"{idx}. {det['object']} at [{det['x']:.3g}, {det['y']:.3g}] "
            f"(confidence: {det['confidence']:.3g})\n"
        )

    final_question = f"""{question}

{detection_summary}

The image now shows bounding boxes around detected objects with their coordinates.
Use these precise coordinates for actions."""

    final_response = analyze_screenshot(
        annotated_path, final_question, model, include_ocr=False
    )

    return final_response


if __name__ == "__main__":
    # Usage: python -m src.modules.information_gathering.info_gather /path/to/image.jpg "optional prompt"
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.modules.information_gathering.info_gather <image_path> [prompt]"
        )
        sys.exit(1)

    img_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) >= 3 else "Describe the screenshot briefly."

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(1)

    feedback = analyze_screenshot(image_path=img_path, question=prompt)
    print("Feedback:\n", feedback)
