"""
Object detection using GroundingDINO for precise visual element localization.
Zero-shot detection with text prompts - works on any game or app without training.
"""

from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import warnings

# Suppress verbose warnings from torch/transformers (they clutter logs)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")
warnings.filterwarnings("ignore", message=".*device.*argument is deprecated.*")
warnings.filterwarnings("ignore", message=".*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")


# ============================================================================
# GroundingDINO Backend (Preferred)
# ============================================================================


class GroundingDINODetector:
    """Zero-shot object detector using GroundingDINO with text prompts."""

    def __init__(self):
        self.model = None
        self.available = False
        self._try_load()

    def _try_load(self):
        """Attempt to load GroundingDINO."""
        try:
            from groundingdino.util.inference import load_model, load_image, predict
            import torch

            config_path = "./cache/GroundingDINO_SwinB_cfg.py"
            checkpoint_path = "./cache/groundingdino_swinb_cogcoor.pth"

            if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
                print("   GroundingDINO model files not found in ./cache/")
                return

            # Force CPU mode for Mac compatibility (no CUDA)
            # Set default device before loading model
            torch.set_default_device("cpu")

            self.model = load_model(config_path, checkpoint_path)
            self.model = self.model.to("cpu")  # Ensure model is on CPU
            self.load_image = load_image
            self.predict = predict
            self.device = "cpu"
            self.available = True
            print(f"   GroundingDINO loaded (device: cpu)")

        except ImportError:
            print("   GroundingDINO not installed (pip install groundingdino-py)")
        except Exception as e:
            print(f"   GroundingDINO load failed: {e}")

    def detect(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List[Dict]:
        """Detect objects based on text prompt."""
        if not self.available:
            return []

        try:
            image_source, image = self.load_image(image_path)

            if not text_prompt.endswith("."):
                text_prompt = text_prompt + "."

            # Use CPU device
            boxes, logits, phrases = self.predict(
                model=self.model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
            )

            results = []
            for box, logit, phrase in zip(boxes, logits, phrases):
                x_center = float(box[0])
                y_center = float(box[1])
                width = float(box[2])
                height = float(box[3])

                results.append(
                    {
                        "object": phrase,
                        "x": round(x_center, 3),
                        "y": round(y_center, 3),
                        "width": round(width, 3),
                        "height": round(height, 3),
                        "confidence": round(float(logit), 3),
                    }
                )

            return results

        except Exception as e:
            print(f"   GroundingDINO detection failed: {e}")
            return []


# ============================================================================
# Unified Detection Interface
# ============================================================================

# Global detector instance
_detector = None


def get_detector() -> GroundingDINODetector:
    """Get or create singleton GroundingDINO detector."""
    global _detector
    if _detector is None:
        _detector = GroundingDINODetector()
    return _detector


def detect_objects_smart(
    image_path: str, text_prompt: str, confidence_threshold: float = 0.35
) -> List[Dict]:
    """
    Detect objects in image using text description.

    Args:
        image_path: Path to image
        text_prompt: Description (e.g., "red flag", "tower site", "enemy soldier")
        confidence_threshold: Minimum confidence score

    Returns:
        List of detected objects with normalized coordinates
    """
    detector = get_detector()
    if not detector.available:
        print(f"   Object detection disabled (GroundingDINO not available)")
        return []

    results = detector.detect(image_path, text_prompt, confidence_threshold)
    if results:
        print(f"   Found {len(results)} objects matching '{text_prompt}'")
    else:
        print(f"   No objects found matching '{text_prompt}'")

    return results


def annotate_detections(
    image_path: str, detections: List[Dict], output_path: Optional[str] = None
) -> Optional[str]:
    """
    Draw bounding boxes on image with detection results.

    Args:
        image_path: Path to original image
        detections: List of detection results
        output_path: Where to save (optional, auto-generated if None)

    Returns:
        Path to annotated image
    """
    if not detections:
        return None

    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()

        # Draw each detection
        for det in detections:
            # Calculate pixel coordinates from normalized
            x_center = det["x"] * width
            y_center = det["y"] * height
            w = det["width"] * width
            h = det["height"] * height

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label with coordinates
            label = f"{det['object']} [{det['x']:.2f}, {det['y']:.2f}]"

            # Background for text
            text_bbox = draw.textbbox((x1, y1 - 22), label, font=font)
            draw.rectangle(text_bbox, fill="red")
            draw.text((x1, y1 - 22), label, fill="white", font=font)

        # Save annotated image
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_detected{ext}"

        img.save(output_path)
        return output_path

    except Exception as e:
        print(f"   Annotation failed: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.modules.information_gathering.object_detector <image_path> [description]"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    description = sys.argv[2] if len(sys.argv) > 2 else "button"

    print(f"\nDetecting '{description}' in {image_path}...")

    # Use GroundingDINO detection
    objects = detect_objects_smart(image_path, description)

    if objects:
        print(f"\nFound {len(objects)} objects:")
        for obj in objects:
            print(
                f"  - {obj['object']}: [{obj['x']:.3f}, {obj['y']:.3f}] (confidence: {obj['confidence']:.2f})"
            )

        # Annotate image
        annotated = annotate_detections(image_path, objects)
        if annotated:
            print(f"\nAnnotated image saved: {annotated}")
    else:
        print("No objects detected")
