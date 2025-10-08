"""
Object detection module using YOLOv8/YOLOWorld for detecting visual elements.
Works on both Mac and Windows (CPU-compatible).
"""

from typing import List, Dict, Optional
from PIL import Image
import os


def detect_objects(
    image_path: str, target_objects: List[str], confidence_threshold: float = 0.3
) -> List[Dict[str, any]]:
    """Detect visual objects in screenshot using YOLO.

    Args:
        image_path: Path to screenshot
        target_objects: List of object names to detect (e.g., ["red flag", "button", "icon"])
        confidence_threshold: Minimum confidence score (0-1)

    Returns:
        List of detected objects with normalized coordinates [x, y, width, height]
    """
    try:
        from ultralytics import YOLO

        # Initialize YOLOv8 model (cached after first call)
        if not hasattr(detect_objects, "model"):
            print("   Initializing YOLO object detector (first-time setup)...")
            # Use YOLOv8n (nano) for speed on CPU
            detect_objects.model = YOLO("yolov8n.pt")

        model = detect_objects.model

        # Load image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Run detection
        results = model(image_path, verbose=False)

        # Parse results
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Skip if confidence too low
                if confidence < confidence_threshold:
                    continue

                # Check if this object matches any target
                matches_target = any(
                    target.lower() in class_name.lower() for target in target_objects
                )

                if (
                    matches_target or not target_objects
                ):  # Include all if no targets specified
                    # Normalize coordinates to [0,1]
                    center_x = ((x1 + x2) / 2) / img_width
                    center_y = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    detected_objects.append(
                        {
                            "object": class_name,
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height,
                            "confidence": confidence,
                        }
                    )

        return detected_objects

    except ImportError:
        print("   Warning: ultralytics not installed, object detection disabled")
        return []
    except Exception as e:
        print(f"   Warning: Object detection failed: {e}")
        return []


def detect_objects_by_description(
    image_path: str, descriptions: List[str], confidence_threshold: float = 0.2
) -> List[Dict[str, any]]:
    """Detect objects using text descriptions with YOLO-World.

    This is more flexible than standard YOLO as it can detect objects
    based on natural language descriptions without pre-training.

    Args:
        image_path: Path to screenshot
        descriptions: Text descriptions (e.g., ["red flag marker", "start button"])
        confidence_threshold: Minimum confidence score

    Returns:
        List of detected objects with normalized coordinates
    """
    try:
        from ultralytics import YOLOWorld

        # Initialize YOLO-World model (cached after first call)
        if not hasattr(detect_objects_by_description, "model"):
            print("   Initializing YOLO-World detector (first-time setup)...")
            detect_objects_by_description.model = YOLOWorld("yolov8s-world.pt")

        model = detect_objects_by_description.model

        # Set custom classes for detection
        model.set_classes(descriptions)

        # Load image to get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Run detection
        results = model(image_path, verbose=False)

        # Parse results
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = (
                    descriptions[class_id]
                    if class_id < len(descriptions)
                    else "unknown"
                )

                if confidence < confidence_threshold:
                    continue

                # Normalize coordinates
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                detected_objects.append(
                    {
                        "object": class_name,
                        "x": center_x,
                        "y": center_y,
                        "width": width,
                        "height": height,
                        "confidence": confidence,
                    }
                )

        return detected_objects

    except ImportError:
        print("   Warning: YOLOWorld not available, falling back to standard YOLO")
        return detect_objects(image_path, descriptions, confidence_threshold)
    except Exception as e:
        print(f"   Warning: Object detection by description failed: {e}")
        return []


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

    # Try YOLO-World first
    objects = detect_objects_by_description(image_path, [description])

    if objects:
        print(f"\nFound {len(objects)} objects:")
        for obj in objects:
            print(
                f"  - {obj['object']}: [{obj['x']:.3f}, {obj['y']:.3f}] (confidence: {obj['confidence']:.2f})"
            )
    else:
        print("No objects detected")
