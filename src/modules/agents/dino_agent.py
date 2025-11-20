"""
GroundingDINO Agent (Agent 2) - Visual Element Localization

This agent receives:
- Screenshot path
- Detection prompt from VLM Agent (e.g., "player, walls, goal")

This agent outputs:
- Structured spatial data with bounding boxes and coordinates
- List of detected objects with positions, sizes, and confidence scores

The output is consumed by the LLM Agent for decision making.
"""

from typing import Dict, List, Optional
import os
from pathlib import Path


class DINOAgent:
    """
    Agent 2: GroundingDINO-based object detector for spatial localization.
    
    Uses zero-shot detection to identify and locate game elements based on
    text descriptions from the VLM Agent.
    """
    
    def __init__(self, box_threshold: float = 0.25, text_threshold: float = 0.20):
        """
        Initialize DINO Agent.
        
        Args:
            box_threshold: Minimum confidence for bounding box detection (default: 0.25)
            text_threshold: Minimum confidence for text matching (default: 0.20)
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Load GroundingDINO detector."""
        try:
            # Import using absolute path from src.modules
            import sys
            from pathlib import Path
            
            # Add modules directory to path if needed
            modules_dir = Path(__file__).parent.parent
            if str(modules_dir) not in sys.path:
                sys.path.insert(0, str(modules_dir))
            
            from information_gathering.object_detector import get_detector
            self.detector = get_detector()
            
            if self.detector and self.detector.available:
                print("[DINO Agent] ✓ GroundingDINO detector initialized")
            else:
                print("[DINO Agent] ⚠️  GroundingDINO not available")
        except Exception as e:
            print(f"[DINO Agent] ❌ Failed to initialize detector: {e}")
            self.detector = None
    
    def detect(
        self,
        screenshot_path: str,
        detection_prompt: str
    ) -> Dict[str, any]:
        """
        Detect objects in screenshot based on VLM Agent's prompt.
        
        Args:
            screenshot_path: Path to screenshot image
            detection_prompt: Text prompt from VLM Agent describing objects to detect
                            (e.g., "player, walls, goal, obstacles")
        
        Returns:
            Dictionary containing:
            - detected_objects: List of objects with spatial data
            - screenshot_size: Image dimensions (width, height)
            - detection_prompt: The prompt used for detection
            - total_detections: Count of detected objects
        """
        print(f"\n[DINO Agent] Detecting objects in: {Path(screenshot_path).name}")
        print(f"[DINO Agent] Detection prompt: {detection_prompt}")
        
        # Validate inputs
        if not os.path.exists(screenshot_path):
            raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")
        
        if not detection_prompt or not detection_prompt.strip():
            raise ValueError("Detection prompt cannot be empty")
        
        # Check detector availability
        if not self.detector or not self.detector.available:
            print("[DINO Agent] ⚠️  Detector not available, returning empty results")
            return {
                'detected_objects': [],
                'screenshot_size': self._get_image_size(screenshot_path),
                'detection_prompt': detection_prompt,
                'total_detections': 0,
                'error': 'GroundingDINO detector not available'
            }
        
        # Run detection
        try:
            detections = self.detector.detect(
                screenshot_path,
                detection_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            
            # Get image dimensions
            img_size = self._get_image_size(screenshot_path)
            
            # Convert to pixel coordinates
            detected_objects = []
            for det in detections:
                obj_data = {
                    'object': det['object'],
                    'normalized_coords': {
                        'x_center': det['x'],
                        'y_center': det['y'],
                        'width': det['width'],
                        'height': det['height']
                    },
                    'pixel_coords': self._to_pixel_coords(det, img_size),
                    'confidence': det['confidence']
                }
                detected_objects.append(obj_data)
            
            result = {
                'detected_objects': detected_objects,
                'screenshot_size': img_size,
                'detection_prompt': detection_prompt,
                'total_detections': len(detected_objects)
            }
            
            print(f"[DINO Agent] ✓ Detected {len(detected_objects)} objects")
            for obj in detected_objects:
                print(f"  → {obj['object']}: "
                      f"pos({obj['pixel_coords']['x_center']}, "
                      f"{obj['pixel_coords']['y_center']}) "
                      f"conf={obj['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            print(f"[DINO Agent] ❌ Detection failed: {e}")
            return {
                'detected_objects': [],
                'screenshot_size': self._get_image_size(screenshot_path),
                'detection_prompt': detection_prompt,
                'total_detections': 0,
                'error': str(e)
            }
    
    def _get_image_size(self, image_path: str) -> Dict[str, int]:
        """Get image dimensions."""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                return {'width': width, 'height': height}
        except Exception:
            return {'width': 0, 'height': 0}
    
    def _to_pixel_coords(
        self,
        detection: Dict,
        img_size: Dict[str, int]
    ) -> Dict[str, int]:
        """Convert normalized coordinates to pixel coordinates."""
        width = img_size['width']
        height = img_size['height']
        
        x_center = int(detection['x'] * width)
        y_center = int(detection['y'] * height)
        bbox_width = int(detection['width'] * width)
        bbox_height = int(detection['height'] * height)
        
        return {
            'x_center': x_center,
            'y_center': y_center,
            'width': bbox_width,
            'height': bbox_height,
            'x1': x_center - bbox_width // 2,
            'y1': y_center - bbox_height // 2,
            'x2': x_center + bbox_width // 2,
            'y2': y_center + bbox_height // 2
        }
    
    def annotate_detections(
        self,
        screenshot_path: str,
        detection_result: Dict,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Draw bounding boxes on screenshot for visualization.
        
        Args:
            screenshot_path: Original screenshot
            detection_result: Result from detect() method
            output_path: Where to save annotated image (auto-generated if None)
        
        Returns:
            Path to annotated image or None if failed
        """
        if not detection_result['detected_objects']:
            print("[DINO Agent] No objects to annotate")
            return None
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.open(screenshot_path)
            draw = ImageDraw.Draw(img)
            
            # Try to load a nice font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw each detection
            for obj in detection_result['detected_objects']:
                coords = obj['pixel_coords']
                x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw label
                label = f"{obj['object']} ({obj['confidence']:.2f})"
                text_bbox = draw.textbbox((x1, y1 - 22), label, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1 - 22), label, fill="white", font=font)
            
            # Save annotated image
            if output_path is None:
                base = Path(screenshot_path).stem
                output_path = str(Path(screenshot_path).parent / f"{base}_annotated.png")
            
            img.save(output_path)
            print(f"[DINO Agent] ✓ Annotated image saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[DINO Agent] ❌ Annotation failed: {e}")
            return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m src.modules.agents.dino_agent <screenshot> <prompt>")
        print("Example: python -m src.modules.agents.dino_agent maze.png 'player, walls, goal'")
        sys.exit(1)
    
    screenshot = sys.argv[1]
    prompt = sys.argv[2]
    
    # Run detection
    agent = DINOAgent()
    result = agent.detect(screenshot, prompt)
    
    # Print results
    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)
    print(f"Total detections: {result['total_detections']}")
    print(f"Screenshot size: {result['screenshot_size']}")
    
    if result['detected_objects']:
        print("\nDetected objects:")
        for obj in result['detected_objects']:
            print(f"\n  Object: {obj['object']}")
            print(f"  Confidence: {obj['confidence']:.2f}")
            print(f"  Position: ({obj['pixel_coords']['x_center']}, {obj['pixel_coords']['y_center']})")
            print(f"  Size: {obj['pixel_coords']['width']} x {obj['pixel_coords']['height']}")
    
    # Save annotated image
    agent.annotate_detections(screenshot, result)
