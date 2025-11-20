"""
Test suite for DINO Agent (Agent 2) - GroundingDINO Object Detector

This test allows manual testing with custom screenshots and prompts.
The user provides:
1. Screenshot filename (must be in the same directory as this test file)
2. Detection prompt (e.g., "player, walls, goal")

Run with:
    python -m src.modules.test.test_dino_agent

For automated unit tests:
    pytest src/modules/test/test_dino_agent.py
"""

import sys
import unittest
from pathlib import Path
from src.modules.agents.dino_agent import DINOAgent

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestDINOAgent(unittest.TestCase):
    """Unit tests for DINO Agent."""
    
    def setUp(self):
        """Initialize agent before each test."""
        self.agent = DINOAgent()
        self.test_dir = Path(__file__).parent
    
    def test_01_initialization(self):
        """Test that DINO Agent initializes correctly."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.box_threshold, 0.35)
        self.assertEqual(self.agent.text_threshold, 0.25)
        print("\n✓ Test 01: Initialization successful")
    
    def test_02_output_structure(self):
        """Test output has correct structure (even if detector unavailable)."""
        # Create a dummy image for testing
        test_img_path = self.test_dir / "dummy_test.png"
        
        # Create minimal test image if it doesn't exist
        if not test_img_path.exists():
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='white')
            img.save(test_img_path)
        
        result = self.agent.detect(
            str(test_img_path),
            "test object"
        )
        
        # Check structure
        self.assertIn('detected_objects', result)
        self.assertIn('screenshot_size', result)
        self.assertIn('detection_prompt', result)
        self.assertIn('total_detections', result)
        
        # Check types
        self.assertIsInstance(result['detected_objects'], list)
        self.assertIsInstance(result['screenshot_size'], dict)
        self.assertIsInstance(result['detection_prompt'], str)
        self.assertIsInstance(result['total_detections'], int)
        
        # Cleanup
        if test_img_path.exists():
            test_img_path.unlink()
        
        print("\n✓ Test 02: Output structure correct")
    
    def test_03_invalid_screenshot(self):
        """Test handling of invalid screenshot path."""
        with self.assertRaises(FileNotFoundError):
            self.agent.detect("nonexistent.png", "test")
        print("\n✓ Test 03: Invalid screenshot handled correctly")
    
    def test_04_empty_prompt(self):
        """Test handling of empty detection prompt."""
        # Create dummy image
        test_img_path = self.test_dir / "dummy_test.png"
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='white')
        img.save(test_img_path)
        
        with self.assertRaises(ValueError):
            self.agent.detect(str(test_img_path), "")
        
        with self.assertRaises(ValueError):
            self.agent.detect(str(test_img_path), "   ")
        
        # Cleanup
        test_img_path.unlink()
        print("\n✓ Test 04: Empty prompt rejected correctly")
    
    def test_05_screenshot_size_detection(self):
        """Test that screenshot size is correctly detected."""
        from PIL import Image
        
        # Create test image with known size
        test_img_path = self.test_dir / "size_test.png"
        img = Image.new('RGB', (640, 480), color='blue')
        img.save(test_img_path)
        
        result = self.agent.detect(str(test_img_path), "test")
        
        self.assertEqual(result['screenshot_size']['width'], 640)
        self.assertEqual(result['screenshot_size']['height'], 480)
        
        # Cleanup
        test_img_path.unlink()
        print("\n✓ Test 05: Screenshot size detected correctly")
    
    def test_06_detection_result_format(self):
        """Test that detection results have correct format when objects
        found."""
        # This test only runs if GroundingDINO is available
        if not self.agent.detector or not self.agent.detector.available:
            print("\n⊘ Test 06: Skipped (GroundingDINO not available)")
            return
        
        # If detector is available, check result format
        from PIL import Image
        test_img_path = self.test_dir / "format_test.png"
        img = Image.new('RGB', (400, 400), color='white')
        # Draw a red square that might be detected
        pixels = img.load()
        for x in range(100, 200):
            for y in range(100, 200):
                pixels[x, y] = (255, 0, 0)
        img.save(test_img_path)
        
        result = self.agent.detect(str(test_img_path), "red square")
        
        # If objects detected, check their format
        for obj in result['detected_objects']:
            self.assertIn('object', obj)
            self.assertIn('normalized_coords', obj)
            self.assertIn('pixel_coords', obj)
            self.assertIn('confidence', obj)
            
            # Check normalized coords structure
            norm = obj['normalized_coords']
            self.assertIn('x_center', norm)
            self.assertIn('y_center', norm)
            self.assertIn('width', norm)
            self.assertIn('height', norm)
            
            # Check pixel coords structure
            pix = obj['pixel_coords']
            self.assertIn('x_center', pix)
            self.assertIn('y_center', pix)
            self.assertIn('width', pix)
            self.assertIn('height', pix)
            self.assertIn('x1', pix)
            self.assertIn('y1', pix)
            self.assertIn('x2', pix)
            self.assertIn('y2', pix)
        
        # Cleanup
        test_img_path.unlink()
        print("\n✓ Test 06: Detection result format correct")


def run_manual_test():
    """
    Interactive manual test where user provides screenshot and prompt.
    
    The screenshot must be in the same directory as this test file.
    """
    print("\n" + "=" * 80)
    print("MANUAL DINO AGENT TEST - INTERACTIVE")
    print("=" * 80)
    
    test_dir = Path(__file__).parent
    
    # List available screenshots in test directory
    print("\n📸 Available screenshots in test directory:")
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(test_dir.glob(ext))
    
    if image_files:
        for idx, img_path in enumerate(sorted(image_files), 1):
            print(f"   {idx}. {img_path.name}")
    else:
        print("   (No images found)")
    
    # Get screenshot filename from user
    print("\n📋 Enter screenshot filename:")
    print("   - Type the exact filename (e.g., captured_window_3.png)")
    print("   - Or enter the number from the list above")
    print("   - Screenshot must be in the test directory")
    
    screenshot_input = input("\nScreenshot: ").strip()
    
    # Determine screenshot path
    screenshot_path = None
    if screenshot_input.isdigit() and image_files:
        idx = int(screenshot_input) - 1
        if 0 <= idx < len(image_files):
            screenshot_path = image_files[idx]
    else:
        screenshot_path = test_dir / screenshot_input
    
    if not screenshot_path or not screenshot_path.exists():
        print(f"\n❌ Screenshot not found: {screenshot_input}")
        print(f"   Make sure the file is in: {test_dir}")
        return
    
    # Get detection prompt from user
    print("\n🔍 Enter detection prompt:")
    print("   - Describe objects to detect (e.g., 'player, walls, goal')")
    print("   - Use simple, clear terms")
    print("   - Separate multiple objects with commas")
    
    detection_prompt = input("\nPrompt: ").strip()
    
    if not detection_prompt:
        print("\n❌ Detection prompt cannot be empty")
        return
    
    # Run detection
    print("\n" + "=" * 80)
    print(f"Screenshot: {screenshot_path.name}")
    print(f"Prompt: {detection_prompt}")
    print("=" * 80)
    
    agent = DINOAgent()
    result = agent.detect(str(screenshot_path), detection_prompt)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 DETECTION RESULTS")
    print("=" * 80)
    
    width = result['screenshot_size']['width']
    height = result['screenshot_size']['height']
    print(f"\nScreenshot size: {width} x {height}")
    print(f"Total detections: {result['total_detections']}")
    
    if 'error' in result:
        print(f"\n⚠️  Error: {result['error']}")
    
    if result['detected_objects']:
        print("\n✓ Detected objects:")
        for idx, obj in enumerate(result['detected_objects'], 1):
            print(f"\n  {idx}. {obj['object'].upper()}")
            print(f"     Confidence: {obj['confidence']:.2%}")
            x_center = obj['pixel_coords']['x_center']
            y_center = obj['pixel_coords']['y_center']
            print(f"     Position: ({x_center}, {y_center})")
            width = obj['pixel_coords']['width']
            height = obj['pixel_coords']['height']
            print(f"     Size: {width} x {height}")
            x1 = obj['pixel_coords']['x1']
            y1 = obj['pixel_coords']['y1']
            x2 = obj['pixel_coords']['x2']
            y2 = obj['pixel_coords']['y2']
            print(f"     Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
            norm_x = obj['normalized_coords']['x_center']
            norm_y = obj['normalized_coords']['y_center']
            print(f"     Normalized: ({norm_x:.3f}, {norm_y:.3f})")
    else:
        print("\n✗ No objects detected")
        print("   Try adjusting the prompt or using different thresholds")
    
    # Save annotated image
    if result['detected_objects']:
        print("\n📝 Saving annotated image...")
        annotated_path = agent.annotate_detections(
            str(screenshot_path), result
        )
        if annotated_path:
            print(f"   ✓ Saved: {annotated_path}")
    
    print("\n" + "=" * 80)
    print("✅ Manual test complete!")
    print("=" * 80)


if __name__ == "__main__":
    # If run directly, do manual test for user input
    if len(sys.argv) == 1:
        run_manual_test()
    else:
        # Run unit tests
        unittest.main()
