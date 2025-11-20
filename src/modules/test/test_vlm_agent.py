"""
Test suite for VLM Agent (Agent 1).

Tests:
1. Input validation (screenshot path, task description)
2. Output structure (detection_prompt and game_context keys)
3. Output format (non-empty, reasonable length)
4. API integration (actual OpenAI call with real screenshot)
5. Dual output parsing (correct separation of detection vs context)
"""

import os
import sys
import unittest
from pathlib import Path
from PIL import Image
import tempfile

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if API key is already set

from src.modules.agents.vlm_agent import VLMAgent
from src.modules.screen_input.screen_capture import _enum_windows_windows, take_screenshot


class TestVLMAgent(unittest.TestCase):
    """Test cases for VLM Agent."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Check for API key
        cls.api_key = os.getenv("OPENAI_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("OPENAI_API_KEY not set")
        
        # Create a test screenshot (simple colored rectangle game-like image)
        cls.test_screenshot = tempfile.NamedTemporaryFile(
            suffix='.png', delete=False
        )
        cls.test_screenshot_path = cls.test_screenshot.name
        
        # Create a simple maze-like test image
        img = Image.new('RGB', (400, 400), color='white')
        pixels = img.load()
        
        # Draw blue player square (50x50) at top-left
        for x in range(50, 100):
            for y in range(50, 100):
                pixels[x, y] = (0, 0, 255)  # Blue
        
        # Draw red goal square (50x50) at bottom-right
        for x in range(300, 350):
            for y in range(300, 350):
                pixels[x, y] = (255, 0, 0)  # Red
        
        # Draw black walls
        for x in range(150, 250):
            for y in range(100, 300):
                pixels[x, y] = (0, 0, 0)  # Black wall
        
        img.save(cls.test_screenshot_path)
        cls.test_screenshot.close()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.test_screenshot_path):
            os.unlink(cls.test_screenshot_path)
    
    def test_01_agent_initialization(self):
        """Test VLM Agent can be initialized with API key."""
        agent = VLMAgent(model="gpt-4o-mini")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.model, "gpt-4o-mini")
        self.assertIsNotNone(agent.client)
    
    def test_02_agent_initialization_without_api_key(self):
        """Test VLM Agent raises error without API key."""
        # Temporarily remove API key
        original_key = os.environ.get("OPENAI_API_KEY")
        if original_key:
            del os.environ["OPENAI_API_KEY"]
        
        with self.assertRaises(ValueError) as context:
            VLMAgent()
        
        self.assertIn("OpenAI_API_KEY", str(context.exception))
        
        # Restore API key
        if original_key:
            os.environ["OpenAI_API_KEY"] = original_key
    
    def test_03_analyze_output_structure(self):
        """Test analyze() returns correct output structure."""
        agent = VLMAgent(model="gpt-4o-mini")
        result = agent.analyze(
            self.test_screenshot_path,
            task_description="navigate maze to reach goal"
        )
        
        # Check result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check required keys exist
        self.assertIn('detection_prompt', result)
        self.assertIn('game_context', result)
        self.assertIn('raw_analysis', result)
        
        # Check values are strings
        self.assertIsInstance(result['detection_prompt'], str)
        self.assertIsInstance(result['game_context'], str)
        self.assertIsInstance(result['raw_analysis'], str)
    
    def test_04_analyze_output_non_empty(self):
        """Test analyze() returns non-empty outputs."""
        agent = VLMAgent(model="gpt-4o-mini")
        result = agent.analyze(
            self.test_screenshot_path,
            task_description="simple maze game"
        )
        
        # Detection prompt should not be empty
        self.assertGreater(
            len(result['detection_prompt']),
            0,
            "Detection prompt should not be empty"
        )
        
        # Game context should not be empty
        self.assertGreater(
            len(result['game_context']),
            0,
            "Game context should not be empty"
        )
        
        # Check reasonable length (at least 10 characters each)
        self.assertGreater(
            len(result['detection_prompt']),
            10,
            "Detection prompt too short"
        )
        self.assertGreater(
            len(result['game_context']),
            10,
            "Game context too short"
        )
    
    def test_05_analyze_detection_prompt_format(self):
        """Test detection prompt contains relevant keywords."""
        agent = VLMAgent(model="gpt-4o-mini")
        result = agent.analyze(
            self.test_screenshot_path,
            task_description="maze navigation game"
        )
        
        detection_prompt = result['detection_prompt'].lower()
        
        # Should mention detection-related keywords
        # (at least one of these should appear)
        detection_keywords = [
            'detect', 'player', 'goal', 'wall', 'obstacle',
            'blue', 'red', 'black', 'square', 'object'
        ]
        
        found_keywords = [
            kw for kw in detection_keywords if kw in detection_prompt
        ]
        
        self.assertGreater(
            len(found_keywords),
            0,
            f"Detection prompt should contain relevant keywords. "
            f"Found: {found_keywords}"
        )
    
    def test_06_analyze_game_context_format(self):
        """Test game context contains game understanding."""
        agent = VLMAgent(model="gpt-4o-mini")
        result = agent.analyze(
            self.test_screenshot_path,
            task_description="reach the goal in maze"
        )
        
        game_context = result['game_context'].lower()
        
        # Should mention game-related concepts
        game_keywords = [
            'game', 'maze', 'goal', 'objective', 'player',
            'navigate', 'reach', 'avoid', 'rule'
        ]
        
        found_keywords = [
            kw for kw in game_keywords if kw in game_context
        ]
        
        self.assertGreater(
            len(found_keywords),
            0,
            f"Game context should contain game understanding. "
            f"Found: {found_keywords}"
        )
    
    def test_07_analyze_without_task_description(self):
        """Test analyze() works without task description."""
        agent = VLMAgent(model="gpt-4o-mini")
        result = agent.analyze(self.test_screenshot_path)
        
        # Should still return valid structure
        self.assertIn('detection_prompt', result)
        self.assertIn('game_context', result)
        self.assertGreater(len(result['detection_prompt']), 0)
        self.assertGreater(len(result['game_context']), 0)
    
    def test_08_analyze_dual_output_separation(self):
        """Test detection_prompt and game_context are different."""
        agent = VLMAgent(model="gpt-4o-mini")
        result = agent.analyze(
            self.test_screenshot_path,
            task_description="maze game with player and goal"
        )
        
        # The two outputs should be different
        # (not just copying the same text)
        self.assertNotEqual(
            result['detection_prompt'],
            result['game_context'],
            "Detection prompt and game context should be different"
        )
        
        # Detection prompt should be shorter (concise object list)
        # Game context should be longer (detailed explanation)
        # This is a heuristic but generally should hold
        print(f"\nDetection prompt length: "
              f"{len(result['detection_prompt'])}")
        print(f"Game context length: {len(result['game_context'])}")


def run_manual_test():
    """
    Manual test with real window/screen capture.
    Run with: python -m src.modules.test.test_vlm_agent
    
    Workflow:
    1. List available windows for user to select
    2. If selection made, capture that window; else capture full screen
    3. Save screenshot in same directory as test file
    4. Run VLM Agent analysis
    5. Display results in terminal
    """
    print("\n" + "=" * 80)
    print("MANUAL VLM AGENT TEST - WINDOW CAPTURE")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("OpenAI_API_KEY"):
        print("\n❌ OpenAI_API_KEY not set. Please set it first:")
        print("   export OpenAI_API_KEY='your-key-here'")
        return
    
    # List available windows
    print("\n🪟 Available windows:")
    if sys.platform.startswith("win"):
        windows = _enum_windows_windows()
        if windows:
            for idx, (hwnd, title) in enumerate(windows[:20], 1):
                print(f"   {idx}. {title}")
        else:
            print("   No windows detected")
            windows = []
    else:
        print("   (Window enumeration only supported on Windows)")
        windows = []
    
    # Prompt user for selection
    print("\n📋 Select a window to capture:")
    if windows:
        print(f"   - Enter window number (1-{len(windows[:20])})")
    print("   - Press ENTER for full screen capture")
    
    user_input = input("\nYour choice: ").strip()
    
    # Determine capture target
    screenshot_path = None
    test_dir = Path(__file__).parent
    
    if user_input and user_input.isdigit() and windows:
        idx = int(user_input) - 1
        if 0 <= idx < len(windows[:20]):
            _, window_title = windows[idx]
            print(f"\n📸 Capturing window: {window_title}")
            
            screenshot_path = test_dir / f"captured_window_{idx+1}.png"
            
            try:
                screenshot_file, _ = take_screenshot(
                    window_title=window_title,
                    output_dir=str(test_dir),
                    focus_window=True
                )
                # Convert jpg to png and rename
                img = Image.open(screenshot_file)
                img.save(str(screenshot_path))
                os.unlink(screenshot_file)
                print(f"   ✅ Saved to: {screenshot_path}")
            except Exception as e:
                print(f"   ❌ Error capturing window: {e}")
                screenshot_path = None
    
    # Fallback to full screen
    if screenshot_path is None:
        print("\n📸 Capturing full screen...")
        screenshot_path = test_dir / "captured_fullscreen.png"
        
        try:
            screenshot_file, _ = take_screenshot(
                screen_region=None,
                output_dir=str(test_dir)
            )
            # Convert jpg to png and rename
            img = Image.open(screenshot_file)
            img.save(str(screenshot_path))
            os.unlink(screenshot_file)
            print(f"   ✅ Saved to: {screenshot_path}")
        except Exception as e:
            print(f"   ❌ Error capturing screen: {e}")
            return
    
    # Run VLM Agent
    print("\n🤖 Initializing VLM Agent...")
    agent = VLMAgent(model="gpt-4o-mini")
    
    print("\n🔍 Analyzing screenshot...")
    result = agent.analyze(
        str(screenshot_path),
        task_description="Analyze the captured screen and identify key UI elements and context"
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    
    print("\n1️⃣  DETECTION PROMPT (Output to GroundingDINO Agent):")
    print("-" * 80)
    print(result['detection_prompt'])
    
    print("\n2️⃣  GAME CONTEXT (Output to LLM Agent):")
    print("-" * 80)
    print(result['game_context'])
    
    print("\n📋 RAW ANALYSIS (Debug):")
    print("-" * 80)
    print(result['raw_analysis'])
    
    print("\n📏 Statistics:")
    print(f"   Detection prompt length: {len(result['detection_prompt'])} chars")
    print(f"   Game context length: {len(result['game_context'])} chars")
    print(f"   Raw analysis length: {len(result['raw_analysis'])} chars")
    
    print("\n" + "=" * 80)
    print("✅ Manual test complete!")
    print(f"   Screenshot saved at: {screenshot_path}")
    print("=" * 80)


if __name__ == "__main__":
    # If run directly, do manual test for visual inspection
    if len(sys.argv) == 1:
        run_manual_test()
    else:
        # Run unit tests
        unittest.main()
