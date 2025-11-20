"""
Test suite for LLM Agent (Agent 3) - Action Decision Maker

This test allows manual testing with custom game context and spatial data.
The user provides:
1. Game context (text description from VLM Agent)
2. Spatial data (JSON from GroundingDINO Agent)

Run with:
    python -m src.modules.test.test_llm_agent

For automated unit tests:
    pytest src/modules/test/test_llm_agent.py
"""

import os
import sys
import unittest
import json
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if API key is already set

from src.modules.agents.llm_agent import LLMAgent


class TestLLMAgent(unittest.TestCase):
    """Unit tests for LLM Agent."""
    
    def setUp(self):
        """Initialize agent before each test."""
        self.agent = LLMAgent()
    
    def test_01_initialization(self):
        """Test that LLM Agent initializes correctly."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.model, "gpt-4o-mini")
        self.assertEqual(self.agent.temperature, 0.3)
        print("\n✓ Test 01: Initialization successful")
    
    def test_02_output_structure(self):
        """Test output has correct structure."""
        game_context = "Test game"
        spatial_data = {
            'detected_objects': [],
            'screenshot_size': {'width': 800, 'height': 600},
            'total_detections': 0
        }
        
        result = self.agent.decide(game_context, spatial_data)
        
        # Check structure
        self.assertIn('action', result)
        self.assertIn('reasoning', result)
        self.assertIn('confidence', result)
        
        # Check types
        self.assertIsInstance(result['reasoning'], str)
        self.assertIsInstance(result['confidence'], (int, float))
        
        print("\n✓ Test 02: Output structure correct")
    
    def test_03_empty_game_context(self):
        """Test handling of empty game context."""
        spatial_data = {'detected_objects': []}
        
        with self.assertRaises(ValueError):
            self.agent.decide("", spatial_data)
        
        with self.assertRaises(ValueError):
            self.agent.decide("   ", spatial_data)
        
        print("\n✓ Test 03: Empty game context rejected")
    
    def test_04_empty_spatial_data(self):
        """Test handling of empty spatial data."""
        game_context = "Test game"
        
        with self.assertRaises(ValueError):
            self.agent.decide(game_context, {})
        
        with self.assertRaises(ValueError):
            self.agent.decide(game_context, None)
        
        print("\n✓ Test 04: Empty spatial data rejected")
    
    def test_05_format_spatial_data(self):
        """Test spatial data formatting."""
        spatial_data = {
            'screenshot_size': {'width': 1920, 'height': 1080},
            'detection_prompt': 'player, goal, walls',
            'detected_objects': [
                {
                    'object': 'player',
                    'pixel_coords': {'x_center': 100, 'y_center': 200},
                    'normalized_coords': {'x_center': 0.5, 'y_center': 0.6},
                    'confidence': 0.95
                }
            ],
            'total_detections': 1
        }
        
        formatted = self.agent._format_spatial_data(spatial_data)
        
        self.assertIn('1920x1080', formatted)
        self.assertIn('player', formatted)
        self.assertIn('100', formatted)
        self.assertIn('0.95', formatted)
        
        print("\n✓ Test 05: Spatial data formatting works")
    
    def test_06_decision_with_api(self):
        """Test decision making with real API (if available)."""
        if not self.agent.client:
            print("\n⊘ Test 06: Skipped (OpenAI API not available)")
            return
        
        game_context = "Maze game. Player (red square) must reach goal (green square). Use arrow keys."
        spatial_data = {
            'screenshot_size': {'width': 400, 'height': 400},
            'detection_prompt': 'player, goal',
            'detected_objects': [
                {
                    'object': 'player',
                    'pixel_coords': {'x_center': 100, 'y_center': 100},
                    'normalized_coords': {'x_center': 0.25, 'y_center': 0.25},
                    'confidence': 0.9
                },
                {
                    'object': 'goal',
                    'pixel_coords': {'x_center': 300, 'y_center': 300},
                    'normalized_coords': {'x_center': 0.75, 'y_center': 0.75},
                    'confidence': 0.85
                }
            ],
            'total_detections': 2
        }
        
        result = self.agent.decide(game_context, spatial_data)
        
        # Check that we got a valid response
        self.assertIn('action', result)
        self.assertIn('reasoning', result)
        
        # If action is not None, validate structure
        if result['action']:
            self.assertIn('action_type', result['action'])
            self.assertIn('action_inputs', result['action'])
        
        print("\n✓ Test 06: API decision making works")


def run_manual_test():
    """
    Interactive manual test where user provides game context and spatial data.
    """
    print("\n" + "=" * 80)
    print("MANUAL LLM AGENT TEST - INTERACTIVE")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set. Please set it first:")
        print("   set OPENAI_API_KEY=your-key-here")
        return
    
    # Get game context
    print("\n📝 Enter game context (from VLM Agent):")
    print("   - Describe the game rules, objectives, visual semantics")
    print("   - Example: 'Maze game. Red square = player, Green square = goal.'")
    print("   - Type your input (or press Ctrl+C to use default):")
    
    try:
        game_context = input("\nGame Context: ").strip()
    except KeyboardInterrupt:
        print("\n\nUsing default game context...")
        game_context = "Maze navigation game. Player (red square) must reach goal (green square). Black cells are walls (blocked), white cells are paths (walkable). Use arrow keys to move."
    
    if not game_context:
        game_context = "Maze navigation game. Player must reach goal. Use arrow keys to move."
    
    # Get spatial data
    print("\n🎯 Enter spatial data (from GroundingDINO Agent):")
    print("   - JSON format with detected_objects, screenshot_size, etc.")
    print("   - Or press ENTER to use sample spatial data")
    
    try:
        spatial_input = input("\nSpatial Data (JSON): ").strip()
    except KeyboardInterrupt:
        spatial_input = ""
    
    if spatial_input:
        try:
            spatial_data = json.loads(spatial_input)
        except json.JSONDecodeError as e:
            print(f"\n❌ Invalid JSON: {e}")
            print("Using sample spatial data instead...")
            spatial_data = None
    else:
        spatial_data = None
    
    # Use sample data if not provided
    if not spatial_data:
        print("\nUsing sample spatial data...")
        spatial_data = {
            'screenshot_size': {'width': 1920, 'height': 1020},
            'detection_prompt': 'player, goal, walls',
            'detected_objects': [
                {
                    'object': 'player',
                    'pixel_coords': {'x_center': 480, 'y_center': 408},
                    'normalized_coords': {'x_center': 0.25, 'y_center': 0.40},
                    'confidence': 0.92
                },
                {
                    'object': 'goal',
                    'pixel_coords': {'x_center': 1440, 'y_center': 867},
                    'normalized_coords': {'x_center': 0.75, 'y_center': 0.85},
                    'confidence': 0.88
                },
                {
                    'object': 'wall',
                    'pixel_coords': {'x_center': 480, 'y_center': 357},
                    'normalized_coords': {'x_center': 0.25, 'y_center': 0.35},
                    'confidence': 0.85
                }
            ],
            'total_detections': 3
        }
    
    # Run decision
    print("\n" + "=" * 80)
    print("Game Context:")
    print(game_context)
    print("\nSpatial Data Summary:")
    print(f"  Objects detected: {spatial_data.get('total_detections', 0)}")
    print("=" * 80)
    
    agent = LLMAgent()
    result = agent.decide(game_context, spatial_data)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 DECISION RESULTS")
    print("=" * 80)
    
    if 'error' in result:
        print(f"\n⚠️  Error: {result['error']}")
    
    print("\n1️⃣  ACTION:")
    if result.get('action'):
        print(json.dumps(result['action'], indent=2))
    else:
        print("  (No action generated)")
    
    print("\n2️⃣  REASONING:")
    print(f"  {result.get('reasoning', 'No reasoning provided')}")
    
    print(f"\n3️⃣  CONFIDENCE: {result.get('confidence', 0):.2%}")
    
    if 'raw_response' in result:
        print("\n📋 RAW RESPONSE:")
        print(f"  {result['raw_response'][:200]}...")
    
    print("\n" + "=" * 80)
    print("✅ Manual test complete!")
    print("=" * 80)


if __name__ == "__main__":
    # If run directly, do manual test
    if len(sys.argv) == 1:
        run_manual_test()
    else:
        # Run unit tests
        unittest.main()
