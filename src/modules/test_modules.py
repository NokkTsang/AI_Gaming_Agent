"""
Simple test to verify all modules can be imported and basic functionality works.
Run from project root: python -m src.modules.test_modules
"""

import sys
from pathlib import Path

# Add src directory to path so we can import from src.modules
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        from modules.screen_input.screen_capture import take_screenshot

        print("✓ screen_capture imported")
    except Exception as e:
        print(f"✗ screen_capture failed: {e}")
        return False

    try:
        from modules.information_gathering.info_gather import analyze_screenshot

        print("✓ info_gather imported")
    except Exception as e:
        print(f"✗ info_gather failed: {e}")
        return False

    try:
        from modules.action_planning.planner import ActionPlanner

        print("✓ action_planning imported")
    except Exception as e:
        print(f"✗ action_planning failed: {e}")
        return False

    try:
        from modules.ui_automation.automator import UIAutomator
        from modules.ui_automation.executor import ActionExecutor

        print("✓ ui_automation imported")
    except Exception as e:
        print(f"✗ ui_automation failed: {e}")
        return False

    return True


def test_screenshot():
    """Test screenshot capture."""
    print("\nTesting screenshot capture...")
    from modules.screen_input.screen_capture import take_screenshot

    try:
        path = take_screenshot()
        print(f"✓ Screenshot saved: {path}")
        return True
    except Exception as e:
        print(f"✗ Screenshot failed: {e}")
        return False


def test_action_parser():
    """Test action parsing."""
    print("\nTesting action execution...")
    from modules.ui_automation.executor import ActionExecutor
    from modules.ui_automation.automator import UIAutomator

    try:
        automator = UIAutomator()
        executor = ActionExecutor(1920, 1080, automator)

        # Test a simple wait action (safe)
        test_action = {"action_type": "wait", "action_inputs": {}}
        executor.execute(test_action)
        print("✓ Action execution works")
        return True
    except Exception as e:
        print(f"✗ Action execution failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("AI Gaming Agent - Module Tests")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_screenshot()
    all_passed &= test_action_parser()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Check output above.")
    print("=" * 60)
