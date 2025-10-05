"""
Test script without LLM calls - validates basic module functionality.
"""

import os
import sys

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("Testing AI Gaming Agent Modules (No LLM)")
print("=" * 80)

# Test 1: Screen Capture
print("\n1️⃣ Testing Screen Capture...")
try:
    from src.modules.screen_input.screen_capture import (
        take_screenshot,
        get_fullscreen_region,
    )

    region = get_fullscreen_region()
    print("✅ Screen capture functions available")
    print(f"   Screen region: {region}")
except Exception as e:
    print(f"❌ ScreenCapture failed: {e}")

# Test 2: Memory Modules
print("\n2️⃣ Testing Memory Modules...")
try:
    from src.modules.memory.short_term import TaskState
    from src.modules.memory.long_term import SkillDatabase

    st = TaskState()
    st.initialize_task("Test task", ["Subtask 1", "Subtask 2"])
    print(f"✅ Short-term memory: {st.get_current_subtask()}")

    db = SkillDatabase()
    skill_id = db.add_skill(
        "test_skill",
        "A test skill",
        "def test_skill():\n    print('test')",
        "Test context",
    )
    print(f"✅ Long-term memory: Added skill {skill_id}")
    print(f"   Total skills: {db.get_skill_count()}")

except Exception as e:
    print(f"❌ Memory modules failed: {e}")

# Test 3: UI Automation Tools
print("\n3️⃣ Testing UI Automation...")
try:
    from src.modules.ui_automation.atomic_actions import UIAutomator
    from src.modules.memory import tools

    automator = UIAutomator()
    print("✅ UIAutomator initialized")

    # Check tool availability
    tool_names = [name for name in dir(tools) if not name.startswith("_")]
    print(f"✅ Available tools ({len(tool_names)}): {', '.join(tool_names[:5])}...")

except Exception as e:
    print(f"❌ UI Automation failed: {e}")

# Test 4: Object Detector (Placeholder)
print("\n4️⃣ Testing Object Detector...")
try:
    from src.modules.information_gathering.object_detector import ObjectDetector

    od = ObjectDetector()
    detections = od.detect_objects("test.jpg")
    print(f"✅ ObjectDetector initialized (placeholder mode)")
    print(f"   Detections: {len(detections)}")
except Exception as e:
    print(f"❌ Object Detector failed: {e}")

# Test 5: Test Executor
print("\n5️⃣ Testing Action Executor...")
try:
    from modules.ui_automation.executor import ActionExecutor
    import pyautogui

    w, h = pyautogui.size()
    executor = ActionExecutor(screen_width=w, screen_height=h)

    # Test wait action
    test_action = {"action_type": "wait", "action_inputs": {}}
    executor.execute(test_action)
    print("✅ ActionExecutor working")

except Exception as e:
    print(f"❌ Action Executor failed: {e}")

# Test 6: File Structure
print("\n6️⃣ Checking File Structure...")
expected_files = [
    "src/modules/screen_input/screen_capture.py",
    "src/modules/information_gathering/info_gather.py",
    "src/modules/information_gathering/object_detector.py",
    "src/modules/action_planning/planner.py",
    "src/modules/self_reflection/reflector.py",
    "src/modules/task_inference/task_breaker.py",
    "src/modules/skill_curation/skill_manager.py",
    "src/modules/memory/short_term.py",
    "src/modules/memory/long_term.py",
    "src/modules/memory/skill_retrieval.py",
    "src/modules/memory/tools.py",
    "src/modules/ui_automation/atomic_actions.py",
    "src/modules/test/executor.py",
    "src/modules/test/test_modules.py",
    "src/modules/main.py",
]

for file_path in expected_files:
    full_path = os.path.join(repo_root, file_path)
    if os.path.exists(full_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path} - MISSING")

print("\n" + "=" * 80)
print("🎉 Basic module tests complete!")
print("\nNote: LLM-dependent modules (info_gather, planner, reflector, etc.)")
print("      require OPENAI_API_KEY to test fully.")
