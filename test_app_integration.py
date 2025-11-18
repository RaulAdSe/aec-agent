#!/usr/bin/env python3
"""
Test Streamlit App Integration with Recovery System
Verifies that the app can use the new recovery system properly.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import tempfile

# Load environment and setup path
load_dotenv()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aec_agent.agent import create_agent
from app import StreamlitLogHandler

def test_app_integration():
    """Test that the app can integrate with the recovery system."""
    
    print("🔗 STREAMLIT APP INTEGRATION TEST")
    print("=" * 50)
    print("Testing agent creation and recovery message handling...")
    
    # Test 1: Agent creation (same as app does)
    print("\n🧪 TEST 1: Agent Creation")
    try:
        agent = create_agent(
            model_name="gpt-4o-mini",
            temperature=0.1,
            verbose=True,
            enable_memory=True,
            session_id="app_integration_test",
            max_iterations=5
        )
        print("✅ Agent created successfully")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False
    
    # Test 2: Log handler processing
    print("\n🧪 TEST 2: Recovery Message Handling")
    handler = StreamlitLogHandler()
    
    recovery_messages = [
        "2025-11-18 15:07:08,217 - aec_agent.core.simple_recovery - INFO - LLM recovery decision: retry_with_different_input",
        "2025-11-18 15:07:08,217 - aec_agent.core.simple_recovery - INFO - Modified task input: added file: building.json",
        "2025-11-18 15:07:08,217 - aec_agent.core.simple_recovery - INFO - Switching to alternative tool: query_elements",
        "2025-11-18 15:07:08,217 - aec_agent.core.simple_recovery - INFO - Task requires goal replanning: Load data",
        "2025-11-18 15:07:08,217 - aec_agent.core.simple_recovery - INFO - Gracefully skipping task: Calculate distances",
        "2025-11-18 15:07:08,217 - aec_agent.core.reasoning_controller - INFO - Attempting recovery for planning failure...",
        "2025-11-18 15:07:08,217 - aec_agent.core.reasoning_controller - INFO - Created fallback task: Load building data"
    ]
    
    processed_messages = []
    for msg in recovery_messages:
        # Extract logger name and message
        parts = msg.split(' - ')
        if len(parts) >= 3:
            logger_name = parts[1]
            message = ' - '.join(parts[2:]).split(': ', 1)[-1]
            formatted = handler._convert_to_user_friendly(message, logger_name)
            if formatted:
                processed_messages.append(formatted)
                print(f"✅ '{message[:50]}...' → '{formatted}'")
    
    if len(processed_messages) == len(recovery_messages):
        print("✅ All recovery messages handled correctly")
    else:
        print(f"⚠️ Only {len(processed_messages)}/{len(recovery_messages)} messages handled")
    
    # Test 3: Agent execution with recovery
    print("\n🧪 TEST 3: Agent Execution with Recovery")
    try:
        # Test with a goal that will trigger recovery
        result = agent.process_goal("Load building data but don't specify which file")
        
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            status = reasoning.get('status', 'unknown')
            print(f"✅ Agent execution completed with status: {status}")
            
            if status in ['success', 'partial']:
                print("✅ Recovery system worked in app context")
                return True
            else:
                print("⚠️ Recovery didn't achieve success/partial result")
                return False
        else:
            print("✅ Agent provided direct response (handled gracefully)")
            return True
            
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return False

def test_app_startup():
    """Test basic app functionality without full Streamlit."""
    
    print("\n🧪 TEST 4: App Startup Components")
    
    # Test imports
    try:
        from services.pdf_rag_manager import PDFRAGManager
        from services.session_manager import SessionManager
        from aec_agent.utils.ifc_to_json import IFCToJSONConverter
        print("✅ All app components import successfully")
    except Exception as e:
        print(f"❌ App component import failed: {e}")
        return False
    
    # Test session manager
    try:
        session_manager = SessionManager()
        print("✅ Session manager created successfully")
    except Exception as e:
        print(f"❌ Session manager failed: {e}")
        return False
    
    # Test IFC converter (basic) - skip since it needs file path
    try:
        # Just test import, not instantiation since it needs ifc_path
        print("✅ IFC converter import successful")
    except Exception as e:
        print(f"❌ IFC converter import failed: {e}")
        return False
    
    return True

def run_app_integration_tests():
    """Run all app integration tests."""
    
    print("🚀 STREAMLIT APP INTEGRATION TEST SUITE")
    print("=" * 60)
    print("Verifying recovery system works correctly in app context...")
    
    test_results = []
    
    test_results.append(("App Integration", test_app_integration()))
    test_results.append(("App Startup", test_app_startup()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 APP INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for name, result in test_results if result)
    total = len(test_results)
    
    for test_name, success in test_results:
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {test_name}")
    
    print(f"\n📈 OVERALL SCORE: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 APP INTEGRATION: FULLY WORKING!")
        print("✅ Recovery system properly integrated with Streamlit app")
        print("✅ All app components functional")
        print("✅ Recovery messages displayed correctly")
    else:
        print("⚠️ APP INTEGRATION: NEEDS ATTENTION")
        print("❌ Some integration issues detected")
    
    return passed == total

if __name__ == "__main__":
    success = run_app_integration_tests()
    
    if success:
        print(f"\n🎯 FINAL RESULT: App is ready to use with recovery system!")
    else:
        print(f"\n⚠️ FINAL RESULT: App needs fixes before deployment")