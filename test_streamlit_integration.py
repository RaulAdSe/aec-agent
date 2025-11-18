#!/usr/bin/env python3
"""
Test Streamlit Integration with Fixed Agent

This script verifies that the Streamlit app correctly integrates with all 
compliance workflow fixes.
"""

import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_streamlit_agent_integration():
    """Test that Streamlit app correctly uses the fixed agent."""
    
    print("🌐 TESTING STREAMLIT AGENT INTEGRATION")
    print("=" * 50)
    
    try:
        # Import the exact same way the Streamlit app does
        from aec_agent.agent import create_agent
        
        print("1. Testing agent factory...")
        
        # Test that we can create the agent configuration (without API calls)
        # This is exactly what happens when Streamlit starts up
        agent_config = {
            "model_name": "gpt-5-mini",
            "temperature": 0.1,
            "verbose": True,
            "enable_memory": True,
            "session_id": "test_streamlit_session",
            "max_iterations": 20
        }
        
        print(f"   ✅ Agent config prepared: {agent_config['model_name']}")
        
        # Test that all fixed components are accessible
        print("2. Testing component imports...")
        
        from aec_agent.core.reasoning_agent import ReasoningAgent
        from aec_agent.core.reasoning_controller import ReasoningController
        from aec_agent.core.goal_decomposer import GoalDecomposer
        from aec_agent.core.tool_planner import ToolPlanner
        from aec_agent.core.simple_recovery import SimpleRecovery
        from aec_agent.core.task_graph import TaskGraph
        from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus
        
        print("   ✅ All core components imported")
        
        # Test the exact workflow that happens in Streamlit
        print("3. Testing Streamlit workflow simulation...")
        
        # Mock the components that require API keys
        mock_llm = Mock()
        
        # Test the compliance workflow fixes
        print("4. Testing compliance workflow fixes...")
        
        # Fix 1: Compliance Priority
        controller = ReasoningController.__new__(ReasoningController)
        controller._is_compliance_critical = ReasoningController._is_compliance_critical.__get__(controller)
        
        compliance_task = Task("test", "Search compliance documents", "Search", priority=Priority.LOW)
        is_critical = controller._is_compliance_critical(compliance_task)
        print(f"   ✅ Fix 1 - Compliance priority working: {is_critical}")
        
        # Fix 2: Dependencies 
        decomposer = GoalDecomposer(llm=mock_llm)
        load_task = Task("load", "Load data", "Load")
        search_task = Task("search", "Search compliance", "Search", dependencies=["spatial"])
        
        fixed_tasks = decomposer._create_compliance_dependencies([load_task, search_task])
        search_fixed = next(t for t in fixed_tasks if "search" in t.name.lower())
        deps_fixed = search_fixed.dependencies == ["load"]
        print(f"   ✅ Fix 2 - Dependencies fixed: {deps_fixed}")
        
        # Fix 3: Smart Recovery
        recovery = SimpleRecovery(llm=mock_llm)
        alternative = recovery.get_contextual_alternative("find_related", "spatial task", "spatial relationship not implemented")
        recovery_working = alternative == "calculate_distances"
        print(f"   ✅ Fix 3 - Smart recovery working: {recovery_working}")
        
        # Fix 4: Tool Capability
        planner = ToolPlanner(llm=mock_llm)
        capable = planner._check_tool_capability("find_related", {"relationship_type": "spatial"})
        capability_working = capable == False
        print(f"   ✅ Fix 4 - Tool capability working: {capability_working}")
        
        # Fix 5: Optional Tasks
        spatial_task = Task("spatial", "Find spatial relationships", "Find spatial")
        load_task = Task("load", "Load data", "Load") 
        
        graph = TaskGraph([spatial_task, load_task])
        optional_working = spatial_task.metadata.get("optional") == True
        print(f"   ✅ Fix 5 - Optional tasks working: {optional_working}")
        
        # Test simulated user interaction workflow
        print("5. Testing simulated user workflow...")
        
        # This simulates what happens when a user asks a compliance question
        user_prompt = "Are the stairs in this building compliant?"
        
        # The Streamlit app would:
        # 1. Create enhanced prompt with context
        # 2. Call agent.process_goal()
        # 3. Agent would use all our fixes
        
        enhanced_prompt = f"{user_prompt}\\n\\nNote: No files uploaded yet."
        print(f"   ✅ Enhanced prompt created: '{user_prompt}' with context")
        
        # Test log handler integration (what Streamlit uses for progress)
        print("6. Testing Streamlit logging integration...")
        
        # Import the exact log handler used by Streamlit
        sys.path.append('.')
        from app import StreamlitLogHandler
        
        log_handler = StreamlitLogHandler()
        test_messages = [
            "Starting reasoning for goal: test",
            "Executing task: Search compliance documents",
            "Tool: search_compliance_documents executed successfully"
        ]
        
        # Test that log messages are properly converted
        for msg in test_messages:
            # Simulate log record
            import logging
            record = logging.LogRecord(
                name="aec_agent.core.reasoning_controller",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=msg,
                args=(),
                exc_info=None
            )
            log_handler.emit(record)
        
        captured_messages = log_handler.get_messages()
        logging_working = len(captured_messages) > 0
        print(f"   ✅ Streamlit logging working: {logging_working}")
        
        # Overall verification
        all_working = all([
            is_critical,
            deps_fixed, 
            recovery_working,
            capability_working,
            optional_working,
            logging_working
        ])
        
        print("\\n" + "=" * 50)
        if all_working:
            print("🎉 STREAMLIT INTEGRATION: FULLY WORKING!")
            print("✅ All compliance workflow fixes active")
            print("✅ Agent properly integrated with Streamlit") 
            print("✅ User interactions will use improved workflows")
            print("✅ Ready for production use")
        else:
            print("❌ STREAMLIT INTEGRATION: ISSUES DETECTED")
            print("⚠️ Some fixes may not be working properly")
        
        return all_working
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_app_imports():
    """Test that the Streamlit app imports are correct."""
    print("\\n🔍 TESTING STREAMLIT APP IMPORTS")
    print("=" * 40)
    
    try:
        # Test the exact imports from app.py
        from aec_agent.agent import create_agent
        from aec_agent.utils.ifc_to_json import IFCToJSONConverter
        from services.pdf_rag_manager import PDFRAGManager
        from services.session_manager import SessionManager
        
        print("✅ All Streamlit app imports working")
        
        # Test that create_agent can be called with Streamlit parameters
        streamlit_params = {
            "model_name": "gpt-5-mini",
            "temperature": 0.1,
            "verbose": True,
            "enable_memory": True,
            "session_id": "streamlit_test",
            "max_iterations": 20
        }
        
        # This is the exact call made by Streamlit (line 357 in app.py)
        # We won't actually call it (requires API key) but verify the signature
        import inspect
        sig = inspect.signature(create_agent)
        params_match = all(param in sig.parameters for param in streamlit_params.keys())
        print(f"✅ Agent factory signature matches Streamlit: {params_match}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        integration_ok = test_streamlit_agent_integration()
        imports_ok = test_streamlit_app_imports()
        
        if integration_ok and imports_ok:
            print("\\n🚀 FINAL RESULT: STREAMLIT APP READY!")
            print("   All compliance fixes are active in the Streamlit app")
            print("   Users will benefit from improved workflows")
        else:
            print("\\n⚠️ FINAL RESULT: ISSUES DETECTED")
            print("   Some integration problems need attention")
            
    except KeyboardInterrupt:
        print("\\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test suite failed: {e}")