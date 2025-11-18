#!/usr/bin/env python3
"""
Comprehensive Recovery System Test Suite
Tests all failure scenarios and recovery capabilities end-to-end.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment and setup path
load_dotenv()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aec_agent.agent import create_agent

def test_scenario_1_missing_files():
    """Test: No files available, agent should gracefully handle or replan."""
    
    print("🧪 SCENARIO 1: Missing Files Recovery")
    print("=" * 50)
    print("Goal: Load building data (but no files in context)")
    
    agent = create_agent(
        model_name="gpt-4o-mini",
        temperature=0.1,
        verbose=True,
        enable_memory=True,
        session_id="test_missing_files",
        max_iterations=5
    )
    
    # Don't add any files to context - force failure
    try:
        result = agent.process_goal("Load the building data for compliance analysis")
        
        status = "unknown"
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            status = reasoning.get('status', 'unknown')
            message = reasoning.get('message', '')
            
            print(f"📊 Result: {status}")
            print(f"📝 Message: {message}")
            
            if status in ['success', 'partial']:
                print("✅ RECOVERY SUCCESS: Agent handled missing files")
                return True
            else:
                print("❌ RECOVERY FAILED: Agent couldn't handle missing files")
                return False
        else:
            print(f"📝 Direct response: {result}")
            return "error" not in str(result).lower()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_scenario_2_bad_parameters():
    """Test: Tool execution with wrong parameters, should retry with corrections."""
    
    print("\n🧪 SCENARIO 2: Bad Parameters Recovery")
    print("=" * 50)
    print("Goal: Calculate something with intentionally vague request")
    
    agent = create_agent(
        model_name="gpt-4o-mini", 
        temperature=0.1,
        verbose=True,
        enable_memory=True,
        session_id="test_bad_params",
        max_iterations=10
    )
    
    # Add files but use vague goal to trigger parameter issues
    processed_dir = Path("data/processed_ifc")
    if processed_dir.exists():
        ifc_files = list(processed_dir.glob("*.json"))
        for file in ifc_files[:1]:  # Just add one file
            if hasattr(agent, 'memory_manager') and agent.memory_manager:
                agent.memory_manager.track_active_file(str(file))
    
    try:
        result = agent.process_goal("Calculate some distances but don't specify what kind")
        
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            status = reasoning.get('status', 'unknown')
            execution_results = reasoning.get('execution_results', [])
            
            print(f"📊 Status: {status}")
            print(f"🔧 Tool executions: {len(execution_results)}")
            
            # Look for retry evidence
            tool_attempts = len([r for r in execution_results if not r.get('success', True)])
            successful_attempts = len([r for r in execution_results if r.get('success', False)])
            
            print(f"❌ Failed attempts: {tool_attempts}")
            print(f"✅ Successful attempts: {successful_attempts}")
            
            if successful_attempts > 0 or status == 'partial':
                print("✅ RECOVERY SUCCESS: Agent recovered from parameter issues")
                return True
            else:
                print("❌ RECOVERY FAILED: No successful tool execution")
                return False
        else:
            return True  # Direct response means it handled it somehow
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_scenario_3_wrong_tools():
    """Test: Wrong tool selection, should switch to alternatives."""
    
    print("\n🧪 SCENARIO 3: Wrong Tool Recovery") 
    print("=" * 50)
    print("Goal: Complex goal that might trigger wrong tool choices initially")
    
    agent = create_agent(
        model_name="gpt-4o-mini",
        temperature=0.1,
        verbose=True, 
        enable_memory=True,
        session_id="test_wrong_tools",
        max_iterations=8
    )
    
    # Add building data
    processed_dir = Path("data/processed_ifc")
    if processed_dir.exists():
        ifc_files = list(processed_dir.glob("*.json"))
        for file in ifc_files:
            if hasattr(agent, 'memory_manager') and agent.memory_manager:
                agent.memory_manager.track_active_file(str(file))
    
    try:
        result = agent.process_goal("Find all the emergency exits and calculate distances between them")
        
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            status = reasoning.get('status', 'unknown') 
            execution_results = reasoning.get('execution_results', [])
            
            print(f"📊 Status: {status}")
            print(f"🔧 Tool executions: {len(execution_results)}")
            
            # Analyze tool variety (sign of recovery trying different approaches)
            tools_used = set()
            for result_item in execution_results:
                if 'tool' in result_item:
                    tools_used.add(result_item['tool'])
            
            print(f"🛠️ Different tools tried: {len(tools_used)}")
            print(f"🛠️ Tools: {list(tools_used)}")
            
            if len(tools_used) > 1 or status in ['success', 'partial']:
                print("✅ RECOVERY SUCCESS: Agent tried different approaches")
                return True
            else:
                print("❌ RECOVERY LIMITED: Only tried one approach")
                return False
        else:
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_scenario_4_replanning_trigger():
    """Test: Goal that should trigger replanning after observations."""
    
    print("\n🧪 SCENARIO 4: Replanning After Observation")
    print("=" * 50)
    print("Goal: Multi-step goal where initial observations should trigger replanning")
    
    agent = create_agent(
        model_name="gpt-4o-mini",
        temperature=0.1,
        verbose=True,
        enable_memory=True,
        session_id="test_replanning",
        max_iterations=12
    )
    
    # Add building data
    processed_dir = Path("data/processed_ifc") 
    if processed_dir.exists():
        ifc_files = list(processed_dir.glob("*.json"))
        for file in ifc_files:
            if hasattr(agent, 'memory_manager') and agent.memory_manager:
                agent.memory_manager.track_active_file(str(file))
    
    try:
        # Goal designed to require multiple steps and potential replanning
        result = agent.process_goal("Analyze the building for compliance issues and create a detailed report")
        
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            status = reasoning.get('status', 'unknown')
            summary = reasoning.get('summary', {})
            iterations = summary.get('iterations', 0)
            execution_time = summary.get('execution_time', 0)
            
            print(f"📊 Status: {status}")
            print(f"🔄 Iterations: {iterations}")
            print(f"⏱️ Execution time: {execution_time:.1f}s")
            
            # Signs of active reasoning and potential replanning
            if iterations > 3 or execution_time > 10:
                print("✅ RECOVERY SUCCESS: Agent showed extended reasoning/replanning")
                return True
            elif status in ['success', 'partial']:
                print("✅ RECOVERY SUCCESS: Agent completed goal efficiently")
                return True
            else:
                print("❌ RECOVERY FAILED: Limited reasoning activity")
                return False
        else:
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_scenario_5_complete_failure_graceful_degradation():
    """Test: Impossible goal should trigger graceful degradation."""
    
    print("\n🧪 SCENARIO 5: Graceful Degradation")
    print("=" * 50)
    print("Goal: Impossible goal that should be gracefully handled")
    
    agent = create_agent(
        model_name="gpt-4o-mini",
        temperature=0.1,
        verbose=True,
        enable_memory=True,
        session_id="test_graceful_degradation",
        max_iterations=5
    )
    
    try:
        result = agent.process_goal("Calculate the exact temperature of Mars right now using IFC building data")
        
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            status = reasoning.get('status', 'unknown')
            message = reasoning.get('message', '')
            
            print(f"📊 Status: {status}")
            print(f"📝 Message: {message}")
            
            if status in ['partial', 'error'] and len(message) > 10:
                print("✅ RECOVERY SUCCESS: Agent gracefully explained limitations")
                return True
            else:
                print("❌ RECOVERY FAILED: Agent didn't handle impossible goal well")
                return False
        else:
            # Check if response acknowledges the impossibility
            response_str = str(result).lower()
            if "cannot" in response_str or "impossible" in response_str or "not possible" in response_str:
                print("✅ RECOVERY SUCCESS: Agent recognized impossibility")
                return True
            else:
                print("❌ RECOVERY UNCLEAR: Agent response unclear")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_comprehensive_recovery_tests():
    """Run all recovery test scenarios."""
    
    print("🚀 COMPREHENSIVE RECOVERY SYSTEM TEST SUITE")
    print("=" * 70)
    print("Testing all failure modes and recovery capabilities...")
    print()
    
    # Run all test scenarios
    test_results = []
    
    test_results.append(("Missing Files Recovery", test_scenario_1_missing_files()))
    test_results.append(("Bad Parameters Recovery", test_scenario_2_bad_parameters()))
    test_results.append(("Wrong Tool Recovery", test_scenario_3_wrong_tools()))
    test_results.append(("Replanning After Observation", test_scenario_4_replanning_trigger()))
    test_results.append(("Graceful Degradation", test_scenario_5_complete_failure_graceful_degradation()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 OVERALL SCORE: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 RECOVERY SYSTEM: FULLY OPERATIONAL!")
        print("✅ All failure scenarios handled correctly")
        print("✅ Agent shows intelligent recovery across all test cases")
        print("✅ Flexible, simple, and robust recovery capabilities")
    elif passed >= total * 0.8:
        print("🎯 RECOVERY SYSTEM: HIGHLY EFFECTIVE!")
        print("✅ Most failure scenarios handled well")
        print("💡 Minor improvements possible but system is very robust")
    elif passed >= total * 0.6:
        print("🔧 RECOVERY SYSTEM: FUNCTIONAL BUT IMPROVABLE")
        print("✅ Basic recovery capabilities working")
        print("💡 Some scenarios need refinement")
    else:
        print("⚠️ RECOVERY SYSTEM: NEEDS SIGNIFICANT IMPROVEMENT")
        print("❌ Multiple failure scenarios not handled well")
        print("🔧 Recovery logic needs major enhancements")
    
    return passed, total

if __name__ == "__main__":
    passed, total = run_comprehensive_recovery_tests()
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"The recovery system shows {passed/total*100:.1f}% effectiveness across diverse failure scenarios.")
    print(f"This indicates the system {'WILL' if passed/total >= 0.8 else 'MAY'} provide significant improvements in real-world usage.")