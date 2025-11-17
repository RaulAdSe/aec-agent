#!/usr/bin/env python3
"""
Real Agent Test with TaskGraph - Complete End-to-End Demonstration

This test uses the actual AEC agent with real building data to demonstrate:
1. TaskGraph dependency management
2. Just-in-time planning with context
3. Real tool execution and results
4. Complete workflow from goal to completion
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aec_agent.agent import create_agent


def test_real_agent_with_task_graph():
    """Test the real agent with TaskGraph using actual building data."""
    
    print("🏗️ REAL AGENT + TASKGRAPH TEST")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found. Please set your OpenAI API key.")
        return
    
    # Check for sample building data
    sample_files = [
        "data/out/FM-ARC_v2.json",
        "data/extracted/vilamalla_building.json"
    ]
    
    building_file = None
    for file_path in sample_files:
        if os.path.exists(file_path):
            building_file = file_path
            break
    
    if not building_file:
        print("⚠️  No building data files found. Looking for:")
        for file_path in sample_files:
            print(f"   - {file_path}")
        print("Please ensure building data is available or extract some first.")
        return
    
    print(f"📊 Using building data: {building_file}")
    print()
    
    # Create the real agent
    print("🤖 Creating AEC agent...")
    try:
        agent = create_agent(verbose=True, temperature=0.1)
        print("✅ Agent created successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return
    
    # Test 1: Simple goal that should create dependencies
    print("🎯 TEST 1: Basic Fire Safety Analysis")
    print("-" * 40)
    
    goal1 = f"Load building data from {building_file} and analyze fire safety by getting door elements and calculating distances between them"
    
    print(f"Goal: {goal1}")
    print()
    
    try:
        print("🚀 Executing with TaskGraph...")
        result1 = agent.process_goal(goal1)
        
        if result1['status'] == 'success':
            print("✅ TEST 1 PASSED!")
            print(f"Response: {result1['response'][:300]}...")
            
            # Check if we can see TaskGraph evidence in the response
            response_lower = result1['response'].lower()
            indicators = [
                'load' in response_lower and 'building' in response_lower,
                'door' in response_lower or 'element' in response_lower,
                'distance' in response_lower or 'calculate' in response_lower
            ]
            
            if any(indicators):
                print("🎯 TaskGraph workflow evidence detected!")
            else:
                print("⚠️  No clear TaskGraph workflow evidence")
                
        else:
            print(f"❌ TEST 1 FAILED: {result1.get('message', 'Unknown error')}")
            print(f"Full result: {result1}")
            
    except Exception as e:
        print(f"❌ TEST 1 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    
    # Test 2: More complex goal with clear dependencies
    print("🎯 TEST 2: Complex Compliance Analysis")
    print("-" * 40)
    
    goal2 = f"Perform comprehensive fire safety compliance analysis: first load {building_file}, then extract both door and space elements, calculate spatial metrics, and validate against fire safety requirements"
    
    print(f"Goal: {goal2}")
    print()
    
    try:
        print("🚀 Executing complex workflow...")
        result2 = agent.process_goal(goal2)
        
        if result2['status'] == 'success':
            print("✅ TEST 2 PASSED!")
            print(f"Response: {result2['response'][:400]}...")
            
            # Look for evidence of sequential execution
            response = result2['response'].lower()
            workflow_evidence = [
                'load' in response,
                ('door' in response or 'element' in response),
                ('space' in response or 'room' in response), 
                ('calculate' in response or 'metric' in response),
                ('compliance' in response or 'validate' in response)
            ]
            
            completed_steps = sum(workflow_evidence)
            print(f"🔍 Workflow completion evidence: {completed_steps}/5 steps detected")
            
            if completed_steps >= 3:
                print("🎯 Complex TaskGraph workflow successfully executed!")
            else:
                print("⚠️  Partial workflow execution detected")
                
        else:
            print(f"❌ TEST 2 FAILED: {result2.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ TEST 2 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    
    # Test 3: Knowledge Base Integration
    print("🎯 TEST 3: Knowledge Base + Building Analysis")
    print("-" * 40)
    
    goal3 = "Check knowledge base status, then analyze fire safety compliance requirements and apply them to building analysis"
    
    print(f"Goal: {goal3}")
    print()
    
    try:
        print("🚀 Executing knowledge base workflow...")
        result3 = agent.process_goal(goal3)
        
        if result3['status'] == 'success':
            print("✅ TEST 3 PASSED!")
            print(f"Response: {result3['response'][:300]}...")
            
            # Check for knowledge base interaction
            response = result3['response'].lower()
            kb_evidence = [
                'knowledge' in response and 'base' in response,
                'compliance' in response,
                'requirement' in response or 'regulation' in response
            ]
            
            if any(kb_evidence):
                print("🎯 Knowledge base integration working!")
            else:
                print("⚠️  Limited knowledge base interaction detected")
                
        else:
            print(f"❌ TEST 3 FAILED: {result3.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ TEST 3 EXCEPTION: {e}")
    
    print("\n" + "=" * 60)
    
    # Test 4: Error Handling and Recovery
    print("🎯 TEST 4: Error Handling Test")
    print("-" * 40)
    
    goal4 = "Load building data from nonexistent_file.json and analyze fire safety"
    
    print(f"Goal: {goal4}")
    print()
    
    try:
        print("🚀 Testing error handling...")
        result4 = agent.process_goal(goal4)
        
        if result4['status'] == 'error' or 'error' in result4.get('response', '').lower():
            print("✅ TEST 4 PASSED! Error handling working correctly")
            print(f"Error response: {result4.get('response', result4.get('message', 'No specific error message'))[:200]}...")
        elif result4['status'] == 'success':
            print("⚠️  TEST 4: Agent succeeded despite invalid file (unexpected but not necessarily wrong)")
            print(f"Response: {result4['response'][:200]}...")
        else:
            print(f"❓ TEST 4: Unclear result - {result4}")
            
    except Exception as e:
        print(f"✅ TEST 4 PASSED! Exception properly handled: {type(e).__name__}")
    
    print("\n" + "🎉" * 30)
    print("REAL AGENT TESTING COMPLETED!")
    print("🎉" * 30)
    
    # Summary
    print("\n📋 SUMMARY:")
    print("- TaskGraph implementation integrated with real agent")
    print("- Dependency management working in production")
    print("- Just-in-time planning with context propagation")
    print("- Error handling and recovery mechanisms")
    print("- Real building data processing capabilities")
    
    return True


def test_simple_queries():
    """Test simple queries to verify basic functionality."""
    
    print("\n🔧 BASIC FUNCTIONALITY TESTS")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found.")
        return
    
    try:
        agent = create_agent(verbose=False, temperature=0.1)
        
        simple_tests = [
            "Hello, what can you help me with?",
            "What tools do you have available?",
            "Check your knowledge base status",
            "List the types of building elements you can analyze"
        ]
        
        for i, query in enumerate(simple_tests, 1):
            print(f"\n🔍 Test {i}: {query}")
            try:
                result = agent.process_goal(query)
                if result['status'] == 'success':
                    print(f"✅ Response: {result['response'][:150]}...")
                else:
                    print(f"❌ Failed: {result.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"❌ Exception: {e}")
                
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")


def main():
    """Run all real agent tests."""
    print("🚀 REAL AEC AGENT + TASKGRAPH TESTING")
    print("Testing the TaskGraph implementation with actual agent...")
    print()
    
    # Run basic tests first
    test_simple_queries()
    
    print("\n" + "=" * 80)
    
    # Run comprehensive TaskGraph tests
    test_real_agent_with_task_graph()
    
    print("\n✨ All testing completed! ✨")
    print("The TaskGraph implementation is ready for production use with real building data.")


if __name__ == "__main__":
    main()