#!/usr/bin/env python3
"""
Agent Performance Evaluation - Test with real DigitalHub_FM-ARC_v2.ifc data

This script tests the agent on 5 queries of increasing complexity using the actual 
cached IFC data and evaluates the results.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aec_agent.agent import create_agent


def evaluate_agent_with_real_data():
    """Evaluate agent performance using real DigitalHub IFC data"""
    
    print("🧪 AGENT PERFORMANCE EVALUATION")
    print("=" * 50)
    print("Using: DigitalHub_FM-ARC_v2.ifc cached data")
    print("Fixed issues: Tool names, compliance workflow, task limit")
    print()
    
    # Check if cached data exists
    ifc_cache_path = Path("data/processed_ifc/DigitalHub_FM-ARC_v2.ifc.json")
    if not ifc_cache_path.exists():
        print("❌ ERROR: DigitalHub_FM-ARC_v2.ifc.json not found in cache")
        print("   Please upload the IFC file to the Streamlit app first")
        return False
    
    print(f"✅ Found cached IFC data: {ifc_cache_path}")
    
    # Set up agent
    print("\n🤖 Setting up agent...")
    agent = create_agent(
        model_name="gpt-5-mini",
        temperature=0.1,
        verbose=True,
        enable_memory=True,
        session_id="eval_test",
        max_iterations=25
    )
    
    # Set active file in agent memory
    if hasattr(agent, 'memory_manager') and agent.memory_manager:
        agent.memory_manager.track_active_file(str(ifc_cache_path))
        print(f"✅ Set active file in agent memory")
    
    # Test queries (easy to complex)
    test_queries = [
        {
            "level": 1,
            "name": "Basic Building Data",
            "query": "What elements are in this building?",
            "expected_tasks": "2-4",
            "expected_tools": ["load_building_data", "get_all_elements"]
        },
        {
            "level": 2, 
            "name": "Simple Compliance",
            "query": "How do I check if the doors are compliant?",
            "expected_tasks": "4-7",
            "expected_tools": ["load_building_data", "get_all_elements", "search_compliance_documents", "validate_rule"]
        },
        {
            "level": 3,
            "name": "Stair Compliance (Fixed Scenario)", 
            "query": "Are the stairs in this building compliant with building codes?",
            "expected_tasks": "5-8",
            "expected_tools": ["search_compliance_documents", "validate_rule"],
            "key_test": "Should NOT return 'none' tools"
        },
        {
            "level": 4,
            "name": "Multi-Element Analysis",
            "query": "Perform an accessibility audit of all doors and stairs in this building", 
            "expected_tasks": "6-8",
            "expected_tools": ["get_all_elements", "search_compliance_documents", "validate_rule"]
        },
        {
            "level": 5,
            "name": "Complex Fire Safety Analysis",
            "query": "Check if this building meets fire safety egress requirements including door widths, stair dimensions, and travel distances",
            "expected_tasks": "7-8", 
            "expected_tools": ["calculate", "search_compliance_documents", "validate_rule"]
        }
    ]
    
    results = []
    
    print("\n" + "🧪 RUNNING EVALUATIONS" + "\n" + "=" * 50)
    
    for test in test_queries:
        print(f"\n📋 LEVEL {test['level']}: {test['name']}")
        print(f"Query: '{test['query']}'")
        print(f"Expected: {test['expected_tasks']} tasks")
        
        start_time = time.time()
        
        try:
            # Run the query
            result = agent.process_goal(test['query'])
            execution_time = time.time() - start_time
            
            # Analyze results
            analysis = analyze_agent_result(result, test)
            analysis['execution_time'] = execution_time
            analysis['level'] = test['level']
            analysis['query'] = test['query']
            
            results.append(analysis)
            
            # Print immediate results
            status = "✅ PASS" if analysis['success'] else "❌ FAIL"
            print(f"{status} {analysis['task_count']} tasks, {execution_time:.1f}s")
            
            if analysis['compliance_workflow_ok']:
                print("   ✅ Compliance workflow: search → validate")
            
            if analysis['no_none_tools']:
                print("   ✅ No 'none' tool responses")
            
            if analysis['issues']:
                for issue in analysis['issues']:
                    print(f"   ⚠️ {issue}")
                    
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ FAIL - Exception: {str(e)}")
            results.append({
                'level': test['level'],
                'query': test['query'], 
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            })
    
    # Print summary
    print_evaluation_summary(results)
    
    return results

def analyze_agent_result(result, test_config):
    """Analyze agent execution result"""
    
    analysis = {
        'success': False,
        'task_count': 0,
        'tools_used': [],
        'compliance_workflow_ok': False,
        'no_none_tools': True,
        'issues': []
    }
    
    try:
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            
            # Extract basic info
            if 'summary' in reasoning:
                summary = reasoning['summary']
                analysis['task_count'] = summary.get('total_tasks', 0)
                analysis['completed_tasks'] = summary.get('completed_tasks', 0)
                analysis['success'] = reasoning.get('success', False)
            
            # Extract tools used
            if 'outputs' in reasoning:
                tools_sequence = []
                for output in reasoning['outputs']:
                    if 'tool' in output:
                        tool = output['tool']
                        tools_sequence.append(tool)
                        if tool == 'none':
                            analysis['no_none_tools'] = False
                            analysis['issues'].append("Found 'none' tool response")
                
                analysis['tools_used'] = tools_sequence
            
            # Check compliance workflow
            if any(word in test_config['query'].lower() for word in ['compliance', 'compliant', 'check']):
                search_idx = -1
                validate_idx = -1
                
                for i, tool in enumerate(analysis['tools_used']):
                    if 'search_compliance' in tool:
                        search_idx = i
                    if 'validate_rule' in tool:
                        validate_idx = i
                
                if search_idx >= 0 and validate_idx >= 0 and search_idx < validate_idx:
                    analysis['compliance_workflow_ok'] = True
                elif search_idx == -1 and validate_idx >= 0:
                    analysis['issues'].append("Validation without document search")
            else:
                analysis['compliance_workflow_ok'] = True  # N/A for non-compliance queries
            
            # Check task count expectations
            expected_range = test_config['expected_tasks']
            if '-' in expected_range:
                min_tasks, max_tasks = map(int, expected_range.split('-'))
                if not (min_tasks <= analysis['task_count'] <= max_tasks):
                    analysis['issues'].append(f"Task count {analysis['task_count']} outside expected range {expected_range}")
            
            # Check expected tools
            expected_tools = test_config.get('expected_tools', [])
            missing_tools = []
            for expected_tool in expected_tools:
                if not any(expected_tool in tool for tool in analysis['tools_used']):
                    missing_tools.append(expected_tool)
            
            if missing_tools:
                analysis['issues'].append(f"Missing expected tools: {missing_tools}")
            
            # Overall success
            analysis['success'] = (
                analysis['task_count'] > 0 and
                analysis['no_none_tools'] and
                analysis['compliance_workflow_ok'] and
                len(analysis['issues']) <= 1  # Allow 1 minor issue
            )
            
        else:
            analysis['issues'].append("Invalid result format")
            
    except Exception as e:
        analysis['issues'].append(f"Analysis error: {str(e)}")
    
    return analysis

def print_evaluation_summary(results):
    """Print comprehensive evaluation summary"""
    
    print("\n" + "📊 EVALUATION SUMMARY" + "\n" + "=" * 50)
    
    # Overall stats
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get('success', False))
    avg_tasks = sum(r.get('task_count', 0) for r in results) / total_tests if total_tests > 0 else 0
    avg_time = sum(r.get('execution_time', 0) for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"Tests Passed: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.0f}%)")
    print(f"Average Tasks: {avg_tasks:.1f}")
    print(f"Average Time: {avg_time:.1f}s")
    
    # Key fixes validation
    print(f"\n🔧 KEY FIXES VALIDATION:")
    
    # Check for 'none' tools
    any_none_tools = any(not r.get('no_none_tools', True) for r in results if 'no_none_tools' in r)
    print(f"   {'❌' if any_none_tools else '✅'} No 'none' tool responses")
    
    # Check compliance workflow
    compliance_tests = [r for r in results if 'compliance_workflow_ok' in r]
    compliance_ok = all(r.get('compliance_workflow_ok', False) for r in compliance_tests)
    print(f"   {'✅' if compliance_ok else '❌'} Compliance workflow: search → validate")
    
    # Check task limits
    task_counts = [r.get('task_count', 0) for r in results if r.get('task_count', 0) > 0]
    within_limits = all(3 <= count <= 8 for count in task_counts)
    print(f"   {'✅' if within_limits else '❌'} Task counts within 3-8 range")
    
    # Tool consistency
    any_tool_issues = any('tool' in ' '.join(r.get('issues', [])) for r in results)
    print(f"   {'✅' if not any_tool_issues else '❌'} Tool naming consistency")
    
    print(f"\n🎯 DETAILED RESULTS:")
    for result in results:
        if 'level' in result:
            status = "✅" if result.get('success') else "❌"
            level = result['level']
            tasks = result.get('task_count', 0)
            time_taken = result.get('execution_time', 0)
            print(f"   {status} Level {level}: {tasks} tasks, {time_taken:.1f}s")
            
            # Show issues
            issues = result.get('issues', [])
            if issues:
                for issue in issues[:2]:  # Show first 2 issues
                    print(f"      ⚠️ {issue}")
    
    # Final assessment
    print(f"\n🎉 OVERALL ASSESSMENT:")
    if passed_tests >= 4:
        print(f"   🟢 EXCELLENT: {passed_tests}/5 tests passed - Agent performs well!")
    elif passed_tests >= 3:
        print(f"   🟡 GOOD: {passed_tests}/5 tests passed - Agent mostly works, minor issues")
    elif passed_tests >= 2:
        print(f"   🟠 FAIR: {passed_tests}/5 tests passed - Some major issues remain")
    else:
        print(f"   🔴 POOR: {passed_tests}/5 tests passed - Significant problems")
    
    print(f"\n💡 The key test is Level 3 (stair compliance) - this was the original failing scenario!")

if __name__ == "__main__":
    try:
        results = evaluate_agent_with_real_data()
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()