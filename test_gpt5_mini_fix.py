#!/usr/bin/env python3
"""
Quick test of GPT-5 mini performance on stair compliance
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

def test_gpt5_mini_stair_compliance():
    """Test GPT-5 mini on the key stair compliance scenario"""
    
    print("🧪 TESTING GPT-5 MINI ON STAIR COMPLIANCE")
    print("=" * 45)
    
    # Check if cached data exists
    ifc_cache_path = Path("data/processed_ifc/DigitalHub_FM-ARC_v2.ifc.json")
    if not ifc_cache_path.exists():
        print("❌ ERROR: DigitalHub_FM-ARC_v2.ifc.json not found in cache")
        return False
    
    print(f"✅ Found cached IFC data")
    
    # Set up agent with GPT-5 mini
    agent = create_agent(
        model_name="gpt-5-mini",  # Using the upgraded model
        temperature=0.1,
        verbose=True,
        enable_memory=True,
        session_id="gpt5_mini_test",
        max_iterations=15
    )
    
    # Set active file in agent memory
    if hasattr(agent, 'memory_manager') and agent.memory_manager:
        agent.memory_manager.track_active_file(str(ifc_cache_path))
    
    query = "Are the stairs in this building compliant with building codes?"
    print(f"\n🎯 Query: {query}")
    print("\n🚀 Running with GPT-5 mini...")
    
    start_time = time.time()
    
    try:
        result = agent.process_goal(query)
        execution_time = time.time() - start_time
        
        print(f"\n⏱️ Execution time: {execution_time:.1f}s")
        
        # Analyze tools used
        tools_used = []
        task_names = []
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            
            # Get task information
            if 'summary' in reasoning:
                summary = reasoning['summary']
                task_count = summary.get('total_tasks', 0)
                completed_tasks = summary.get('completed_tasks', 0)
                print(f"📊 Tasks: {completed_tasks}/{task_count} completed")
            
            # Get tools and task names
            if 'outputs' in reasoning:
                for output in reasoning['outputs']:
                    if 'tool' in output:
                        tools_used.append(output['tool'])
                    if 'task_name' in output:
                        task_names.append(output['task_name'])
        
        print(f"\n🔧 Tools used: {tools_used}")
        if task_names:
            print(f"\n📋 Task breakdown:")
            for i, task in enumerate(task_names, 1):
                print(f"   {i}. {task}")
        
        # Check for key improvements
        has_search = any('search_compliance' in tool for tool in tools_used)
        has_validate = any('validate_rule' in tool for tool in tools_used)
        has_none = any(tool == 'none' for tool in tools_used)
        
        print(f"\n📈 GPT-5 Mini Analysis:")
        print(f"   {'✅' if has_search else '❌'} Document search performed")
        print(f"   {'✅' if has_validate else '❌'} Validation performed") 
        print(f"   {'✅' if not has_none else '❌'} No 'none' tool responses")
        
        # Check task naming quality
        validation_tasks = [task for task in task_names if 'validate' in task.lower()]
        search_tasks = [task for task in task_names if 'search' in task.lower()]
        
        print(f"\n🧠 Task Quality:")
        print(f"   Search tasks: {len(search_tasks)}")
        print(f"   Validation tasks: {len(validation_tasks)}")
        
        if validation_tasks:
            print(f"   Sample validation task: '{validation_tasks[0]}'")
        if search_tasks:
            print(f"   Sample search task: '{search_tasks[0]}'")
        
        # Overall assessment
        workflow_complete = has_search and has_validate and not has_none
        good_task_breakdown = len(validation_tasks) > 0 and len(search_tasks) > 0
        
        if workflow_complete and good_task_breakdown:
            print(f"\n🎉 EXCELLENT: Complete workflow with quality task breakdown")
            return True
        elif workflow_complete:
            print(f"\n✅ GOOD: Complete workflow, task breakdown could be improved")
            return True
        else:
            print(f"\n⚠️ PARTIAL: Workflow incomplete")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_gpt5_mini_stair_compliance()
    
    if success:
        print(f"\n🚀 GPT-5 MINI SHOWS IMPROVEMENT!")
        print(f"   The upgraded model provides better reasoning and task decomposition")
    else:
        print(f"\n⚠️ GPT-5 MINI NEEDS MORE TUNING")
        print(f"   The workflow still has some issues to address")