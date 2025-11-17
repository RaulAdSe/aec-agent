#!/usr/bin/env python3
"""
Test Door Distance Analysis - Simple test to check distances between doors.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('"\'')
                os.environ[key] = value

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent))

from aec_agent import ReasoningAgent, AgentConfig, AgentProfile


def test_door_distances():
    """Test calculating distances between all doors in the building."""
    
    print("🚪 Door Distance Analysis Test")
    print("=" * 50)
    
    # Check for building data file
    building_data_path = "data/out/FM-ARC_v2.json"
    full_path = Path(building_data_path)
    
    if not full_path.exists():
        print(f"❌ Building data file not found: {building_data_path}")
        return False
    
    print(f"✅ Building data found: {building_data_path}")
    print(f"   File size: {full_path.stat().st_size / 1024:.1f} KB")
    
    # Create configuration for development
    config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
    config.logging.enable_langsmith = True
    config.logging.langsmith_project = "Door-Distance-Analysis"
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.llm.model_name}")
    print(f"  LangSmith Project: {config.logging.langsmith_project}")
    
    try:
        print(f"\n🤖 Creating AEC Reasoning Agent...")
        agent = ReasoningAgent(config=config)
        print("✅ Agent created successfully!")
        
        # Test goal: Calculate distances between all doors
        print(f"\n📏 Testing Door Distance Calculation...")
        
        goal = f"""Load the building data from '{building_data_path}' and calculate the distance in meters between all doors in the building. For each door, show its ID, location, and the distances to all other doors. Present the results in a clear format showing which doors are closest to each other."""
        
        print(f"Goal: {goal}")
        print(f"\n⏳ Processing...")
        
        # Process the goal
        result = agent.process_goal(goal)
        
        print(f"\n📊 Results:")
        print(f"=" * 40)
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message', 'No message provided')}")
        
        # Show execution details
        if result.get('reasoning_result'):
            reasoning = result['reasoning_result']
            
            if reasoning.get('summary'):
                summary = reasoning['summary']
                print(f"\n📈 Execution Summary:")
                print(f"  Completed Tasks: {summary.get('completed_tasks', 0)}")
                print(f"  Total Tasks: {summary.get('total_tasks', 0)}")
                print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
                print(f"  Execution Time: {summary.get('total_execution_time', 0):.2f}s")
            
            # Show completed tasks
            if reasoning.get('tasks_completed'):
                print(f"\n✅ Completed Tasks:")
                for task in reasoning['tasks_completed']:
                    print(f"  - {task}")
            
            # Show failed tasks
            if reasoning.get('tasks_failed'):
                print(f"\n❌ Failed Tasks:")
                for task in reasoning['tasks_failed']:
                    print(f"  - {task}")
        
        print(f"\n🔍 LangSmith Traces:")
        print(f"  Project: {config.logging.langsmith_project}")
        print(f"  URL: https://smith.langchain.com/")
        
        success_rate = 0
        if result.get('reasoning_result', {}).get('summary'):
            completed = result['reasoning_result']['summary'].get('completed_tasks', 0)
            total = result['reasoning_result']['summary'].get('total_tasks', 1)
            success_rate = completed / total if total > 0 else 0
        
        if success_rate > 0.5:
            print(f"\n✅ Door distance analysis successful!")
        elif success_rate > 0:
            print(f"\n⚠️  Analysis partially successful.")
        else:
            print(f"\n❌ Analysis failed.")
        
        return success_rate > 0.5
        
    except Exception as e:
        print(f"\n💥 Error during analysis: {e}")
        return False


if __name__ == "__main__":
    test_door_distances()