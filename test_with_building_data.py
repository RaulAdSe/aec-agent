#!/usr/bin/env python3
"""
Test Agent with Real Building Data - Complete end-to-end test with actual building data.

This script tests the agent with the actual building data file to demonstrate
the complete reasoning process with LangSmith tracing.
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


def test_agent_with_building_data():
    """Test the agent with actual building data for fire safety analysis."""
    
    print("🏢 AEC Agent Test with Real Building Data")
    print("=" * 60)
    
    # Check for building data file
    building_data_path = "data/out/FM-ARC_v2.json"
    full_path = Path(building_data_path)
    
    if not full_path.exists():
        print(f"❌ Building data file not found: {building_data_path}")
        print("Please ensure the building data file exists at the specified path.")
        return False
    
    print(f"✅ Building data found: {building_data_path}")
    print(f"   File size: {full_path.stat().st_size / 1024:.1f} KB")
    
    # Create development configuration
    config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
    config.logging.enable_langsmith = True
    config.logging.langsmith_project = "AEC-Agent-Building-Analysis"
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Max Iterations: {config.reasoning.max_iterations}")
    print(f"  LangSmith Project: {config.logging.langsmith_project}")
    
    try:
        print(f"\n🤖 Creating AEC Reasoning Agent...")
        agent = ReasoningAgent(config=config)
        print("✅ Agent created successfully!")
        
        # Test with a specific fire safety query that includes the file path
        print(f"\n🔥 Testing Fire Safety Compliance Analysis...")
        
        goal = f"""Load the building data from '{building_data_path}' and analyze the building elements for fire safety compliance in this residential building. Identify fire-rated doors, walls, and other safety elements, then document your findings with specific recommendations."""
        
        print(f"Goal: {goal}")
        print(f"\n⏳ Processing (this may take 30-60 seconds)...")
        
        # Process the goal with the building data
        result = agent.process_goal(goal)
        
        print(f"\n📊 Analysis Results:")
        print(f"="*50)
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message', 'No message provided')}")
        
        # Show detailed reasoning results
        if result.get('reasoning_result'):
            reasoning = result['reasoning_result']
            
            if reasoning.get('summary'):
                summary = reasoning['summary']
                print(f"\n📈 Execution Summary:")
                print(f"  Completed Tasks: {summary.get('completed_tasks', 0)}")
                print(f"  Total Tasks: {summary.get('total_tasks', 0)}")
                print(f"  Failed Tasks: {summary.get('failed_tasks', 0)}")
                print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
                print(f"  Execution Time: {summary.get('total_execution_time', 0):.2f}s")
                print(f"  Iterations: {summary.get('iteration', 0)}")
            
            # Show task details if available
            if reasoning.get('tasks_completed'):
                print(f"\n✅ Completed Tasks:")
                for task in reasoning['tasks_completed']:
                    print(f"  - {task}")
            
            if reasoning.get('tasks_failed'):
                print(f"\n❌ Failed Tasks:")
                for task in reasoning['tasks_failed']:
                    print(f"  - {task}")
            
            # Show guardrail status
            if reasoning.get('summary', {}).get('guardrails'):
                guardrails = reasoning['summary']['guardrails']
                print(f"\n🛡️ Guardrail Status:")
                print(f"  Replanning Events: {guardrails.get('replanning_events', 'N/A')}")
                print(f"  Total Steps: {guardrails.get('total_steps', 'N/A')}")
                if guardrails.get('task_attempts'):
                    print(f"  Task Attempts: {guardrails['task_attempts']}")
        
        # Show session summary if available
        session_summary = result.get('session_summary')
        if session_summary:
            print(f"\n🧠 Session Memory:")
            print(f"  Goal: {session_summary.get('goal', 'N/A')}")
            print(f"  Active Files: {session_summary.get('active_files', [])}")
            if session_summary.get('recent_tools'):
                print(f"  Recent Tools: {session_summary.get('recent_tools', [])}")
        
        print(f"\n🔍 LangSmith Traces:")
        print(f"  Project: {config.logging.langsmith_project}")
        print(f"  URL: https://smith.langchain.com/")
        print(f"  Look for traces showing:")
        print(f"    - Goal decomposition into specific tasks")
        print(f"    - Tool planning for each task")
        print(f"    - Building data loading and analysis")
        print(f"    - Fire safety element identification")
        print(f"    - Documentation generation")
        
        print(f"\n🎉 Test completed!")
        
        success_rate = 0
        if result.get('reasoning_result', {}).get('summary'):
            completed = result['reasoning_result']['summary'].get('completed_tasks', 0)
            total = result['reasoning_result']['summary'].get('total_tasks', 1)
            success_rate = completed / total if total > 0 else 0
        
        if success_rate > 0.5:
            print(f"✅ Analysis successful! Check LangSmith for detailed traces.")
        elif success_rate > 0:
            print(f"⚠️  Analysis partially successful. Some tasks completed.")
        else:
            print(f"❌ Analysis failed. Check logs and LangSmith for details.")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Error during analysis: {e}")
        print("\nCheck the error details above and ensure:")
        print("1. Building data file format is correct")
        print("2. API keys are properly set")
        print("3. Internet connection is available")
        return False


def quick_building_data_check():
    """Quick check of the building data file."""
    
    print("🔍 Building Data Quick Check")
    print("-" * 30)
    
    building_data_path = Path("data/out/FM-ARC_v2.json")
    
    if not building_data_path.exists():
        print(f"❌ File not found: {building_data_path}")
        return
    
    try:
        import json
        
        with open(building_data_path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ File loaded successfully")
        print(f"   File size: {building_data_path.stat().st_size / 1024:.1f} KB")
        
        # Check basic structure
        if isinstance(data, dict):
            print(f"   Structure: Dictionary with {len(data)} top-level keys")
            print(f"   Keys: {list(data.keys())[:5]}{'...' if len(data) > 5 else ''}")
            
            # Look for common building elements
            element_counts = {}
            for key, value in data.items():
                if isinstance(value, list):
                    element_counts[key] = len(value)
            
            if element_counts:
                print(f"   Element counts:")
                for element_type, count in sorted(element_counts.items()):
                    print(f"     {element_type}: {count}")
        
        print(f"✅ Building data appears valid for analysis")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


def main():
    """Main test function."""
    
    # Quick data check first
    quick_building_data_check()
    
    print("\n" + "="*60)
    
    # Run the full test
    success = test_agent_with_building_data()
    
    print("\n" + "="*60)
    if success:
        print("🎯 Test completed! Check LangSmith dashboard for complete traces.")
        print("\nWhat to look for in LangSmith:")
        print("1. Goal decomposition showing task breakdown")
        print("2. Tool planning for each specific task")
        print("3. Building data loading and processing")
        print("4. Element analysis and compliance checking")
        print("5. Documentation generation")
        print("6. External guardrail monitoring throughout")
    else:
        print("💥 Test failed. Please check the error messages above.")


if __name__ == "__main__":
    main()