#!/usr/bin/env python3
"""
Test Just-In-Time Planning - Simple test to see if the new planning works.
"""

import os
import sys
from pathlib import Path

# Load environment variables
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"\'')
                os.environ[key] = value

sys.path.insert(0, str(Path(__file__).parent))

from aec_agent import ReasoningAgent, AgentConfig, AgentProfile

def test_jit_planning():
    """Test just-in-time planning with a simple goal."""
    
    print("🧪 Testing Just-In-Time Planning")
    print("=" * 60)
    
    # Create agent with development config for faster testing
    config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
    config.logging.enable_langsmith = False  # Disable for cleaner output
    
    agent = ReasoningAgent(config=config)
    
    # Test distance calculation - the critical case!  
    goal = "Load building data from 'data/out/FM-ARC_v2.json', get all door elements, and calculate distances between doors"
    
    print(f"Goal: {goal}")
    print("\nExecuting with just-in-time planning...")
    
    try:
        result = agent.process_goal(goal)
        print(f"\n✅ Result: {result['status']}")
        print(f"Summary: {result.get('summary', {})}")
        
        # Check if tool selection improved
        if 'tasks' in result:
            print(f"\nTask Details:")
            for i, task in enumerate(result['tasks']):
                print(f"  {i+1}. {task['name']} -> {task['tools']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_jit_planning()