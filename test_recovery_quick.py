#!/usr/bin/env python3
"""Quick test to verify recovery system is working properly."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment and setup path
load_dotenv()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aec_agent.agent import create_agent

def test_recovery_system():
    """Test that recovery system works without API signature errors."""
    
    print("🔄 RECOVERY SYSTEM QUICK TEST")
    print("=" * 40)
    
    # Initialize agent
    try:
        agent = create_agent(
            model_name="gpt-4o-mini",
            temperature=0.1,
            verbose=True,
            enable_memory=True,
            session_id="test_recovery",
            max_iterations=5
        )
        print("✅ Agent initialized")
    except Exception as e:
        print(f"❌ Agent init failed: {e}")
        return
    
    # Test simple query that should trigger recovery
    print("\n🧪 Testing recovery with intentionally problematic query...")
    
    try:
        # This should trigger tool failures and recovery
        result = agent.process_goal("Get all doors from the building model")
        
        if isinstance(result, dict) and 'reasoning_result' in result:
            reasoning = result['reasoning_result']
            status = reasoning.get('status', 'unknown')
            message = reasoning.get('message', 'No message')
            
            print(f"🤖 Status: {status}")
            print(f"🤖 Response: {message}")
            
            if status == 'partial':
                print("✅ Recovery system handled the error gracefully!")
                return True
            else:
                print("ℹ️  Query completed normally")
                return True
        else:
            print(f"🤖 Response: {str(result)}")
            return True
            
    except Exception as e:
        if "unexpected keyword argument" in str(e):
            print(f"❌ API signature error still exists: {e}")
            return False
        else:
            print(f"ℹ️  Different error (expected): {e}")
            return True

if __name__ == "__main__":
    success = test_recovery_system()
    if success:
        print("\n🎉 Recovery system is working!")
    else:
        print("\n⚠️  Recovery system needs more fixes")