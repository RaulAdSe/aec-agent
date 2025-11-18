#!/usr/bin/env python3
"""Test to verify recovery system messages are captured for Streamlit display."""

import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment and setup path
load_dotenv()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aec_agent.agent import create_agent
from app import StreamlitLogHandler

def test_recovery_messages():
    """Test that recovery system messages are properly captured and formatted."""
    
    print("🔄 TESTING RECOVERY MESSAGES FOR STREAMLIT")
    print("=" * 50)
    
    # Set up the Streamlit log handler
    log_handler = StreamlitLogHandler()
    
    # Get root logger and add our handler
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.INFO)
    
    # Initialize agent
    try:
        agent = create_agent(
            model_name="gpt-4o-mini",
            temperature=0.1,
            verbose=True,
            enable_memory=True,
            session_id="test_messages",
            max_iterations=5
        )
        print("✅ Agent initialized with recovery system")
    except Exception as e:
        print(f"❌ Agent init failed: {e}")
        return
    
    # Clear any existing messages
    log_handler.clear()
    
    # Test query that should trigger recovery
    print("\n🧪 Running query that triggers recovery...")
    
    try:
        result = agent.process_goal("Get all doors from the building that doesn't exist")
        
        # Get captured messages
        messages = log_handler.get_messages()
        
        print(f"\n📋 Captured {len(messages)} user-friendly messages:")
        print("-" * 50)
        
        for i, msg in enumerate(messages, 1):
            level_icon = "❌" if msg["level"] == "ERROR" else "⚠️" if msg["level"] == "WARNING" else "ℹ️"
            print(f"{i:2d}. [{msg['component']:10}] {level_icon} {msg['message']}")
        
        # Check for recovery-specific messages
        recovery_messages = [m for m in messages if any(keyword in m['message'].lower() for keyword in [
            'recovery', 'failure', 'attempting', 'analyzing', 'replanning', 'skipped'
        ])]
        
        print(f"\n🔄 Recovery-related messages: {len(recovery_messages)}")
        for msg in recovery_messages:
            print(f"   🔄 {msg['message']}")
        
        if recovery_messages:
            print("\n✅ Recovery system messages are being captured for Streamlit!")
            return True
        else:
            print("\n⚠️  No recovery messages captured - may need more triggers")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_recovery_messages()
    
    if success:
        print("\n🎉 Recovery messages will be visible to users in Streamlit!")
        print("\n📱 In the Streamlit interface, users will see:")
        print("   • Real-time status updates during recovery")
        print("   • Expandable 'View progress' showing recovery steps")
        print("   • User-friendly icons and descriptions")
        print("   • Clear indication when recovery succeeds or fails")
    else:
        print("\n💡 Recovery messages may not be fully visible yet")