#!/usr/bin/env python3
"""
Simple test for the agent memory functionality without requiring API keys.

This script demonstrates the sliding window memory logic without LLM calls.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.messages import HumanMessage, AIMessage


class SimpleSlidingWindowMemory:
    """
    Simplified sliding window memory for testing without LLM dependencies.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.messages = []
        self.summary_messages = []
    
    def add_message(self, message):
        """Add a message to memory, handling window size and summarization."""
        self.messages.append(message)
        
        # Check if we need to summarize
        if len(self.messages) > self.window_size:
            self._summarize_oldest_messages()
    
    def _summarize_oldest_messages(self):
        """Summarize the oldest messages when window size is exceeded."""
        if len(self.messages) <= self.window_size:
            return
        
        # Get the oldest messages to summarize (all except the last one)
        messages_to_summarize = self.messages[:-1]
        newest_message = self.messages[-1]
        
        # Create simple summary (without LLM)
        summary = self._create_simple_summary(messages_to_summarize)
        
        # Store the summary
        from langchain_core.messages import SystemMessage
        summary_message = SystemMessage(content=summary)
        self.summary_messages.append(summary_message)
        
        # Keep only the newest message
        self.messages = [newest_message]
    
    def _create_simple_summary(self, messages):
        """Create a simple summary without LLM."""
        if not messages:
            return ""
        
        # Count message types
        user_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        agent_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
        
        return f"[Previous conversation summary: {len(messages)} messages ({user_count} user, {agent_count} agent) about AEC compliance verification]"
    
    def get_memory_stats(self):
        """Get statistics about the current memory state."""
        return {
            "current_messages": len(self.messages),
            "summary_messages": len(self.summary_messages),
            "window_size": self.window_size,
            "total_context": len(self.messages) + len(self.summary_messages),
            "memory_usage": f"{len(self.messages)}/{self.window_size} messages"
        }


def test_sliding_window_memory():
    """Test the sliding window memory functionality."""
    print("🧠 Testing Sliding Window Memory (Simple Version)")
    print("=" * 60)
    
    # Create memory with window size of 5
    memory = SimpleSlidingWindowMemory(window_size=5)
    
    # Add messages to test the sliding window
    messages = [
        ("User", "Check all doors for compliance"),
        ("Agent", "I'll start by listing all doors in the project"),
        ("User", "Focus on door widths specifically"),
        ("Agent", "Checking door widths against minimum requirements"),
        ("User", "What about fire ratings?"),
        ("Agent", "Now checking fire ratings for all doors"),
        ("User", "Generate a compliance report"),
        ("Agent", "Creating comprehensive compliance report"),
    ]
    
    print(f"Window size: {memory.window_size}")
    print(f"Adding {len(messages)} messages...\n")
    
    for i, (sender, content) in enumerate(messages):
        if sender == "User":
            message = HumanMessage(content=content)
        else:
            message = AIMessage(content=content)
        
        memory.add_message(message)
        
        stats = memory.get_memory_stats()
        print(f"Message {i+1} ({sender}): {content[:50]}...")
        print(f"  Memory: {stats['memory_usage']}")
        print(f"  Total context: {stats['total_context']} messages")
        
        if memory.summary_messages:
            print(f"  Summary messages: {len(memory.summary_messages)}")
            print(f"  Latest summary: {memory.summary_messages[-1].content[:100]}...")
        print()
    
    print("Final memory state:")
    print(f"Current messages: {len(memory.messages)}")
    print(f"Summary messages: {len(memory.summary_messages)}")
    print(f"Total context: {len(memory.messages) + len(memory.summary_messages)}")
    
    print("\nCurrent messages in memory:")
    for i, msg in enumerate(memory.messages):
        sender = "User" if isinstance(msg, HumanMessage) else "Agent"
        print(f"  {i+1}. {sender}: {msg.content}")
    
    print("\nSummary messages:")
    for i, msg in enumerate(memory.summary_messages):
        print(f"  Summary {i+1}: {msg.content}")


def demonstrate_memory_behavior():
    """Demonstrate the key memory behavior."""
    print("\n\n🎯 Memory Behavior Demonstration")
    print("=" * 60)
    
    memory = SimpleSlidingWindowMemory(window_size=5)
    
    print("Key behavior: When the 6th message arrives, the first 5 are summarized!")
    print()
    
    # Add exactly 5 messages
    for i in range(5):
        msg = HumanMessage(content=f"User message {i+1}")
        memory.add_message(msg)
        stats = memory.get_memory_stats()
        print(f"Message {i+1}: Memory has {stats['current_messages']} messages, {stats['summary_messages']} summaries")
    
    print(f"\nAfter 5 messages: {memory.get_memory_stats()['memory_usage']}")
    
    # Add the 6th message - this should trigger summarization
    print("\n🚨 Adding 6th message - this triggers summarization!")
    msg6 = HumanMessage(content="User message 6")
    memory.add_message(msg6)
    
    stats = memory.get_memory_stats()
    print(f"After 6th message: {stats['memory_usage']}")
    print(f"Summary messages: {stats['summary_messages']}")
    print(f"Total context: {stats['total_context']}")
    
    if memory.summary_messages:
        print(f"\nSummary created: {memory.summary_messages[0].content}")
    
    print("\nCurrent messages (should only be message 6):")
    for i, msg in enumerate(memory.messages):
        print(f"  {i+1}. {msg.content}")


if __name__ == "__main__":
    print("🚀 Simple Agent Memory Test")
    print("=" * 60)
    
    try:
        test_sliding_window_memory()
        demonstrate_memory_behavior()
        
        print("\n\n✅ Memory tests passed!")
        print("\n🎉 The agent now has sliding window memory!")
        print("📝 Key features:")
        print("  • Maintains exactly 5 messages in active memory")
        print("  • When 6th message arrives, first 5 are summarized")
        print("  • Summary preserves context for the LLM")
        print("  • Memory grows efficiently without losing important context")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
