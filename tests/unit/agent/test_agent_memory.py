#!/usr/bin/env python3
"""
Test script for the agent memory functionality.

This script demonstrates how the sliding window memory works with the ReAct agent.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.memory import SlidingWindowMemory, MemoryManager
from langchain_core.messages import HumanMessage, AIMessage


def test_sliding_window_memory():
    """Test the sliding window memory functionality."""
    print("🧠 Testing Sliding Window Memory")
    print("=" * 50)
    
    # Create memory with window size of 5
    memory = SlidingWindowMemory(window_size=5)
    
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
        print(f"  Summary {i+1}: {msg.content[:150]}...")


def test_memory_manager():
    """Test the high-level memory manager."""
    print("\n\n🎯 Testing Memory Manager")
    print("=" * 50)
    
    manager = MemoryManager(window_size=5)
    manager.start_session("test_session_001")
    
    # Simulate a conversation
    conversation = [
        ("user", "List all doors in the project"),
        ("agent", "I found 3 doors: D001 (90cm), D002 (80cm), D003 (75cm)"),
        ("user", "Check if they meet minimum width requirements"),
        ("agent", "D001 and D002 are compliant (≥80cm), D003 is non-compliant (75cm)"),
        ("user", "What about fire ratings?"),
        ("agent", "Checking fire ratings: D001 (RF-60), D002 (no rating), D003 (RF-30)"),
        ("user", "Generate a compliance report"),
        ("agent", "Creating report: 2/3 doors compliant for width, fire ratings vary"),
    ]
    
    for sender, content in conversation:
        if sender == "user":
            manager.add_user_message(content)
        else:
            manager.add_agent_message(content)
        
        info = manager.get_memory_info()
        print(f"{sender.capitalize()}: {content}")
        print(f"  Memory: {info['memory_usage']}, Context: {info['total_context']}")
        print()
    
    print("Final memory info:")
    final_info = manager.get_memory_info()
    for key, value in final_info.items():
        print(f"  {key}: {value}")


def test_memory_export_import():
    """Test memory export and import functionality."""
    print("\n\n💾 Testing Memory Export/Import")
    print("=" * 50)
    
    # Create memory and add some messages
    memory1 = SlidingWindowMemory(window_size=3)
    
    test_messages = [
        HumanMessage(content="First user message"),
        AIMessage(content="First agent response"),
        HumanMessage(content="Second user message"),
        AIMessage(content="Second agent response"),
        HumanMessage(content="Third user message"),
    ]
    
    for msg in test_messages:
        memory1.add_message(msg)
    
    print("Original memory:")
    print(f"  Messages: {len(memory1.messages)}")
    print(f"  Summaries: {len(memory1.summary_messages)}")
    
    # Export memory
    exported = memory1.export_memory()
    print(f"\nExported memory data: {len(exported['messages'])} messages, {len(exported['summary_messages'])} summaries")
    
    # Create new memory and import
    memory2 = SlidingWindowMemory(window_size=3)
    memory2.import_memory(exported)
    
    print("\nImported memory:")
    print(f"  Messages: {len(memory2.messages)}")
    print(f"  Summaries: {len(memory2.summary_messages)}")
    
    # Verify they match
    assert len(memory1.messages) == len(memory2.messages)
    assert len(memory1.summary_messages) == len(memory2.summary_messages)
    print("✅ Export/Import test passed!")


if __name__ == "__main__":
    print("🚀 Agent Memory Test Suite")
    print("=" * 60)
    
    try:
        test_sliding_window_memory()
        test_memory_manager()
        test_memory_export_import()
        
        print("\n\n✅ All memory tests passed!")
        print("\nThe agent now has sliding window memory with automatic summarization!")
        print("When the 6th message arrives, the first 5 are summarized into a single summary message.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
