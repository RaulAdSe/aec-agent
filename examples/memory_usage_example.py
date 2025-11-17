"""
Example demonstrating the new memory system for the AEC compliance agent.

This example shows how to:
1. Initialize an agent with memory enabled
2. Set session goals and track subtasks
3. Use the agent with persistent conversation memory
4. Access memory information and statistics
"""

import os
from pathlib import Path
from aec_agent.agent import create_agent
from aec_agent.memory import MemoryManagerConfig, TaskStatus

def main():
    """Demonstrate memory system functionality."""
    
    print("🧠 AEC Compliance Agent Memory System Demo")
    print("=" * 50)
    
    # Create a temporary directory for session persistence
    session_dir = Path("./data/demo_sessions")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure memory with persistence
    memory_config = MemoryManagerConfig(
        session_persistence_path=session_dir,
        enable_persistence=True,
        auto_save_interval=3  # Save every 3 operations
    )
    
    # Create agent with memory enabled
    print("Creating agent with memory enabled...")
    agent = create_agent(
        enable_memory=True,
        memory_config=memory_config,
        verbose=False  # Reduce noise for demo
    )
    
    session_id = agent.memory_manager.get_session_id()
    print(f"✓ Agent initialized with session ID: {session_id[:8]}...")
    
    # Set a session goal
    goal = "Analyze building compliance for fire safety regulations"
    print(f"\n📋 Setting session goal: {goal}")
    agent.set_session_goal(goal, "Focus on exit routes, fire doors, and emergency systems")
    
    # Add some subtasks
    print("\n📝 Adding subtasks:")
    task1 = agent.add_subtask("Load building IFC data")
    task2 = agent.add_subtask("Identify all fire doors", [task1])  # Depends on task1
    task3 = agent.add_subtask("Check exit route compliance")
    task4 = agent.add_subtask("Generate compliance report")
    
    print(f"   ✓ Task 1: Load building IFC data ({task1[:8]}...)")
    print(f"   ✓ Task 2: Identify all fire doors ({task2[:8]}...)")
    print(f"   ✓ Task 3: Check exit route compliance ({task3[:8]}...)")
    print(f"   ✓ Task 4: Generate compliance report ({task4[:8]}...)")
    
    # Simulate working on tasks
    print(f"\n🔄 Working on tasks...")
    agent.update_subtask_status(task1, TaskStatus.IN_PROGRESS, "Starting data load")
    print("   ✓ Task 1 set to IN_PROGRESS")
    
    # Simulate tool usage (this would normally happen during agent.process())
    agent.memory_manager.record_tool_execution(
        tool_name="load_building_data",
        arguments={"path": "/demo/building.ifc"},
        success=True,
        result_summary="Successfully loaded building with 1,250 elements"
    )
    
    agent.update_subtask_status(task1, TaskStatus.COMPLETED, "Building data loaded successfully")
    print("   ✓ Task 1 completed")
    
    # Track active file
    agent.memory_manager.track_active_file("/demo/building.ifc")
    agent.memory_manager.set_building_data_context("/demo/building.ifc", {
        "project_name": "Demo Office Building",
        "total_elements": 1250,
        "building_type": "Commercial"
    })
    
    # Simulate conversation turns
    print(f"\n💬 Simulating conversation with memory...")
    agent.memory_manager.add_conversation_turn(
        "What building data do we have loaded?",
        "We have loaded the Demo Office Building with 1,250 elements from /demo/building.ifc"
    )
    
    agent.memory_manager.add_conversation_turn(
        "What's our current goal?",
        "Our goal is to analyze building compliance for fire safety regulations, focusing on exit routes, fire doors, and emergency systems."
    )
    
    # Get memory summary
    print(f"\n📊 Memory Summary:")
    summary = agent.get_memory_summary()
    
    print(f"   Session ID: {summary['session_id'][:8]}...")
    print(f"   Goal: {summary['goal']}")
    print(f"   Active Files: {summary['active_files_count']}")
    print(f"   Total Subtasks: {summary['total_subtasks']}")
    print(f"   Pending Subtasks: {summary['pending_subtasks']}")
    print(f"   Tool Executions: {summary['tool_executions']}")
    print(f"   Building Data Loaded: {summary['building_data_loaded']}")
    
    # Show current context that would be sent to LLM
    print(f"\n🎯 Current Context for LLM:")
    print("-" * 40)
    context = agent.memory_manager.get_full_context_for_prompt()
    print(context[:500] + "..." if len(context) > 500 else context)
    
    # Save session manually
    print(f"\n💾 Saving session...")
    agent.save_session()
    print(f"   ✓ Session saved to {session_dir}")
    
    # Demonstrate loading existing session
    print(f"\n🔄 Demonstrating session persistence...")
    agent2 = create_agent(
        enable_memory=True,
        memory_config=memory_config,
        session_id=session_id,  # Load existing session
        verbose=False
    )
    
    # Verify session loaded correctly
    loaded_summary = agent2.get_memory_summary()
    print(f"   ✓ Reloaded session with {loaded_summary['total_subtasks']} subtasks")
    print(f"   ✓ Goal persisted: {loaded_summary['goal'][:50]}...")
    
    print(f"\n🎉 Memory system demo completed successfully!")
    print(f"Session files saved in: {session_dir}")


if __name__ == "__main__":
    main()