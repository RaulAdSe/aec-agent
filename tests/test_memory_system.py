"""
Comprehensive tests for the memory system implementation.

Tests cover short-term memory, session memory, memory manager,
and integration with the compliance agent.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from aec_agent.memory import (
    MemoryManager,
    MemoryManagerConfig, 
    ShortTermMemory,
    ShortTermMemoryConfig,
    SessionMemory,
    TaskStatus,
    SubTask,
    ToolExecution
)
from aec_agent.agent import ComplianceAgent, create_agent


class TestShortTermMemory:
    """Test short-term memory functionality."""
    
    def test_short_term_memory_initialization(self):
        """Test basic initialization of short-term memory."""
        config = ShortTermMemoryConfig(window_size=5, max_token_limit=1000)
        memory = ShortTermMemory(config)
        
        assert memory.config.window_size == 5
        assert memory.config.max_token_limit == 1000
        assert memory.buffer_memory is not None
        assert memory.summary_memory is not None
    
    def test_conversation_turn_storage(self):
        """Test storing conversation turns."""
        memory = ShortTermMemory()
        
        # Add conversation turns
        memory.add_conversation_turn("Hello", "Hi there!")
        memory.add_conversation_turn("How are you?", "I'm doing well, thank you!")
        
        # Get memory variables
        variables = memory.get_memory_variables()
        assert "recent_conversation" in variables
        assert "conversation_summary" in variables
    
    def test_conversation_context_formatting(self):
        """Test conversation context formatting."""
        memory = ShortTermMemory()
        
        # Add some conversation
        memory.add_conversation_turn("Test question", "Test answer")
        
        context = memory.get_conversation_context()
        assert isinstance(context, str)
        # Should contain formatted conversation
        assert len(context) > 0
    
    def test_memory_clearing(self):
        """Test memory clearing functionality."""
        memory = ShortTermMemory()
        
        # Add conversation
        memory.add_conversation_turn("Test", "Response")
        
        # Clear memory
        memory.clear_memory()
        
        # Check that memory is cleared
        variables = memory.get_memory_variables()
        recent = variables.get("recent_conversation", [])
        assert len(recent) == 0
    
    def test_memory_stats(self):
        """Test memory statistics."""
        memory = ShortTermMemory()
        
        # Add conversation
        memory.add_conversation_turn("Question", "Answer")
        
        stats = memory.get_memory_stats()
        assert "recent_messages_count" in stats
        assert "window_size" in stats
        assert "has_summary" in stats
        assert stats["recent_messages_count"] > 0


class TestSessionMemory:
    """Test session memory functionality."""
    
    def test_session_initialization(self):
        """Test session memory initialization."""
        memory = SessionMemory()
        
        assert memory.state.session_id is not None
        assert memory.state.goal is None
        assert len(memory.state.active_files) == 0
        assert len(memory.state.subtasks) == 0
    
    def test_goal_management(self):
        """Test goal setting and retrieval."""
        memory = SessionMemory()
        
        goal = "Analyze building compliance"
        context = "Focus on fire safety regulations"
        
        memory.set_goal(goal, context)
        
        assert memory.get_goal() == goal
        assert memory.state.context == context
    
    def test_file_tracking(self):
        """Test file tracking functionality."""
        memory = SessionMemory()
        
        file_path = "/path/to/building.ifc"
        
        # Add active file
        memory.add_active_file(file_path)
        assert file_path in memory.state.active_files
        
        # Mark file as modified
        memory.mark_file_modified(file_path)
        assert file_path in memory.state.modified_files
        
        # Remove active file
        memory.remove_active_file(file_path)
        assert file_path not in memory.state.active_files
    
    def test_subtask_management(self):
        """Test subtask creation and management."""
        memory = SessionMemory()
        
        # Add subtask
        task_id = memory.add_subtask("Load building data")
        assert task_id is not None
        assert len(memory.state.subtasks) == 1
        
        # Update subtask status
        success = memory.update_subtask_status(task_id, TaskStatus.COMPLETED, "Task finished")
        assert success
        
        # Get completed subtask
        subtask = next(t for t in memory.state.subtasks if t.id == task_id)
        assert subtask.status == TaskStatus.COMPLETED
        assert subtask.notes == "Task finished"
    
    def test_tool_execution_tracking(self):
        """Test tool execution recording."""
        memory = SessionMemory()
        
        # Record tool execution
        execution_id = memory.record_tool_execution(
            tool_name="load_building_data",
            arguments={"path": "/test/path.ifc"},
            success=True,
            result_summary="Building data loaded successfully"
        )
        
        assert execution_id is not None
        assert len(memory.state.tool_history) == 1
        
        # Get recent executions
        recent = memory.get_recent_tool_executions(5)
        assert len(recent) == 1
        assert recent[0].tool_name == "load_building_data"
    
    def test_building_data_context(self):
        """Test building data context management."""
        memory = SessionMemory()
        
        data_path = "/path/to/building.json"
        context = {"project_name": "Test Building", "total_elements": 100}
        
        memory.set_building_data(data_path, context)
        
        assert memory.state.loaded_building_data == data_path
        assert memory.state.building_context["project_name"] == "Test Building"
    
    def test_session_summary(self):
        """Test session summary generation."""
        memory = SessionMemory()
        
        # Add some data
        memory.set_goal("Test goal")
        memory.add_subtask("Test task")
        memory.add_active_file("/test/file.ifc")
        
        summary = memory.get_session_summary()
        
        assert "session_id" in summary
        assert "goal" in summary
        assert "total_subtasks" in summary
        assert summary["goal"] == "Test goal"
        assert summary["total_subtasks"] == 1
    
    def test_context_for_prompt(self):
        """Test context generation for LLM prompts."""
        memory = SessionMemory()
        
        # Set up session state
        memory.set_goal("Analyze compliance")
        memory.add_active_file("/building.ifc")
        task_id = memory.add_subtask("Load data")
        memory.set_current_subtask(task_id)
        
        context = memory.get_context_for_prompt()
        
        assert isinstance(context, str)
        assert "Session Goal: Analyze compliance" in context
        assert "Load data" in context


class TestMemoryManager:
    """Test memory manager coordination functionality."""
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryManagerConfig(
                session_persistence_path=Path(temp_dir),
                enable_persistence=True
            )
            
            manager = MemoryManager(config)
            
            assert manager.short_term is not None
            assert manager.session is not None
            assert manager.config.enable_persistence
    
    def test_conversation_management(self):
        """Test conversation management through memory manager."""
        manager = MemoryManager()
        
        # Add conversation turn
        manager.add_conversation_turn("Hello", "Hi there!")
        
        # Get conversation context
        context = manager.get_conversation_context()
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_session_goal_management(self):
        """Test session goal management."""
        manager = MemoryManager()
        
        goal = "Analyze building compliance"
        manager.set_session_goal(goal)
        
        # Check that goal is set in session memory
        assert manager.session.get_goal() == goal
    
    def test_subtask_management(self):
        """Test subtask management through memory manager."""
        manager = MemoryManager()
        
        # Add subtask
        task_id = manager.add_subtask("Load building data")
        assert task_id is not None
        
        # Update status
        success = manager.update_subtask_status(task_id, TaskStatus.IN_PROGRESS)
        assert success
        
        # Set as current task
        success = manager.set_current_subtask(task_id)
        assert success
    
    def test_file_tracking(self):
        """Test file tracking through memory manager."""
        manager = MemoryManager()
        
        file_path = "/test/building.ifc"
        
        # Track file
        manager.track_active_file(file_path)
        assert file_path in manager.get_active_files()
        
        # Mark as modified
        manager.mark_file_modified(file_path)
        assert file_path in manager.get_modified_files()
    
    def test_tool_execution_recording(self):
        """Test tool execution recording."""
        manager = MemoryManager()
        
        # Record tool execution
        execution_id = manager.record_tool_execution(
            tool_name="query_elements",
            arguments={"element_type": "walls"},
            success=True,
            result_summary="Found 50 wall elements"
        )
        
        assert execution_id is not None
        
        # Get recent executions
        recent = manager.get_recent_tool_executions(5)
        assert len(recent) == 1
    
    def test_full_context_generation(self):
        """Test full context generation for LLM prompts."""
        manager = MemoryManager()
        
        # Set up some context
        manager.set_session_goal("Test goal")
        manager.add_conversation_turn("Question", "Answer")
        task_id = manager.add_subtask("Test task")
        manager.set_current_subtask(task_id)
        
        # Get full context
        context = manager.get_full_context_for_prompt()
        
        assert isinstance(context, str)
        assert "Session Context" in context
        assert "Conversation History" in context
        assert "Test goal" in context
    
    def test_memory_variables_for_langchain(self):
        """Test LangChain-compatible memory variables."""
        manager = MemoryManager()
        
        # Set up some data
        manager.set_session_goal("Test goal")
        manager.add_conversation_turn("Question", "Answer")
        
        variables = manager.get_memory_variables_for_langchain()
        
        assert isinstance(variables, dict)
        assert "session_goal" in variables
        assert "recent_conversation" in variables
        assert "full_memory_context" in variables
        assert variables["session_goal"] == "Test goal"
    
    def test_session_persistence(self):
        """Test session persistence functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryManagerConfig(
                session_persistence_path=Path(temp_dir),
                enable_persistence=True
            )
            
            # Create manager and add data
            manager = MemoryManager(config)
            manager.set_session_goal("Persistent goal")
            manager.add_subtask("Persistent task")
            
            session_id = manager.get_session_id()
            
            # Save session
            manager.save_session()
            
            # Create new manager with same session ID
            manager2 = MemoryManager(config, session_id=session_id)
            
            # Check that data persisted
            assert manager2.session.get_goal() == "Persistent goal"
            assert len(manager2.session.state.subtasks) == 1


class TestAgentIntegration:
    """Test memory integration with the compliance agent."""
    
    @patch('aec_agent.agent.ChatOpenAI')
    def test_agent_with_memory_enabled(self, mock_llm):
        """Test agent initialization with memory enabled."""
        mock_llm.return_value = Mock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryManagerConfig(
                session_persistence_path=Path(temp_dir),
                enable_persistence=True
            )
            
            agent = create_agent(
                enable_memory=True,
                memory_config=config,
                verbose=False
            )
            
            assert agent.enable_memory
            assert agent.memory_manager is not None
            assert agent.memory_manager.get_session_id() is not None
    
    @patch('aec_agent.agent.ChatOpenAI')
    def test_agent_with_memory_disabled(self, mock_llm):
        """Test agent initialization with memory disabled."""
        mock_llm.return_value = Mock()
        
        agent = create_agent(enable_memory=False, verbose=False)
        
        assert not agent.enable_memory
        assert agent.memory_manager is None
    
    @patch('aec_agent.agent.ChatOpenAI')
    def test_memory_methods_on_agent(self, mock_llm):
        """Test memory-related methods on the agent."""
        mock_llm.return_value = Mock()
        
        agent = create_agent(enable_memory=True, verbose=False)
        
        # Test goal setting
        agent.set_session_goal("Test compliance analysis")
        assert agent.memory_manager.session.get_goal() == "Test compliance analysis"
        
        # Test subtask management
        task_id = agent.add_subtask("Load building data")
        assert task_id is not None
        
        success = agent.update_subtask_status(task_id, TaskStatus.COMPLETED)
        assert success
        
        # Test memory summary
        summary = agent.get_memory_summary()
        assert summary is not None
        assert "session_id" in summary
    
    def test_agent_status_with_memory(self):
        """Test agent status includes memory information."""
        with patch('aec_agent.agent.ChatOpenAI') as mock_llm:
            mock_llm.return_value = Mock()
            
            agent = create_agent(enable_memory=True, verbose=False)
            status = agent.get_status()
            
            assert "memory_enabled" in status
            assert status["memory_enabled"]
            assert "session_id" in status
            assert "memory_summary" in status


class TestMemoryPersistence:
    """Test memory persistence functionality."""
    
    def test_session_state_serialization(self):
        """Test session state can be serialized and deserialized."""
        memory = SessionMemory()
        
        # Add data
        memory.set_goal("Test goal")
        memory.add_subtask("Test task")
        memory.add_active_file("/test/file.ifc")
        
        # Test serialization
        state_dict = memory.state.model_dump()
        assert isinstance(state_dict, dict)
        assert state_dict["goal"] == "Test goal"
        
        # Test JSON serialization works
        json_str = json.dumps(state_dict, default=str)
        assert isinstance(json_str, str)
        
        # Test deserialization
        loaded_dict = json.loads(json_str)
        assert loaded_dict["goal"] == "Test goal"
    
    def test_tool_execution_serialization(self):
        """Test tool execution records can be serialized."""
        execution = ToolExecution(
            tool_name="test_tool",
            arguments={"param": "value"},
            success=True,
            result_summary="Test execution"
        )
        
        # Test serialization
        execution_dict = execution.model_dump()
        assert isinstance(execution_dict, dict)
        assert execution_dict["tool_name"] == "test_tool"
        
        # Test JSON serialization
        json_str = json.dumps(execution_dict, default=str)
        assert isinstance(json_str, str)
    
    def test_subtask_serialization(self):
        """Test subtask serialization."""
        subtask = SubTask(
            name="Test subtask",
            status=TaskStatus.PENDING,
            notes="Test notes"
        )
        
        # Test serialization
        subtask_dict = subtask.model_dump()
        assert isinstance(subtask_dict, dict)
        assert subtask_dict["name"] == "Test subtask"
        assert subtask_dict["status"] == TaskStatus.PENDING
        
        # Test JSON serialization
        json_str = json.dumps(subtask_dict, default=str)
        assert isinstance(json_str, str)


if __name__ == "__main__":
    pytest.main([__file__])