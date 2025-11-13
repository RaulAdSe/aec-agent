"""
Shared test fixtures for reactive agent testing.
"""

import pytest
import os
import uuid
from typing import Dict, Any, List, Optional

# Set test environment 
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")

from aec_agent.core.reasoning_utils import Task, TaskStatus, Priority, ExecutionResult
from aec_agent.core.replanner import ReplanningTrigger, ReplanningResult
from aec_agent.memory.execution_memory import ExecutionMemory, ValidationLevel
from aec_agent.memory import MemoryManager, MemoryManagerConfig, ShortTermMemoryConfig


@pytest.fixture
def test_config():
    """Memory manager configuration for testing (disables LLM features)."""
    return MemoryManagerConfig(
        short_term_config=ShortTermMemoryConfig(
            enable_conversation_summarization=False,
            buffer_memory_max_size=100
        ),
        enable_persistence=False
    )


@pytest.fixture 
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="test-task-123",
        name="Count doors",
        description="Count all doors in the building",
        priority=Priority.HIGH
    )


@pytest.fixture
def sample_tasks():
    """Create a list of sample tasks for testing."""
    return [
        Task(
            id="task-1",
            name="Load building data",
            description="Load building data from file",
            priority=Priority.HIGH
        ),
        Task(
            id="task-2", 
            name="Get door elements",
            description="Get all door elements from building",
            priority=Priority.MEDIUM
        ),
        Task(
            id="task-3",
            name="Count doors", 
            description="Count total number of doors",
            priority=Priority.MEDIUM
        ),
        Task(
            id="task-4",
            name="Report results",
            description="Report door count results",
            priority=Priority.LOW
        )
    ]


@pytest.fixture
def successful_execution_result():
    """Create a successful execution result."""
    return ExecutionResult(
        success=True,
        tool_name="load_building_data",
        output={
            "status": "success",
            "data": {
                "doors": 15,
                "total_elements": 150,
                "building_name": "Test Building"
            }
        },
        execution_time=0.5
    )


@pytest.fixture  
def failed_execution_result():
    """Create a failed execution result."""
    return ExecutionResult(
        success=False,
        tool_name="query_elements",
        output=None,
        error_message="Query failed: invalid parameters",
        execution_time=0.1
    )


@pytest.fixture
def sample_execution_context():
    """Create sample execution context for testing."""
    return {
        "goal": "Count doors in test building",
        "current_iteration": 2,
        "recent_execution_steps": [
            {
                "task": "Load building data",
                "tool": "load_building_data", 
                "success": True,
                "confidence": 0.9
            },
            {
                "task": "Query doors",
                "tool": "query_elements",
                "success": False, 
                "confidence": 0.3,
                "issues": ["Invalid query parameters"]
            }
        ],
        "recent_failures": [
            {
                "task": "Query doors",
                "tool": "query_elements",
                "error": "Query failed: invalid parameters"
            }
        ],
        "recent_context_discoveries": [
            {
                "type": "building_info",
                "key": "total_doors",
                "value": 15,
                "source": "load_building_data",
                "confidence": 0.95
            }
        ],
        "plan_confidence": 0.7,
        "discovered_context_summary": {
            "building_data_loaded": True,
            "total_doors": 15,
            "building_name": "Test Building"
        }
    }


@pytest.fixture
def sample_replanning_trigger():
    """Create a sample replanning trigger."""
    return ReplanningTrigger(
        trigger_type="validation_failure",
        details={
            "failed_task": "Query doors", 
            "tool": "query_elements",
            "error": "Query failed"
        },
        confidence=0.8,
        reasoning="Tool execution failed consistently"
    )


@pytest.fixture
def execution_memory():
    """Create an execution memory instance for testing."""
    memory = ExecutionMemory("test-session", "Count doors in building")
    
    # Add some sample execution steps
    memory.start_iteration(1)
    
    # Step 1: Successful load
    step1_id = memory.record_execution_step(
        task_name="Load data",
        task_id="task-1",
        tool_name="load_building_data",
        tool_arguments={"file_path": "test.json"},
        tool_success=True
    )
    memory.add_validation_result(step1_id, {
        "success": True,
        "confidence": 0.9,
        "validation_level": "llm_intelligent",
        "method": "llm_reasoning",
        "issues": []
    })
    
    # Step 2: Failed query
    memory.start_iteration(2) 
    step2_id = memory.record_execution_step(
        task_name="Query doors",
        task_id="task-2",
        tool_name="query_elements", 
        tool_arguments={"element_type": "doors"},
        tool_success=False,
        error_message="Query failed"
    )
    memory.add_validation_result(step2_id, {
        "success": False,
        "confidence": 0.2,
        "validation_level": "execution",
        "method": "basic_check",
        "issues": ["Tool execution failed"]
    })
    
    # Add some context discoveries
    memory.discover_context("building_info", "total_doors", 15, "load_building_data")
    memory.discover_context("file_info", "building_loaded", True, "load_building_data")
    
    return memory


@pytest.fixture
def memory_manager(test_config):
    """Create a memory manager for testing."""
    return MemoryManager(config=test_config)


from langchain.schema.output_parser import BaseOutputParser

class MockResponse:
    """Mock response object that mimics LangChain response."""
    
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content


class MockOutputParser(BaseOutputParser):
    """Mock output parser that returns strings directly."""
    
    def parse(self, text: str) -> str:
        return text

    @property
    def _type(self) -> str:
        return "mock_parser"


class MockLLM:
    """Mock LLM for testing that avoids API calls and works with LangChain."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = None
    
    def invoke(self, prompt_input: Dict[str, Any]) -> MockResponse:
        """Mock LLM invocation with comprehensive responses for all agent components."""
        self.call_count += 1
        self.last_prompt = prompt_input
        
        # Return predefined responses based on prompt content
        prompt_text = str(prompt_input).lower()
        
        # Progress Evaluator responses
        if "goal_achieved" in prompt_text or "progress" in prompt_text:
            if "doors" in prompt_text and "count" in prompt_text:
                content = '{"goal_achieved": true, "confidence": 0.9, "reasoning": "Door count completed successfully with 15 doors found", "completion_percentage": 100, "missing_requirements": [], "evidence_for_completion": ["Building data loaded", "Door count executed", "15 doors found"], "recommendations": ["Goal achieved"]}'
            else:
                content = '{"goal_achieved": false, "confidence": 0.6, "reasoning": "More work needed", "completion_percentage": 40, "missing_requirements": ["Complete remaining tasks"], "evidence_for_completion": ["Loaded data"], "recommendations": ["Continue execution"]}'
        
        # Replanner responses
        elif "replanning" in prompt_text or "replan" in prompt_text or "revise task" in prompt_text:
            content = '{"reasoning": "Need alternative approach due to validation failure", "confidence": 0.7, "tasks_to_remove": [], "tasks_to_modify": [], "tasks_to_add": [{"name": "Try alternative tool", "description": "Use get_all_elements instead of query_elements", "priority": "HIGH"}]}'
        
        # Tool Planner responses  
        elif "tool selection" in prompt_text or "select" in prompt_text or "best tool" in prompt_text:
            if "doors" in prompt_text:
                content = "get_all_elements"
            elif "load" in prompt_text and "building" in prompt_text:
                content = "load_building_data"
            elif "calculate" in prompt_text or "count" in prompt_text:
                content = "calculate_metrics"
            elif "query" in prompt_text or "filter" in prompt_text:
                content = "query_elements"
            else:
                content = "get_all_elements"
        
        # Validator responses
        elif "validation" in prompt_text or "validate" in prompt_text:
            if "tool output" in prompt_text or "execution" in prompt_text:
                content = '{"success": true, "message": "Tool execution validated successfully", "confidence": 0.9, "issues": []}'
            else:
                content = '{"success": true, "message": "Mock validation successful", "confidence": 0.8, "issues": []}'
        
        # Default response for any other prompts
        else:
            content = '{"success": true, "message": "Mock LLM response", "confidence": 0.7}'
        
        return MockResponse(content)


@pytest.fixture 
def mock_llm():
    """Create a mock LLM for testing."""
    return MockLLM()


def create_test_task(name: str, description: str, task_id: str = None) -> Task:
    """Helper function to create test tasks."""
    return Task(
        id=task_id or str(uuid.uuid4()),
        name=name,
        description=description,
        priority=Priority.MEDIUM
    )


def create_test_execution_result(tool_name: str, success: bool = True, output=None) -> ExecutionResult:
    """Helper function to create test execution results."""
    return ExecutionResult(
        success=success,
        tool_name=tool_name,
        output=output or {"status": "success" if success else "error", "data": {"test": True}},
        error_message=None if success else "Test error",
        execution_time=0.1
    )


def assert_task_valid(task: Task):
    """Helper to assert task is valid."""
    assert task.id is not None
    assert task.name is not None
    assert task.description is not None
    assert isinstance(task.status, TaskStatus)
    assert isinstance(task.priority, Priority)


def assert_execution_result_valid(result: ExecutionResult):
    """Helper to assert execution result is valid."""
    assert isinstance(result.success, bool)
    assert result.tool_name is not None
    assert result.execution_time >= 0
    if result.success:
        assert result.output is not None
    else:
        assert result.error_message is not None


def assert_replanning_result_valid(result: ReplanningResult):
    """Helper to assert replanning result is valid."""
    assert isinstance(result.success, bool)
    assert isinstance(result.new_tasks, list)
    assert isinstance(result.removed_task_ids, list)
    assert isinstance(result.modified_task_ids, list)
    assert result.reasoning is not None
    assert 0.0 <= result.confidence <= 1.0