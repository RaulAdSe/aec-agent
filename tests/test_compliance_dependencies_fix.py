"""
Tests for Fix 2: Compliance Workflow Dependencies

This module tests that compliance search tasks do not depend on spatial analysis
and that validation properly depends on compliance search.
"""

import pytest
import uuid
from unittest.mock import Mock, patch

from aec_agent.core.goal_decomposer import GoalDecomposer
from aec_agent.core.reasoning_utils import Task, Priority
from aec_agent.config import AgentConfig


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    # Mock the chain invoke method
    llm.invoke.return_value = Mock(content='[{"name": "Load building data", "description": "Load data", "priority": "HIGH"}, {"name": "Search compliance documents", "description": "Search docs", "priority": "MEDIUM"}, {"name": "Validate compliance", "description": "Validate", "priority": "LOW"}]')
    return llm


@pytest.fixture
def decomposer(mock_llm):
    """Create a GoalDecomposer with mocked LLM."""
    return GoalDecomposer(llm=mock_llm)


def test_create_compliance_dependencies_basic_workflow():
    """Test basic compliance dependency structure."""
    # Create decomposer with mock LLM to avoid OpenAI API key requirement
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    # Create sample tasks
    load_task = Task(
        id="load-id",
        name="Load building data",
        description="Load building data from file",
        dependencies=[]
    )
    
    search_task = Task(
        id="search-id", 
        name="Search compliance documents",
        description="Search compliance documents",
        dependencies=["load-id"]  # Initially depends on load task
    )
    
    spatial_task = Task(
        id="spatial-id",
        name="Find spatial relationships", 
        description="Find spatial relationships",
        dependencies=["load-id"]  # Should not block compliance search
    )
    
    validate_task = Task(
        id="validate-id",
        name="Validate compliance",
        description="Validate compliance requirements",
        dependencies=["spatial-id"]  # Wrong dependency initially
    )
    
    tasks = [load_task, search_task, spatial_task, validate_task]
    
    # Apply compliance dependency fix
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    # Find tasks after fix
    search_fixed = next(t for t in fixed_tasks if "compliance" in t.name.lower() and "search" in t.name.lower())
    validate_fixed = next(t for t in fixed_tasks if "validate" in t.name.lower())
    spatial_fixed = next(t for t in fixed_tasks if "spatial" in t.name.lower())
    
    # Compliance search should only depend on load task
    assert search_fixed.dependencies == ["load-id"]
    
    # Validation should depend on compliance search (and possibly load task)
    assert "search-id" in validate_fixed.dependencies
    
    # Spatial task should not block compliance workflow
    assert "spatial-id" not in search_fixed.dependencies


def test_compliance_search_independence():
    """Test that compliance search doesn't depend on spatial analysis."""
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    # Create tasks with problematic dependencies
    load_task = Task(
        id="load-id",
        name="Load building data for stair analysis",
        description="Load data",
        dependencies=[]
    )
    
    spatial_task = Task(
        id="spatial-id",
        name="Find spatial relationships for stairs",
        description="Find relationships",
        dependencies=["load-id"]
    )
    
    compliance_task = Task(
        id="compliance-id",
        name="Search compliance documents for stair requirements", 
        description="Search compliance",
        dependencies=["spatial-id"]  # Wrong - should not depend on spatial
    )
    
    tasks = [load_task, spatial_task, compliance_task]
    
    # Apply fix
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    compliance_fixed = next(t for t in fixed_tasks if "compliance" in t.name.lower())
    
    # Compliance search should not depend on spatial analysis
    assert "spatial-id" not in compliance_fixed.dependencies
    
    # Should only depend on load task
    assert compliance_fixed.dependencies == ["load-id"]


def test_validation_depends_on_compliance_search():
    """Test that validation properly depends on compliance search."""
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    load_task = Task(
        id="load-id",
        name="Load building data",
        description="Load data",
        dependencies=[]
    )
    
    search_task = Task(
        id="search-id",
        name="Search compliance documents for fire safety",
        description="Search compliance docs",
        dependencies=["load-id"]
    )
    
    validate_task = Task(
        id="validate-id",
        name="Validate building against fire safety rules",
        description="Validate compliance",
        dependencies=["load-id"]  # Missing dependency on search
    )
    
    tasks = [load_task, search_task, validate_task]
    
    # Apply fix
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    validate_fixed = next(t for t in fixed_tasks if "validate" in t.name.lower())
    
    # Validation should depend on compliance search
    assert "search-id" in validate_fixed.dependencies


def test_no_compliance_tasks_unchanged():
    """Test that non-compliance workflows remain unchanged."""
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    task_a = Task(
        id="a",
        name="Get all doors",
        description="Get doors",
        dependencies=[]
    )
    
    task_b = Task(
        id="b", 
        name="Calculate door areas",
        description="Calculate areas",
        dependencies=["a"]
    )
    
    tasks = [task_a, task_b]
    original_deps = {t.id: list(t.dependencies) for t in tasks}
    
    # Apply fix
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    # Dependencies should remain unchanged
    for task in fixed_tasks:
        assert task.dependencies == original_deps[task.id]


def test_multiple_compliance_tasks():
    """Test handling of multiple compliance-related tasks."""
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    load_task = Task(
        id="load-id",
        name="Load building data",
        description="Load data",
        dependencies=[]
    )
    
    search_fire = Task(
        id="search-fire-id",
        name="Search compliance documents for fire safety",
        description="Search fire compliance",
        dependencies=["load-id"]
    )
    
    search_access = Task(
        id="search-access-id", 
        name="Search compliance documents for accessibility",
        description="Search accessibility compliance",
        dependencies=["load-id"]
    )
    
    validate_fire = Task(
        id="validate-fire-id",
        name="Validate fire safety compliance",
        description="Validate fire safety", 
        dependencies=["load-id"]  # Missing search dependency
    )
    
    tasks = [load_task, search_fire, search_access, validate_fire]
    
    # Apply fix
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    # Find the first search task and validation task
    validate_fixed = next(t for t in fixed_tasks if "validate" in t.name.lower())
    
    # Validation should depend on at least one search task
    search_ids = [t.id for t in fixed_tasks if "search" in t.name.lower() and "compliance" in t.name.lower()]
    assert any(search_id in validate_fixed.dependencies for search_id in search_ids)


def test_case_insensitive_task_matching():
    """Test that task matching is case insensitive."""
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    load_task = Task(
        id="load-id",
        name="LOAD BUILDING DATA",
        description="Load data",
        dependencies=[]
    )
    
    search_task = Task(
        id="search-id",
        name="SEARCH COMPLIANCE DOCUMENTS",
        description="Search docs",
        dependencies=[]
    )
    
    validate_task = Task(
        id="validate-id",
        name="VALIDATE AGAINST RULES",
        description="Validate",
        dependencies=[]
    )
    
    tasks = [load_task, search_task, validate_task]
    
    # Apply fix
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    search_fixed = next(t for t in fixed_tasks if "SEARCH" in t.name)
    validate_fixed = next(t for t in fixed_tasks if "VALIDATE" in t.name)
    
    # Should work with uppercase names
    assert search_fixed.dependencies == ["load-id"]
    assert "search-id" in validate_fixed.dependencies


def test_partial_workflow_missing_load_task():
    """Test behavior when load task is missing."""
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    search_task = Task(
        id="search-id",
        name="Search compliance documents",
        description="Search docs",
        dependencies=[]
    )
    
    validate_task = Task(
        id="validate-id", 
        name="Validate compliance",
        description="Validate",
        dependencies=[]
    )
    
    tasks = [search_task, validate_task]
    
    # Apply fix - should not crash
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    validate_fixed = next(t for t in fixed_tasks if "validate" in t.name.lower())
    
    # Validation should still depend on search
    assert "search-id" in validate_fixed.dependencies


def test_partial_workflow_missing_search_task():
    """Test behavior when search task is missing.""" 
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    
    load_task = Task(
        id="load-id",
        name="Load building data", 
        description="Load data",
        dependencies=[]
    )
    
    validate_task = Task(
        id="validate-id",
        name="Validate compliance",
        description="Validate", 
        dependencies=["load-id"]
    )
    
    tasks = [load_task, validate_task]
    
    # Apply fix - should not crash or change dependencies
    fixed_tasks = decomposer._create_compliance_dependencies(tasks)
    
    validate_fixed = next(t for t in fixed_tasks if "validate" in t.name.lower())
    
    # Should keep existing dependencies
    assert validate_fixed.dependencies == ["load-id"]