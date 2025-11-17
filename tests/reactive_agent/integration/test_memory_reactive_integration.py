"""
Memory system integration with reactive components - Phase 4 reactive implementation.

Tests the integration of ExecutionMemory with reactive reasoning components
and the broader memory architecture (MemoryManager, SessionMemory, ShortTermMemory).
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import tempfile
from pathlib import Path

from aec_agent.memory.memory_manager import MemoryManager, MemoryManagerConfig
from aec_agent.memory.execution_memory import ExecutionMemory
from aec_agent.memory.short_term_memory import ShortTermMemoryConfig
from aec_agent.core.progress_evaluator import ProgressEvaluator, ProgressAssessment
from aec_agent.core.validator import ResultValidator
from aec_agent.core.replanner import Replanner, ReplanningTrigger
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus, ExecutionResult
from tests.reactive_agent.fixtures.reactive_fixtures import (
    create_test_task, create_test_execution_result, mock_llm
)


class TestMemoryManagerReactiveIntegration:
    """Test MemoryManager integration with reactive components."""
    
    def test_memory_manager_with_execution_memory(self):
        """Test MemoryManager can coordinate with ExecutionMemory for reactive reasoning."""
        # Create memory manager with test config
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryManagerConfig(
                session_persistence_path=Path(temp_dir),
                enable_persistence=False  # Disable persistence for testing
            )
            memory_manager = MemoryManager(config)
            
            # Create execution memory for reactive reasoning
            execution_memory = ExecutionMemory("reactive-session-1", "Count doors in building")
            
            # Test that both systems can coexist and maintain separate state
            
            # Add data to memory manager session
            memory_manager.add_conversation_turn("Count all doors in this building", "I'll help you count the doors.")
            memory_manager.set_session_goal("Count doors in building")
            memory_manager.track_active_file("building.ifc")
            
            # Add data to execution memory
            execution_memory.start_iteration(1)
            step_id = execution_memory.record_execution_step(
                task_name="Load building data",
                task_id="task-1",
                tool_name="load_building_data", 
                tool_arguments={"file_path": "building.ifc"},
                tool_output={"elements": 150, "building_loaded": True},
                tool_success=True
            )
            
            execution_memory.add_validation_result(step_id, {
                "success": True,
                "confidence": 0.9,
                "validation_level": "logical",
                "method": "test_controlled",
                "suggests_replanning": False
            })
            
            # Verify both systems maintain independent state
            assert memory_manager.session.state.goal == "Count doors in building"
            assert execution_memory.goal == "Count doors in building"
            assert len(memory_manager.get_active_files()) == 1
            assert len(execution_memory.execution_steps) == 1
            
            # Test context extraction for LLM prompting
            conversation_context = memory_manager.get_conversation_context()
            execution_context = execution_memory.get_context_for_progress_evaluation()
            
            assert execution_context["goal"] == "Count doors in building"
            assert execution_context["unique_tasks_completed"] == 1
            assert "Count all doors" in conversation_context
            assert len(memory_manager.get_active_files()) == 1
    
    def test_reactive_workflow_with_memory_persistence(self):
        """Test reactive workflow with memory state persistence across iterations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MemoryManagerConfig(
                session_persistence_path=Path(temp_dir) / "session.json",
                enable_persistence=True
            )
            
            # Phase 1: Initial workflow with memory persistence
            memory_manager = MemoryManager(config)
            execution_memory = ExecutionMemory("persistent-session", "Analyze building compliance")
            progress_evaluator = ProgressEvaluator()
            
            # Set up initial session state
            memory_manager.add_conversation_turn("Analyze this building for compliance", "I'll analyze the building for compliance.")
            memory_manager.set_session_goal("Analyze building compliance")
            memory_manager.track_active_file("compliance.ifc")
            
            # Execute reactive reasoning iteration
            execution_memory.start_iteration(1)
            step_id = execution_memory.record_execution_step(
                task_name="Load compliance data",
                task_id="compliance-task-1",
                tool_name="load_building_data",
                tool_arguments={"file_path": "compliance.ifc"},
                tool_output={"elements": 200, "compliance_data": True},
                tool_success=True
            )
            
            execution_memory.discover_context("compliance", "building_loaded", True, "load_building_data")
            execution_memory.discover_context("compliance", "validation_result", "preliminary", "load_building_data")
            
            # Save session state
            memory_manager.save_session()
            
            # Phase 2: Simulate new session loading persisted state
            # SessionMemory automatically loads if session_id and persistence_path are provided
            original_session_id = memory_manager.get_session_id()
            memory_manager_2 = MemoryManager(config, session_id=original_session_id)
            
            # Continue reactive reasoning with persisted context
            execution_memory.start_iteration(2)
            
            # Get combined context from both memory systems
            conversation_context = memory_manager_2.get_conversation_context()
            execution_context = execution_memory.get_context_for_progress_evaluation()
            
            # Verify persistence worked
            assert memory_manager_2.session.state.goal == "Analyze building compliance"
            assert len(memory_manager_2.get_active_files()) == 1
            
            # Test progress evaluation with combined context
            current_tasks = [create_test_task("Load compliance data", "Load compliance data", "compliance-task-1")]
            current_tasks[0].status = TaskStatus.COMPLETED
            
            assessment = progress_evaluator.evaluate_progress(
                original_goal="Analyze building compliance",
                execution_context=execution_context,
                current_tasks=current_tasks
            )
            
            assert isinstance(assessment, ProgressAssessment)
            assert assessment.completion_percentage > 0


class TestExecutionMemoryWithReactiveComponents:
    """Test ExecutionMemory working directly with reactive reasoning components."""
    
    def test_execution_memory_progress_evaluator_integration(self):
        """Test ExecutionMemory provides proper context for ProgressEvaluator."""
        execution_memory = ExecutionMemory("progress-test", "Count windows in building")
        progress_evaluator = ProgressEvaluator()
        
        # Simulate successful counting workflow
        execution_memory.start_iteration(1)
        
        # Step 1: Load building
        load_step_id = execution_memory.record_execution_step(
            "Load building", "load-1", "load_building_data", {"file": "test.ifc"},
            tool_output={"elements": 100, "building_loaded": True}, tool_success=True
        )
        execution_memory.add_validation_result(load_step_id, {
            "success": True, "confidence": 0.9, "validation_level": "logical", 
            "method": "test_controlled"
        })
        
        # Step 2: Count windows
        count_step_id = execution_memory.record_execution_step(
            "Count windows", "count-1", "get_all_elements", {"filter": "IfcWindow"},
            tool_output={"windows": [{"id": "w1"}, {"id": "w2"}], "count": 2}, tool_success=True
        )
        execution_memory.add_validation_result(count_step_id, {
            "success": True, "confidence": 0.95, "validation_level": "logical",
            "method": "test_controlled"
        })
        
        # Add required evidence for counting tasks
        execution_memory.discover_context("results", "numeric_result", 2, "get_all_elements")
        execution_memory.discover_context("results", "element_list", ["w1", "w2"], "get_all_elements")
        
        # Test progress evaluation with execution memory context
        execution_context = execution_memory.get_context_for_progress_evaluation()
        current_tasks = [
            create_test_task("Load building", "Load building data", "load-1"),
            create_test_task("Count windows", "Count window elements", "count-1")
        ]
        for task in current_tasks:
            task.status = TaskStatus.COMPLETED
        
        assessment = progress_evaluator.evaluate_progress(
            original_goal="Count windows in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        # Verify progress evaluation works with execution memory context
        assert assessment.goal_achieved is True
        assert assessment.completion_percentage == 100.0
        assert assessment.confidence >= 0.7
        assert len(assessment.evidence_for_completion) > 0
        
        # Verify execution memory is updated with goal achievement
        execution_memory.assess_goal_achievement({
            "goal_achieved": assessment.goal_achieved,
            "confidence": assessment.confidence,
            "completion_percentage": assessment.completion_percentage,
            "evidence": assessment.evidence_for_completion
        })
        
        assert execution_memory.is_goal_achieved is True
    
    def test_execution_memory_validator_integration(self):
        """Test ExecutionMemory working with enhanced Validator for replanning triggers."""
        execution_memory = ExecutionMemory("validation-test", "Query building elements")
        
        # Simulate validation workflow that triggers replanning
        execution_memory.start_iteration(1)
        
        # Step 1: Failed query
        failed_step_id = execution_memory.record_execution_step(
            "Query elements", "query-1", "query_elements", {"type": "IfcDoor"},
            tool_output=None, tool_success=False, error_message="Query syntax error"
        )
        
        # Create validation result that suggests replanning
        validation_result = {
            "success": False,
            "message": "Query failed with syntax error",
            "confidence": 0.1,
            "validation_level": "execution",
            "method": "test_controlled",
            "suggests_replanning": True,
            "replanning_confidence": 0.8,
            "replanning_reason": "Query tool failed repeatedly",
            "replanning_trigger_type": "execution_failure",
            "replanning_details": {"failed_tool": "query_elements", "error": "Query syntax error"}
        }
        
        execution_memory.add_validation_result(failed_step_id, validation_result)
        
        # Verify validation triggers replanning logic
        assert execution_memory.should_trigger_replanning() is True
        
        # Test recording plan modification based on validation
        mod_id = execution_memory.record_plan_modification(
            trigger_type="validation_failure",
            trigger_details={"validation_result": validation_result},
            original_plan_summary="Use query_elements for door retrieval",
            new_plan_summary="Use get_all_elements with filtering",
            modification_reasoning="query_elements failed, switching to alternative approach",
            tasks_added=["filter_task"],
            tasks_removed=["query_task"]
        )
        
        assert len(execution_memory.plan_modifications) == 1
        plan_mod = execution_memory.plan_modifications[0]
        assert plan_mod.trigger_type == "validation_failure"
        assert plan_mod.modification_reasoning == "query_elements failed, switching to alternative approach"
        
        # Simulate successful alternative approach
        execution_memory.start_iteration(2)
        alt_step_id = execution_memory.record_execution_step(
            "Get all elements", "alt-1", "get_all_elements", {"filter_type": "IfcDoor"},
            tool_output={"doors": [{"id": "d1"}], "count": 1}, tool_success=True
        )
        
        execution_memory.add_validation_result(alt_step_id, {
            "success": True,
            "confidence": 0.9,
            "validation_level": "logical",
            "method": "test_controlled",
            "suggests_replanning": False
        })
        
        # Add another successful step to improve average confidence
        execution_memory.start_iteration(3)
        confirm_step_id = execution_memory.record_execution_step(
            "Confirm results", "confirm-1", "verify_results", {"count": 1},
            tool_output={"verified": True, "count": 1}, tool_success=True
        )
        
        execution_memory.add_validation_result(confirm_step_id, {
            "success": True,
            "confidence": 0.95,
            "validation_level": "logical",
            "method": "test_controlled",
            "suggests_replanning": False
        })
        
        # Verify replanning is no longer needed (2 successes, 1 failure = good confidence)
        assert execution_memory.should_trigger_replanning() is False
    
    def test_execution_memory_with_replanner_integration(self):
        """Test ExecutionMemory providing context for Replanner decisions."""
        execution_memory = ExecutionMemory("replanning-test", "Find and analyze doors")
        
        # Build up execution context that would trigger replanning
        for i in range(3):
            execution_memory.start_iteration(i + 1)
            
            # Simulate repeated failures
            failed_step_id = execution_memory.record_execution_step(
                f"Attempt {i+1}", f"attempt-{i+1}", "unreliable_tool", {},
                tool_output=None, tool_success=False, error_message=f"Attempt {i+1} failed"
            )
            
            execution_memory.add_validation_result(failed_step_id, {
                "success": False,
                "confidence": 0.1,
                "validation_level": "execution", 
                "method": "test_controlled",
                "suggests_replanning": True if i >= 1 else False,
                "replanning_confidence": 0.3 + (i * 0.3)
            })
        
        # Verify execution memory recognizes need for replanning
        assert execution_memory.should_trigger_replanning() is True
        
        # Get rich replanning context 
        replanning_context = execution_memory.get_context_for_replanning()
        
        # Verify context contains relevant information for replanning
        assert replanning_context["goal"] == "Find and analyze doors"
        assert replanning_context["current_iteration"] == 3
        assert len(replanning_context["recent_failures"]) == 3
        assert replanning_context["plan_confidence"] == 1.0  # Default high confidence
        
        # Test that replanner could use this context (without actually testing Replanner)
        trigger = ReplanningTrigger(
            trigger_type="execution_failure",
            details={"repeated_failures": 3, "tool": "unreliable_tool"},
            confidence=0.9,
            reasoning="Tool failed 3 times consecutively"
        )
        
        # Simulate recording the replanning decision
        mod_id = execution_memory.record_plan_modification(
            trigger_type=trigger.trigger_type,
            trigger_details=trigger.details,
            original_plan_summary="Use unreliable_tool for door analysis",
            new_plan_summary="Use alternative_tool for door analysis", 
            modification_reasoning=trigger.reasoning,
            tasks_added=["alternative_approach_task"],
            tasks_removed=["unreliable_tool_task"]
        )
        
        assert len(execution_memory.plan_modifications) == 1


class TestCrossMemoryContextSharing:
    """Test context sharing between different memory systems."""
    
    def test_conversation_context_with_execution_state(self):
        """Test combining conversation context with execution state."""
        # Set up memory systems
        execution_memory = ExecutionMemory("context-test", "Building analysis")
        
        # Simulate conversation history through short-term memory interface
        conversation_history = [
            {"role": "user", "content": "Please analyze this building for compliance"},
            {"role": "assistant", "content": "I'll analyze the building. First, let me load the data."},
            {"role": "user", "content": "Focus on fire safety requirements"},
        ]
        
        # Simulate execution state
        execution_memory.start_iteration(1)
        step_id = execution_memory.record_execution_step(
            "Analyze compliance", "compliance-1", "compliance_checker",
            {"focus": "fire_safety"}, 
            tool_output={"fire_exits": 4, "compliance_status": "needs_review"}, 
            tool_success=True
        )
        
        execution_memory.discover_context("compliance", "fire_safety_focus", True, "compliance_checker")
        execution_memory.discover_context("compliance", "fire_exits_count", 4, "compliance_checker")
        
        # Get combined context for LLM prompting
        execution_context = execution_memory.get_current_context()
        
        # Test that execution context could be combined with conversation context
        combined_context = {
            "conversation_history": conversation_history,
            "execution_state": execution_context,
            "session_goal": execution_memory.goal,
            "current_focus": "fire_safety"
        }
        
        # Verify combined context has rich information
        assert combined_context["session_goal"] == "Building analysis"
        assert combined_context["current_focus"] == "fire_safety"
        assert len(combined_context["conversation_history"]) == 3
        assert combined_context["execution_state"]["total_execution_steps"] == 1
        assert "fire_safety_focus" in str(combined_context["execution_state"])
    
    def test_session_state_synchronization(self):
        """Test synchronization patterns between session state and execution state."""
        execution_memory = ExecutionMemory("sync-test", "Multi-step building analysis")
        
        # Simulate session-level task tracking
        session_tasks = {
            "load_data": {"status": "completed", "result": "success"},
            "analyze_structure": {"status": "in_progress", "result": None},
            "check_compliance": {"status": "pending", "result": None}
        }
        
        # Simulate execution-level step tracking
        execution_memory.start_iteration(1)
        
        # Step 1: Load data (completed)
        load_step = execution_memory.record_execution_step(
            "Load building data", "load-1", "load_tool", {},
            tool_output={"loaded": True}, tool_success=True
        )
        # Add validation result for successful step
        execution_memory.add_validation_result(load_step, {
            "success": True,
            "confidence": 0.9,
            "validation_level": "logical",
            "method": "test_controlled"
        })
        execution_memory.discover_context("session", "load_data_completed", True, "load_tool")
        
        # Step 2: Analyze structure (in progress - simulated failure)
        execution_memory.start_iteration(2)
        analysis_step = execution_memory.record_execution_step(
            "Analyze structure", "analyze-1", "analysis_tool", {},
            tool_output=None, tool_success=False, error_message="Analysis incomplete"
        )
        # Add validation result for failed step
        execution_memory.add_validation_result(analysis_step, {
            "success": False,
            "confidence": 0.1,
            "validation_level": "execution",
            "method": "test_controlled"
        })
        
        # Test that both levels maintain consistent state
        exec_summary = execution_memory.get_execution_summary()
        assert exec_summary["successful_steps"] == 1
        assert exec_summary["failed_steps"] == 1
        
        # Verify discovered context reflects session state
        context = execution_memory.get_current_context()
        assert context["load_data_completed"] is True
        
        # Simulate session state update based on execution results
        session_tasks["load_data"]["status"] = "completed"
        session_tasks["analyze_structure"]["status"] = "failed"
        
        # Test context provides information for cross-system coordination
        coordination_context = {
            "execution_summary": exec_summary,
            "session_tasks": session_tasks,
            "needs_replanning": execution_memory.should_trigger_replanning(),
            "discovered_context": execution_memory.current_context
        }
        
        assert coordination_context["execution_summary"]["success_rate"] == 0.5
        assert coordination_context["session_tasks"]["load_data"]["status"] == "completed"
        assert coordination_context["session_tasks"]["analyze_structure"]["status"] == "failed"


class TestMemorySystemErrorHandling:
    """Test error handling across memory system integration."""
    
    def test_memory_system_resilience(self):
        """Test that memory systems remain functional when one component fails."""
        execution_memory = ExecutionMemory("resilience-test", "Test resilience")
        
        # Test execution memory continues working despite external failures
        execution_memory.start_iteration(1)
        
        # Simulate external system failure (e.g., session persistence fails)
        # Execution memory should still work
        step_id = execution_memory.record_execution_step(
            "Test resilience", "resilience-1", "test_tool", {},
            tool_output={"status": "working"}, tool_success=True
        )
        
        # Add validation result to make the step count as successful
        execution_memory.add_validation_result(step_id, {
            "success": True,
            "confidence": 0.8,
            "validation_level": "logical",
            "method": "test_controlled"
        })
        
        # Even with malformed external context, execution memory should function
        external_context_failure = None  # Simulate external context failure
        
        execution_context = execution_memory.get_context_for_progress_evaluation()
        assert execution_context is not None
        assert execution_context["goal"] == "Test resilience"
        assert execution_context["unique_tasks_completed"] == 1
        
    def test_context_corruption_recovery(self):
        """Test recovery from context corruption scenarios."""
        execution_memory = ExecutionMemory("corruption-test", "Test corruption recovery")
        
        # Build some valid state
        execution_memory.start_iteration(1)
        step_id = execution_memory.record_execution_step(
            "Valid step", "valid-1", "valid_tool", {},
            tool_output={"data": "valid"}, tool_success=True
        )
        
        # Simulate context corruption
        try:
            # Try to add corrupted validation result
            execution_memory.add_validation_result(step_id, {
                "invalid_field": "should_not_exist",
                "success": "not_boolean",  # Wrong type
                "confidence": "high"  # Wrong type
            })
        except (ValueError, TypeError, KeyError):
            # Should handle corruption gracefully
            pass
        
        # Verify system still works after corruption attempt
        valid_context = execution_memory.get_current_context()
        assert valid_context["goal"] == "Test corruption recovery"
        assert len(execution_memory.execution_steps) == 1
        
        # Add valid data after corruption
        execution_memory.start_iteration(2)
        recovery_step = execution_memory.record_execution_step(
            "Recovery step", "recovery-1", "recovery_tool", {},
            tool_output={"recovered": True}, tool_success=True
        )
        
        execution_memory.add_validation_result(recovery_step, {
            "success": True,
            "confidence": 0.8,
            "validation_level": "logical",
            "method": "test_controlled"
        })
        
        # Verify system recovered
        assert len(execution_memory.execution_steps) == 2
        recovery_context = execution_memory.get_current_context()
        assert recovery_context["total_execution_steps"] == 2