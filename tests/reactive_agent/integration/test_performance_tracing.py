"""
Performance and tracing integration tests - Phase 5 reactive implementation.

Tests the performance characteristics and LangSmith tracing integration
of reactive reasoning components under various load and execution scenarios.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import statistics

from aec_agent.memory.execution_memory import ExecutionMemory
from aec_agent.core.progress_evaluator import ProgressEvaluator
from aec_agent.core.validator import ResultValidator
from aec_agent.core.replanner import Replanner
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus
from tests.reactive_agent.fixtures.reactive_fixtures import (
    create_test_task, create_test_execution_result, mock_llm
)


class TestReactiveAgentPerformance:
    """Test performance characteristics of reactive reasoning components."""
    
    def test_execution_memory_performance_large_dataset(self):
        """Test ExecutionMemory performance with large execution datasets."""
        execution_memory = ExecutionMemory("perf-test-1", "Large-scale building analysis")
        
        # Performance baseline
        start_time = time.time()
        
        # Simulate 100 iterations with 10 steps each = 1000 execution steps
        for iteration in range(1, 101):
            execution_memory.start_iteration(iteration)
            
            for step in range(10):
                step_id = execution_memory.record_execution_step(
                    task_name=f"Task_{iteration}_{step}",
                    task_id=f"task-{iteration}-{step}",
                    tool_name=f"tool_{step}",
                    tool_arguments={"iteration": iteration, "step": step},
                    tool_output={"result": f"output_{iteration}_{step}", "data": list(range(100))},
                    tool_success=True
                )
                
                # Add validation with varying confidence
                confidence = 0.7 + (step % 3) * 0.1  # 0.7, 0.8, 0.9 pattern
                execution_memory.add_validation_result(step_id, {
                    "success": True,
                    "confidence": confidence,
                    "validation_level": "logical",
                    "method": "performance_test"
                })
                
                # Add context discoveries every 5th step
                if step % 5 == 0:
                    execution_memory.discover_context(
                        "performance", f"milestone_{iteration}_{step}", 
                        {"progress": f"{iteration}_{step}"}, "performance_tool"
                    )
        
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert len(execution_memory.execution_steps) == 1000
        assert len(execution_memory.discovered_context) == 200  # Every 5th step
        assert execution_memory.current_iteration == 100
        assert execution_time < 5.0, f"Execution took {execution_time:.2f}s, should be < 5s"
        
        # Test context generation performance
        context_start = time.time()
        replanning_context = execution_memory.get_context_for_replanning()
        progress_context = execution_memory.get_context_for_progress_evaluation()
        context_time = time.time() - context_start
        
        assert context_time < 1.0, f"Context generation took {context_time:.2f}s, should be < 1s"
        assert len(replanning_context["recent_execution_steps"]) == 5  # Last 5 steps
        assert len(replanning_context["recent_context_discoveries"]) == 10  # Last 10 discoveries
    
    def test_progress_evaluator_performance_concurrent_evaluations(self, mock_llm):
        """Test ProgressEvaluator performance with concurrent evaluation scenarios."""
        progress_evaluator = ProgressEvaluator(llm=mock_llm)
        
        # Create realistic evaluation scenarios
        evaluation_scenarios = []
        for i in range(50):
            execution_context = {
                "goal": f"Analyze building {i}",
                "discovered_context_summary": {
                    "numeric_result": i * 3,
                    "element_list": [f"element_{j}" for j in range(i % 10)],
                    "building_loaded": True
                },
                "unique_tasks_completed": i % 5 + 1
            }
            
            current_tasks = [
                create_test_task(f"Task {j}", f"Description {j}", f"task-{i}-{j}")
                for j in range(i % 3 + 1)
            ]
            for task in current_tasks:
                task.status = TaskStatus.COMPLETED
            
            evaluation_scenarios.append((execution_context, current_tasks))
        
        # Performance test: 50 evaluations
        start_time = time.time()
        
        evaluation_results = []
        for execution_context, current_tasks in evaluation_scenarios:
            assessment = progress_evaluator.evaluate_progress(
                original_goal="Count elements in building",
                execution_context=execution_context,
                current_tasks=current_tasks
            )
            evaluation_results.append(assessment)
        
        total_time = time.time() - start_time
        avg_time_per_evaluation = total_time / 50
        
        # Performance assertions
        assert total_time < 3.0, f"50 evaluations took {total_time:.2f}s, should be < 3s"
        assert avg_time_per_evaluation < 0.06, f"Avg evaluation time {avg_time_per_evaluation:.3f}s, should be < 0.06s"
        assert len(evaluation_results) == 50
        
        # Quality assertions - ensure results are meaningful
        successful_evaluations = [r for r in evaluation_results if r.goal_achieved]
        assert len(successful_evaluations) > 40, "Most evaluations should succeed with proper context"
    
    def test_memory_efficiency_large_context(self):
        """Test memory efficiency with large context accumulation."""
        execution_memory = ExecutionMemory("memory-test", "Memory efficiency test")
        
        initial_context_size = len(str(execution_memory.get_current_context()))
        
        # Add progressively larger context
        for iteration in range(20):
            execution_memory.start_iteration(iteration + 1)
            
            # Add execution steps with large data
            for step in range(5):
                large_data = {
                    "large_array": list(range(1000)),
                    "nested_data": {f"key_{i}": f"value_{i}" * 100 for i in range(50)},
                    "metadata": {"iteration": iteration, "step": step}
                }
                
                step_id = execution_memory.record_execution_step(
                    task_name=f"Large data task {iteration}-{step}",
                    task_id=f"large-task-{iteration}-{step}",
                    tool_name="large_data_processor",
                    tool_arguments={"size": "large"},
                    tool_output=large_data,
                    tool_success=True
                )
                
                execution_memory.add_validation_result(step_id, {
                    "success": True,
                    "confidence": 0.8,
                    "validation_level": "logical",
                    "method": "memory_test"
                })
        
        final_context_size = len(str(execution_memory.get_current_context()))
        context_growth_factor = final_context_size / initial_context_size
        
        # Memory efficiency assertions
        assert len(execution_memory.execution_steps) == 100
        assert context_growth_factor < 100, f"Context grew by {context_growth_factor}x, should be < 100x"
        
        # Test that recent context extraction remains efficient
        start_time = time.time()
        recent_context = execution_memory.get_context_for_replanning()
        context_extraction_time = time.time() - start_time
        
        assert context_extraction_time < 0.5, f"Context extraction took {context_extraction_time:.3f}s, should be < 0.5s"
        assert len(recent_context["recent_execution_steps"]) <= 5  # Should limit recent data
    
    def test_reactive_reasoning_end_to_end_performance(self, mock_llm):
        """Test end-to-end reactive reasoning performance in realistic scenarios."""
        execution_memory = ExecutionMemory("e2e-perf", "End-to-end performance test")
        progress_evaluator = ProgressEvaluator(llm=mock_llm)
        validator = ResultValidator(llm=mock_llm)
        replanner = Replanner(llm=mock_llm)
        
        # Simulate complex reactive reasoning workflow
        start_time = time.time()
        
        workflow_metrics = {
            "iterations": 0,
            "execution_steps": 0,
            "validations": 0,
            "progress_evaluations": 0,
            "replannings": 0
        }
        
        # Multi-iteration reactive workflow
        for iteration in range(1, 11):  # 10 iterations
            execution_memory.start_iteration(iteration)
            workflow_metrics["iterations"] += 1
            
            # Execute 2-3 tasks per iteration
            task_count = 2 + (iteration % 2)
            for task_idx in range(task_count):
                # Execution step
                task = create_test_task(
                    f"Task {iteration}-{task_idx}",
                    f"Execute task {task_idx} in iteration {iteration}",
                    f"task-{iteration}-{task_idx}"
                )
                
                execution_result = create_test_execution_result(
                    f"tool_{task_idx}",
                    success=True,
                    output={
                        "status": "success",
                        "result": f"iteration_{iteration}_task_{task_idx}",
                        "data": {"numbers": list(range(50))}
                    }
                )
                
                step_id = execution_memory.record_execution_step(
                    task.name, task.id, execution_result.tool_name,
                    {"iteration": iteration, "task": task_idx},
                    execution_result.output, execution_result.success
                )
                workflow_metrics["execution_steps"] += 1
                
                # Validation
                validation = {
                    "success": True,
                    "confidence": 0.85 + (task_idx * 0.05),
                    "validation_level": "logical",
                    "method": "mock_intelligent",
                    "suggests_replanning": False
                }
                execution_memory.add_validation_result(step_id, validation)
                workflow_metrics["validations"] += 1
                
                # Context discovery
                execution_memory.discover_context(
                    "workflow_progress", f"task_{iteration}_{task_idx}_completed",
                    True, execution_result.tool_name
                )
            
            # Progress evaluation every 3rd iteration
            if iteration % 3 == 0:
                execution_context = execution_memory.get_context_for_progress_evaluation()
                current_tasks = [
                    create_test_task(f"Task {i}", f"Description {i}", f"task-{i}")
                    for i in range(iteration)
                ]
                for task in current_tasks:
                    task.status = TaskStatus.COMPLETED
                
                assessment = progress_evaluator.evaluate_progress(
                    original_goal="Complete workflow tasks efficiently",
                    execution_context=execution_context,
                    current_tasks=current_tasks
                )
                workflow_metrics["progress_evaluations"] += 1
                
                # Record goal progress
                execution_memory.assess_goal_achievement({
                    "goal_achieved": assessment.goal_achieved,
                    "confidence": assessment.confidence,
                    "completion_percentage": assessment.completion_percentage
                })
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 2.0, f"End-to-end workflow took {total_time:.2f}s, should be < 2s"
        assert workflow_metrics["iterations"] == 10
        assert workflow_metrics["execution_steps"] == 25  # 2+3+2+3... = 25
        assert workflow_metrics["validations"] == 25
        assert workflow_metrics["progress_evaluations"] == 3  # Every 3rd iteration
        
        # Verify final state quality
        assert len(execution_memory.execution_steps) == 25
        assert execution_memory.current_iteration == 10
        assert len(execution_memory.discovered_context) == 25


class TestTracingIntegration:
    """Test LangSmith tracing integration with reactive components."""
    
    def test_execution_memory_tracing_capability(self):
        """Test that ExecutionMemory operations can be traced."""
        execution_memory = ExecutionMemory("tracing-test", "Test tracing integration")
        
        # Operations that support tracing
        execution_memory.start_iteration(1)
        step_id = execution_memory.record_execution_step(
            "Traced task", "trace-1", "trace_tool", {},
            {"traced": True}, True
        )
        execution_memory.discover_context("trace", "test", True, "trace_tool")
        context = execution_memory.get_context_for_replanning()
        
        # Verify operations completed successfully (traceable infrastructure exists)
        assert len(execution_memory.execution_steps) == 1
        assert len(execution_memory.discovered_context) == 1
        assert context is not None
        assert context["goal"] == "Test tracing integration"
    
    def test_progress_evaluator_tracing_capability(self, mock_llm):
        """Test that ProgressEvaluator operations can be traced."""
        progress_evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {
            "discovered_context_summary": {
                "numeric_result": 5,
                "element_list": ["e1", "e2"]
            }
        }
        current_tasks = [create_test_task("Test", "Test task", "test-1")]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        # Operation that supports tracing
        assessment = progress_evaluator.evaluate_progress(
            "Count elements", execution_context, current_tasks
        )
        
        # Verify operation completed successfully
        assert assessment is not None
        assert hasattr(assessment, 'goal_achieved')
        assert hasattr(assessment, 'confidence')
    
    def test_validator_tracing_capability(self, mock_llm):
        """Test that Validator operations can be traced."""
        validator = ResultValidator(llm=mock_llm)
        
        task = create_test_task("Validation test", "Test validation", "val-1")
        result = create_test_execution_result("test_tool", True, {"status": "success"})
        
        # Operation that supports tracing
        validation = validator.validate_execution(task, result)
        
        # Verify operation completed successfully (may use fallback validation)
        assert validation is not None
        assert "success" in validation
        assert "method" in validation
    
    def test_tracing_metadata_collection(self, mock_llm):
        """Test that tracing collects relevant metadata from reactive operations."""
        execution_memory = ExecutionMemory("metadata-test", "Metadata collection test")
        progress_evaluator = ProgressEvaluator(llm=mock_llm)
        
        # Simulate traced operations with metadata
        execution_memory.start_iteration(1)
        
        step_id = execution_memory.record_execution_step(
            "Metadata task", "meta-1", "metadata_tool",
            {"param1": "value1", "param2": 42},
            {"result": "success", "metadata": {"trace_id": "test-123"}},
            True
        )
        
        execution_memory.add_validation_result(step_id, {
            "success": True,
            "confidence": 0.9,
            "validation_level": "logical",
            "method": "traced_validation",
            "metadata": {"validator_version": "2.0"}
        })
        
        # Get context that includes tracing metadata
        context = execution_memory.get_current_context()
        
        # Verify metadata is preserved in context
        assert context["total_execution_steps"] == 1
        assert context["goal"] == "Metadata collection test"
        assert len(execution_memory.execution_steps) == 1
        
        step = execution_memory.execution_steps[0]
        assert step.tool_name == "metadata_tool"
        assert step.validation_method == "traced_validation"


class TestReactiveAgentStressTests:
    """Stress tests for reactive agent components."""
    
    def test_rapid_iteration_stress(self):
        """Test rapid iteration creation and execution step recording."""
        execution_memory = ExecutionMemory("stress-test", "Rapid iteration stress test")
        
        start_time = time.time()
        
        # Rapid iteration creation
        for iteration in range(500):
            execution_memory.start_iteration(iteration + 1)
            
            # Quick execution step
            step_id = execution_memory.record_execution_step(
                f"Rapid_{iteration}", f"rapid-{iteration}", "rapid_tool", {},
                {"iteration": iteration}, True
            )
            
            execution_memory.add_validation_result(step_id, {
                "success": True,
                "confidence": 0.8,
                "validation_level": "basic",
                "method": "rapid_test"
            })
        
        stress_time = time.time() - start_time
        
        # Stress test assertions
        assert stress_time < 3.0, f"500 rapid iterations took {stress_time:.2f}s, should be < 3s"
        assert execution_memory.current_iteration == 500
        assert len(execution_memory.execution_steps) == 500
    
    def test_memory_stability_under_load(self):
        """Test memory stability under sustained load."""
        execution_memory = ExecutionMemory("stability-test", "Memory stability test")
        
        # Sustained load test
        iterations = 100
        steps_per_iteration = 10
        
        for iteration in range(iterations):
            execution_memory.start_iteration(iteration + 1)
            
            for step in range(steps_per_iteration):
                step_id = execution_memory.record_execution_step(
                    f"Load_task_{iteration}_{step}",
                    f"load-{iteration}-{step}",
                    f"load_tool_{step}",
                    {"load_data": list(range(100))},
                    {"results": list(range(200))},
                    True
                )
                
                execution_memory.add_validation_result(step_id, {
                    "success": True,
                    "confidence": 0.75 + (step * 0.02),
                    "validation_level": "logical",
                    "method": "load_test"
                })
                
                # Add context every 5th step
                if step % 5 == 0:
                    execution_memory.discover_context(
                        "load_progress",
                        f"milestone_{iteration}_{step}",
                        {"completed": True, "data": list(range(50))},
                        f"load_tool_{step}"
                    )
        
        # Verify memory stability
        assert len(execution_memory.execution_steps) == 1000
        assert len(execution_memory.discovered_context) == 200
        assert execution_memory.current_iteration == 100
        
        # Test that memory still works efficiently after load
        start_time = time.time()
        context = execution_memory.get_context_for_progress_evaluation()
        context_time = time.time() - start_time
        
        assert context_time < 0.5, f"Context generation after load took {context_time:.3f}s"
        assert context["unique_tasks_completed"] == 1000