"""
End-to-end integration tests for reactive reasoning loops - Phase 3.
Tests complete reactive reasoning workflows with execution memory, 
progress evaluation, validation, and replanning.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from aec_agent.memory.execution_memory import ExecutionMemory
from aec_agent.core.progress_evaluator import ProgressEvaluator, ProgressAssessment
from aec_agent.core.validator import ResultValidator
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus, ExecutionResult
from tests.reactive_agent.fixtures.reactive_fixtures import (
    create_test_task, create_test_execution_result, mock_llm
)


def fix_validation_level(validation_result: Dict[str, Any]) -> Dict[str, Any]:
    """Fix validation_level to be compatible with ExecutionMemory."""
    if validation_result.get("validation_level") == "status":
        validation_result["validation_level"] = "basic"
    return validation_result


class TestEndToEndReactiveLoops:
    """Test complete reactive reasoning workflows."""
    
    def test_successful_door_counting_workflow(self):
        """Test complete successful door counting workflow without replanning."""
        # Initialize components  
        execution_memory = ExecutionMemory("e2e-session-1", "Count all doors in building")
        progress_evaluator = ProgressEvaluator()
        # No validator - we'll create validation results manually
        
        # === ITERATION 1: Load Building Data ===
        execution_memory.start_iteration(1)
        
        # Execute task
        load_task = create_test_task("Load building data", "Load IFC building data", "task-1")
        load_result = create_test_execution_result(
            "load_building_data",
            success=True,
            output={
                "status": "success",
                "elements": 150,
                "building_name": "Test Building",
                "building_loaded": True
            }
        )
        
        # Record execution
        load_step_id = execution_memory.record_execution_step(
            task_name=load_task.name,
            task_id=load_task.id,
            tool_name=load_result.tool_name,
            tool_arguments={"file_path": "building.ifc"},
            tool_output=load_result.output,
            tool_success=load_result.success
        )
        
        # Create successful validation result manually  
        validation = {
            "success": True,
            "message": "Load building data successful",
            "validation_level": "logical",
            "method": "test_controlled",
            "confidence": 0.9,
            "suggests_replanning": False,
            "replanning_confidence": 0.0,
            "replanning_reason": "No replanning needed"
        }
        execution_memory.add_validation_result(load_step_id, validation)
        
        # Discover context
        execution_memory.discover_context("building_info", "building_loaded", True, "load_building_data")
        execution_memory.discover_context("building_info", "total_elements", 150, "load_building_data")
        
        # === ITERATION 2: Count Door Elements ===
        execution_memory.start_iteration(2)
        
        # Execute task
        count_task = create_test_task("Count doors", "Count door elements", "task-2")
        count_result = create_test_execution_result(
            "get_all_elements",
            success=True,
            output={
                "status": "success",
                "elements": [
                    {"id": "door1", "type": "IfcDoor"},
                    {"id": "door2", "type": "IfcDoor"},
                    {"id": "door3", "type": "IfcDoor"}
                ],
                "count": 3,
                "element_type": "IfcDoor"
            }
        )
        
        # Record execution
        count_step_id = execution_memory.record_execution_step(
            task_name=count_task.name,
            task_id=count_task.id,
            tool_name=count_result.tool_name,
            tool_arguments={"filter_type": "IfcDoor"},
            tool_output=count_result.output,
            tool_success=count_result.success
        )
        
        # Create successful validation result manually
        validation = {
            "success": True,
            "message": "Count doors successful", 
            "validation_level": "logical",
            "method": "test_controlled",
            "confidence": 0.9,
            "suggests_replanning": False,
            "replanning_confidence": 0.0,
            "replanning_reason": "No replanning needed"
        }
        execution_memory.add_validation_result(count_step_id, validation)
        
        # Discover context with required evidence for counting tasks
        execution_memory.discover_context("building_elements", "door_count", 3, "get_all_elements")
        execution_memory.discover_context("results", "numeric_result", 3, "get_all_elements") 
        execution_memory.discover_context("results", "element_list", ["door1", "door2", "door3"], "get_all_elements")
        
        # === EVALUATE GOAL PROGRESS ===
        execution_context = execution_memory.get_context_for_progress_evaluation()
        current_tasks = [load_task, count_task]
        
        # Mark tasks as completed
        load_task.status = TaskStatus.COMPLETED
        count_task.status = TaskStatus.COMPLETED
        
        # Evaluate progress
        assessment = progress_evaluator.evaluate_progress(
            original_goal="Count all doors in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        # Record goal achievement
        execution_memory.assess_goal_achievement({
            "goal_achieved": assessment.goal_achieved,
            "confidence": assessment.confidence,
            "completion_percentage": assessment.completion_percentage,
            "evidence": assessment.evidence_for_completion,
            "missing_requirements": assessment.missing_requirements
        })
        
        # === VERIFY SUCCESSFUL WORKFLOW ===
        assert assessment.goal_achieved is True, f"Goal should be achieved. Assessment: {assessment}"
        assert assessment.completion_percentage >= 80, "Should have high completion percentage"
        assert assessment.confidence >= 0.7, "Should have good confidence"
        assert len(assessment.missing_requirements) == 0, "No missing requirements for completed goal"
        
        # Verify execution memory state
        assert execution_memory.current_iteration == 2
        assert len(execution_memory.execution_steps) == 2
        assert len(execution_memory.discovered_context) >= 4
        assert execution_memory.is_goal_achieved is True
        assert execution_memory.should_trigger_replanning() is False
        
        # Verify execution summary
        summary = execution_memory.get_execution_summary()
        assert summary["successful_steps"] == 2
        assert summary["failed_steps"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["goal_achieved"] is True
    
    def test_reactive_replanning_workflow(self):
        """Test reactive workflow with failure and replanning."""
        # Initialize components
        execution_memory = ExecutionMemory("e2e-session-2", "Count doors in building")
        progress_evaluator = ProgressEvaluator()
        validator = ResultValidator()
        
        # === ITERATION 1: Load Building Data (Success) ===
        execution_memory.start_iteration(1)
        
        load_task = create_test_task("Load building data", "Load IFC building data", "task-1")
        load_result = create_test_execution_result(
            "load_building_data",
            success=True,
            output={"elements": 150, "building_loaded": True}
        )
        
        load_step_id = execution_memory.record_execution_step(
            task_name=load_task.name,
            task_id=load_task.id,
            tool_name=load_result.tool_name,
            tool_arguments={"file_path": "building.ifc"},
            tool_success=True,
            tool_output=load_result.output
        )
        
        validation = validator.validate_execution(load_task, load_result)
        execution_memory.add_validation_result(load_step_id, fix_validation_level(validation))
        execution_memory.discover_context("building_info", "building_loaded", True, "load_building_data")
        
        # === ITERATION 2: Query Doors (Failure) ===
        execution_memory.start_iteration(2)
        
        query_task = create_test_task("Query doors", "Query door elements", "task-2")
        query_result = create_test_execution_result(
            "query_elements",
            success=False,
            output=None
        )
        query_result.error_message = "Query failed: invalid parameters"
        
        query_step_id = execution_memory.record_execution_step(
            task_name=query_task.name,
            task_id=query_task.id,
            tool_name=query_result.tool_name,
            tool_arguments={"element_type": "doors"},
            tool_success=False,
            error_message=query_result.error_message
        )
        
        # Get execution context for validation
        execution_context = execution_memory.get_context_for_replanning()
        
        # Validate with replanning assessment
        validation = validator.validate_with_replanning_assessment(
            query_task, query_result, execution_context
        )
        execution_memory.add_validation_result(query_step_id, fix_validation_level(validation))
        
        # === CHECK IF REPLANNING IS TRIGGERED ===
        should_replan = execution_memory.should_trigger_replanning()
        assert should_replan is True, "Repeated failures should trigger replanning"
        
        # === RECORD PLAN MODIFICATION ===
        plan_mod_id = execution_memory.record_plan_modification(
            trigger_type="validation_failure",
            trigger_details={
                "failed_task": query_task.name,
                "tool": query_result.tool_name,
                "error": query_result.error_message
            },
            original_plan_summary="Use query_elements to get doors",
            new_plan_summary="Use get_all_elements and filter for doors",
            modification_reasoning="query_elements failed, switching to alternative tool",
            tasks_added=["get_all_elements_task"],
            tasks_removed=["query_elements_task"]
        )
        
        # === ITERATION 3: Alternative Approach (Success) ===
        execution_memory.start_iteration(3)
        
        alt_task = create_test_task("Get all elements", "Get all elements and filter", "task-3")
        alt_result = create_test_execution_result(
            "get_all_elements",
            success=True,
            output={
                "doors": [{"id": "door1"}, {"id": "door2"}],
                "count": 2,
                "element_type": "IfcDoor"
            }
        )
        
        alt_step_id = execution_memory.record_execution_step(
            task_name=alt_task.name,
            task_id=alt_task.id,
            tool_name=alt_result.tool_name,
            tool_arguments={"filter_type": "IfcDoor"},
            tool_success=True,
            tool_output=alt_result.output
        )
        
        validation = validator.validate_execution(alt_task, alt_result)
        execution_memory.add_validation_result(alt_step_id, fix_validation_level(validation))
        
        # Add required evidence for counting tasks
        execution_memory.discover_context("building_elements", "door_count", 2, "get_all_elements")
        execution_memory.discover_context("results", "numeric_result", 2, "get_all_elements")
        execution_memory.discover_context("results", "element_list", ["door1", "door2"], "get_all_elements")
        
        # === FINAL PROGRESS EVALUATION ===
        execution_context = execution_memory.get_context_for_replanning()
        current_tasks = [load_task, alt_task]  # Skip failed task
        
        load_task.status = TaskStatus.COMPLETED
        alt_task.status = TaskStatus.COMPLETED
        
        assessment = progress_evaluator.evaluate_progress(
            original_goal="Count doors in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        execution_memory.assess_goal_achievement({
            "goal_achieved": assessment.goal_achieved,
            "confidence": assessment.confidence,
            "completion_percentage": assessment.completion_percentage,
            "evidence": assessment.evidence_for_completion,
            "missing_requirements": assessment.missing_requirements
        })
        
        # === VERIFY REACTIVE WORKFLOW ===
        assert assessment.goal_achieved is True, "Goal should be achieved after replanning"
        assert assessment.confidence >= 0.7, "Should have good confidence"
        
        # Verify execution memory shows reactive behavior
        assert execution_memory.current_iteration == 3
        assert len(execution_memory.execution_steps) == 3
        assert len(execution_memory.plan_modifications) == 1
        assert execution_memory.is_goal_achieved is True
        
        # Verify replanning was effective
        summary = execution_memory.get_execution_summary()
        assert summary["plan_modifications"] == 1
        assert summary["successful_steps"] == 2  # Load + Alternative
        assert summary["failed_steps"] == 1     # Failed query
        assert summary["success_rate"] == 2/3   # 2 out of 3 successful
        assert summary["goal_achieved"] is True
    
    def test_multi_iteration_progress_tracking(self):
        """Test progress tracking across multiple iterations."""
        # Initialize components
        execution_memory = ExecutionMemory("e2e-session-3", "Complete building analysis")
        progress_evaluator = ProgressEvaluator()
        validator = ResultValidator()
        
        phases = [
            {
                "iteration": 1,
                "task": create_test_task("Load data", "Load building data", "task-1"),
                "tool": "load_building_data",
                "success": True,
                "output": {"building_loaded": True, "elements": 200},
                "evidence": [("building_info", "building_loaded", True)]
            },
            {
                "iteration": 2,
                "task": create_test_task("Extract doors", "Extract door elements", "task-2"),
                "tool": "extract_elements",
                "success": True,
                "output": {"doors": [{"id": "d1"}, {"id": "d2"}], "count": 2},
                "evidence": [("elements", "doors_extracted", True), ("counts", "door_count", 2)]
            },
            {
                "iteration": 3,
                "task": create_test_task("Extract windows", "Extract window elements", "task-3"),
                "tool": "extract_elements",
                "success": True,
                "output": {"windows": [{"id": "w1"}], "count": 1},
                "evidence": [("elements", "windows_extracted", True), ("counts", "window_count", 1)]
            },
            {
                "iteration": 4,
                "task": create_test_task("Calculate totals", "Calculate total elements", "task-4"),
                "tool": "calculate_totals",
                "success": True,
                "output": {"total_openings": 3, "doors": 2, "windows": 1},
                "evidence": [
                    ("results", "numeric_result", 3),
                    ("results", "calculation_output", {"doors": 2, "windows": 1})
                ]
            }
        ]
        
        assessments = []
        
        # Execute each phase and track progress
        for phase in phases:
            execution_memory.start_iteration(phase["iteration"])
            
            # Execute task
            result = create_test_execution_result(
                phase["tool"],
                success=phase["success"],
                output=phase["output"]
            )
            
            step_id = execution_memory.record_execution_step(
                task_name=phase["task"].name,
                task_id=phase["task"].id,
                tool_name=result.tool_name,
                tool_arguments={},
                tool_success=result.success,
                tool_output=result.output
            )
            
            # Validate
            validation = validator.validate_execution(phase["task"], result)
            execution_memory.add_validation_result(step_id, fix_validation_level(validation))
            
            # Add evidence
            for evidence_type, key, value in phase["evidence"]:
                execution_memory.discover_context(evidence_type, key, value, phase["tool"])
            
            # Evaluate progress
            execution_context = execution_memory.get_context_for_replanning()
            completed_tasks = [p["task"] for p in phases[:phase["iteration"]]]
            for task in completed_tasks:
                task.status = TaskStatus.COMPLETED
            
            assessment = progress_evaluator.evaluate_progress(
                original_goal="Complete building analysis",
                execution_context=execution_context,
                current_tasks=completed_tasks
            )
            assessments.append(assessment)
        
        # === VERIFY PROGRESSIVE IMPROVEMENT ===
        
        # Should show increasing completion over iterations
        completion_percentages = [a.completion_percentage for a in assessments]
        assert completion_percentages == sorted(completion_percentages), "Completion should increase over time"
        assert completion_percentages[-1] >= 80, "Final completion should be high"
        
        # Final assessment should show goal achieved
        final_assessment = assessments[-1]
        assert final_assessment.goal_achieved is True, "Goal should be achieved after all phases"
        assert final_assessment.confidence >= 0.7, "Final confidence should be high"
        
        # Verify execution memory tracks all iterations
        assert execution_memory.current_iteration == 4
        assert len(execution_memory.execution_steps) == 4
        assert len(execution_memory.discovered_context) >= 8  # Evidence from all phases
        
        # All steps should be successful
        summary = execution_memory.get_execution_summary()
        assert summary["successful_steps"] == 4
        assert summary["failed_steps"] == 0
        assert summary["success_rate"] == 1.0
    
    def test_complex_reactive_scenario_with_multiple_replanning(self):
        """Test complex scenario with multiple failures and replanning events."""
        # Initialize components
        execution_memory = ExecutionMemory("e2e-session-4", "Analyze building compliance")
        progress_evaluator = ProgressEvaluator()
        validator = ResultValidator()
        
        # === ITERATION 1: Load Building (Success) ===
        execution_memory.start_iteration(1)
        
        load_task = create_test_task("Load building", "Load building data", "task-1")
        load_result = create_test_execution_result(
            "load_building_data", True, {"building_loaded": True}
        )
        
        load_step_id = execution_memory.record_execution_step(
            load_task.name, load_task.id, "load_building_data", {}, load_result.output, True
        )
        validation = validator.validate_execution(load_task, load_result)
        execution_memory.add_validation_result(load_step_id, fix_validation_level(validation))
        execution_memory.discover_context("building", "loaded", True, "load_building_data")
        
        # === ITERATION 2: First Compliance Check (Failure) ===
        execution_memory.start_iteration(2)
        
        check1_task = create_test_task("Check compliance", "Check building compliance", "task-2")
        check1_result = create_test_execution_result("compliance_checker", False, None)
        check1_result.error_message = "Compliance data unavailable"
        
        check1_step_id = execution_memory.record_execution_step(
            check1_task.name, check1_task.id, "compliance_checker", {}, tool_output=None, tool_success=False, error_message=check1_result.error_message
        )
        
        execution_context = execution_memory.get_context_for_replanning()
        validation = validator.validate_with_replanning_assessment(check1_task, check1_result, execution_context)
        execution_memory.add_validation_result(check1_step_id, fix_validation_level(validation))
        
        # Record first plan modification
        execution_memory.record_plan_modification(
            "execution_failure", {"tool": "compliance_checker"}, 
            "Use compliance_checker", "Use building_analyzer first", 
            "compliance_checker failed, need to analyze building first"
        )
        
        # === ITERATION 3: Analyze Building (Low Confidence) ===
        execution_memory.start_iteration(3)
        
        analyze_task = create_test_task("Analyze building", "Analyze building structure", "task-3")
        analyze_result = create_test_execution_result(
            "building_analyzer", True, {"analysis": "preliminary", "confidence": 0.2}
        )
        
        analyze_step_id = execution_memory.record_execution_step(
            analyze_task.name, analyze_task.id, "building_analyzer", {}, analyze_result.output, True
        )
        
        # Simulate low confidence validation
        low_confidence_validation = {
            "success": True,
            "confidence": 0.2,
            "validation_level": "logical",
            "method": "llm_reasoning",
            "issues": ["Low confidence result", "Incomplete analysis"]
        }
        execution_memory.add_validation_result(analyze_step_id, fix_validation_level(low_confidence_validation))
        
        # Check if low confidence triggers replanning
        should_replan_2 = execution_memory.should_trigger_replanning()
        if should_replan_2:
            execution_memory.record_plan_modification(
                "low_confidence", {"confidence": 0.2},
                "Use basic building_analyzer", "Use detailed_analyzer with more parameters",
                "Low confidence result, need more detailed analysis"
            )
        
        # === ITERATION 4: Detailed Analysis (Success) ===
        execution_memory.start_iteration(4)
        
        detailed_task = create_test_task("Detailed analysis", "Detailed building analysis", "task-4")
        detailed_result = create_test_execution_result(
            "detailed_analyzer", True, {
                "analysis": "complete",
                "compliance_data": {"rules": ["r1", "r2"], "violations": []},
                "confidence": 0.9
            }
        )
        
        detailed_step_id = execution_memory.record_execution_step(
            detailed_task.name, detailed_task.id, "detailed_analyzer", {}, detailed_result.output, True
        )
        
        validation = validator.validate_execution(detailed_task, detailed_result)
        execution_memory.add_validation_result(detailed_step_id, fix_validation_level(validation))
        
        # Add evidence for compliance analysis
        execution_memory.discover_context("compliance", "validation_result", "passed", "detailed_analyzer")
        execution_memory.discover_context("compliance", "rule_check", "compliant", "detailed_analyzer")
        execution_memory.discover_context("analysis", "confidence", 0.9, "detailed_analyzer")
        
        # === FINAL EVALUATION ===
        execution_context = execution_memory.get_context_for_replanning()
        successful_tasks = [load_task, detailed_task]
        for task in successful_tasks:
            task.status = TaskStatus.COMPLETED
        
        assessment = progress_evaluator.evaluate_progress(
            original_goal="Analyze building compliance",
            execution_context=execution_context,
            current_tasks=successful_tasks
        )
        
        execution_memory.assess_goal_achievement({
            "goal_achieved": assessment.goal_achieved,
            "confidence": assessment.confidence,
            "completion_percentage": assessment.completion_percentage,
            "evidence": assessment.evidence_for_completion,
            "missing_requirements": assessment.missing_requirements
        })
        
        # === VERIFY COMPLEX REACTIVE WORKFLOW ===
        assert assessment.goal_achieved is True, "Goal should be achieved after multiple adaptations"
        assert assessment.confidence >= 0.7, "Should have good final confidence"
        
        # Verify reactive adaptations
        assert execution_memory.current_iteration == 4
        assert len(execution_memory.execution_steps) == 4
        assert len(execution_memory.plan_modifications) >= 1  # At least one replanning event
        assert execution_memory.is_goal_achieved is True
        
        summary = execution_memory.get_execution_summary()
        assert summary["plan_modifications"] >= 1
        assert summary["successful_steps"] == 2  # Load + Detailed analysis
        assert summary["failed_steps"] == 1     # Failed compliance check
        assert summary["goal_achieved"] is True
        
        # Verify that replanning was triggered and effective
        replanning_context = execution_memory.get_context_for_replanning()
        assert len(replanning_context["recent_execution_steps"]) >= 4
        assert len(replanning_context["recent_failures"]) >= 1


class TestReactiveLoopsEdgeCases:
    """Test edge cases in reactive reasoning loops."""
    
    def test_early_goal_achievement(self):
        """Test scenario where goal is achieved earlier than expected."""
        execution_memory = ExecutionMemory("early-goal", "Load building data")
        progress_evaluator = ProgressEvaluator()
        validator = ResultValidator()
        
        # Single iteration achieves the goal
        execution_memory.start_iteration(1)
        
        load_task = create_test_task("Load building", "Load building data", "task-1")
        load_result = create_test_execution_result(
            "load_building_data", True, {
                "building_loaded": True,
                "data_output": {"elements": 100},
                "element_properties": {"doors": 5}
            }
        )
        
        step_id = execution_memory.record_execution_step(
            load_task.name, load_task.id, "load_building_data", {}, load_result.output, True
        )
        
        validation = validator.validate_execution(load_task, load_result)
        execution_memory.add_validation_result(step_id, fix_validation_level(validation))
        
        # Add evidence for data extraction goal
        execution_memory.discover_context("loading", "data_output", True, "load_building_data")
        execution_memory.discover_context("loading", "element_properties", True, "load_building_data")
        
        # Evaluate progress
        execution_context = execution_memory.get_context_for_replanning()
        load_task.status = TaskStatus.COMPLETED
        
        assessment = progress_evaluator.evaluate_progress(
            original_goal="Load building data from file",  # Data extraction goal
            execution_context=execution_context,
            current_tasks=[load_task]
        )
        
        # Should achieve goal in single iteration
        assert assessment.goal_achieved is True, "Simple loading goal should be achieved quickly"
        assert assessment.completion_percentage == 100.0
        assert len(assessment.missing_requirements) == 0
    
    def test_persistent_failures_scenario(self):
        """Test scenario with persistent failures that eventually succeed."""
        execution_memory = ExecutionMemory("persistent-fail", "Count doors")
        progress_evaluator = ProgressEvaluator()
        validator = ResultValidator()
        
        # Multiple failed attempts
        failed_tools = ["query_elements", "door_finder", "element_counter"]
        
        for i, tool in enumerate(failed_tools):
            execution_memory.start_iteration(i + 1)
            
            fail_task = create_test_task(f"Attempt {i+1}", f"Try {tool}", f"task-{i+1}")
            fail_result = create_test_execution_result(tool, False, None)
            fail_result.error_message = f"{tool} failed"
            
            step_id = execution_memory.record_execution_step(
                fail_task.name, fail_task.id, tool, {}, tool_output=None, tool_success=False, error_message=fail_result.error_message
            )
            
            execution_context = execution_memory.get_context_for_replanning()
            validation = validator.validate_with_replanning_assessment(fail_task, fail_result, execution_context)
            execution_memory.add_validation_result(step_id, fix_validation_level(validation))
            
            # Record plan modification for each failure
            execution_memory.record_plan_modification(
                "execution_failure", {"tool": tool},
                f"Use {tool}", f"Try alternative approach",
                f"{tool} failed, trying next alternative"
            )
        
        # Final successful attempt
        execution_memory.start_iteration(len(failed_tools) + 1)
        
        success_task = create_test_task("Final attempt", "Use working tool", "task-final")
        success_result = create_test_execution_result(
            "get_all_elements", True, {
                "doors": [{"id": "d1"}], 
                "count": 1
            }
        )
        
        step_id = execution_memory.record_execution_step(
            success_task.name, success_task.id, "get_all_elements", {}, success_result.output, True
        )
        
        validation = validator.validate_execution(success_task, success_result)
        execution_memory.add_validation_result(step_id, fix_validation_level(validation))
        
        # Add required evidence
        execution_memory.discover_context("results", "numeric_result", 1, "get_all_elements")
        execution_memory.discover_context("results", "element_list", ["d1"], "get_all_elements")
        
        # Evaluate final progress
        execution_context = execution_memory.get_context_for_replanning()
        success_task.status = TaskStatus.COMPLETED
        
        assessment = progress_evaluator.evaluate_progress(
            original_goal="Count doors",
            execution_context=execution_context,
            current_tasks=[success_task]
        )
        
        # Should eventually succeed despite multiple failures
        assert assessment.goal_achieved is True, "Should succeed after persistent failures"
        
        # Verify reactive behavior
        summary = execution_memory.get_execution_summary()
        assert summary["plan_modifications"] == len(failed_tools)
        assert summary["failed_steps"] == len(failed_tools)
        assert summary["successful_steps"] == 1
        assert summary["goal_achieved"] is True
    
    def test_mixed_confidence_convergence(self):
        """Test workflow with mixed confidence levels that converge to success."""
        execution_memory = ExecutionMemory("mixed-confidence", "Building analysis")
        progress_evaluator = ProgressEvaluator()
        validator = ResultValidator()
        
        # Series of tasks with varying confidence levels
        tasks_with_confidence = [
            ("Initial scan", 0.3, {"scan": "basic"}),
            ("Detailed scan", 0.6, {"scan": "detailed", "elements": 50}),
            ("Full analysis", 0.9, {"analysis": "complete", "elements": 100, "confidence": 0.9})
        ]
        
        for i, (task_name, conf, output) in enumerate(tasks_with_confidence):
            execution_memory.start_iteration(i + 1)
            
            task = create_test_task(task_name, f"Perform {task_name}", f"task-{i+1}")
            result = create_test_execution_result("analyzer", True, output)
            
            step_id = execution_memory.record_execution_step(
                task.name, task.id, "analyzer", {}, result.output, True
            )
            
            # Simulate validation with varying confidence
            validation = {
                "success": True,
                "confidence": conf,
                "validation_level": "llm_intelligent",
                "method": "llm_reasoning",
                "issues": [] if conf > 0.7 else ["Low confidence"]
            }
            execution_memory.add_validation_result(step_id, fix_validation_level(validation))
            
            # Add progressive evidence
            execution_memory.discover_context("progress", f"step_{i+1}_complete", True, "analyzer")
            if conf >= 0.9:  # Final step adds required evidence
                execution_memory.discover_context("results", "data_output", output, "analyzer")
                execution_memory.discover_context("results", "element_properties", {"count": 100}, "analyzer")
        
        # Evaluate final progress
        execution_context = execution_memory.get_context_for_replanning()
        all_tasks = [
            create_test_task(name, f"Perform {name}", f"task-{i+1}") 
            for i, (name, _, _) in enumerate(tasks_with_confidence)
        ]
        for task in all_tasks:
            task.status = TaskStatus.COMPLETED
        
        assessment = progress_evaluator.evaluate_progress(
            original_goal="Extract building data and analyze structure",  # Data extraction goal
            execution_context=execution_context,
            current_tasks=all_tasks
        )
        
        # Should converge to success despite low initial confidence
        assert assessment.goal_achieved is True, "Should achieve goal despite initial low confidence"
        assert assessment.confidence >= 0.8, "Final confidence should be high"
        
        # Verify confidence progression
        summary = execution_memory.get_execution_summary()
        assert summary["average_validation_confidence"] >= 0.6  # Average should be reasonable
        assert summary["successful_steps"] == 3
        assert summary["failed_steps"] == 0