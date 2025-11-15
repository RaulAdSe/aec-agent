"""
Reasoning Controller - Main orchestrator for the autonomous reasoning system.

The controller coordinates the reasoning loop: ANALYZE → PLAN → EXECUTE → VALIDATE → ITERATE
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .reasoning_utils import ReasoningUtils, Task, TaskStatus, ExecutionResult
from .goal_decomposer import GoalDecomposer
from .tool_planner import ToolPlanner
from .executor import ToolExecutor
from .validator import ResultValidator
from .llm_guardrails import GuardrailConfig, ExecutionGuardrail, GuardrailViolationError
from .task_graph import TaskGraph
from ..config import AgentConfig

# Import LangSmith tracing
from langsmith import traceable


@dataclass
class ReasoningState:
    """Current state of the reasoning process."""
    goal: str
    task_graph: TaskGraph
    context: Dict[str, Any] = field(default_factory=dict)
    current_task_id: Optional[str] = None
    iteration: int = 0
    total_execution_time: float = 0.0
    completed_tasks: int = 0
    failed_tasks: int = 0


class ReasoningController:
    """
    Main orchestrator for autonomous reasoning.
    
    Coordinates the reasoning loop:
    1. ANALYZE: Understand the goal and current context
    2. PLAN: Decompose goal into tasks and plan tool execution
    3. EXECUTE: Run tools with monitoring
    4. VALIDATE: Check results and progress
    5. ITERATE: Adapt plan if needed and continue
    """
    
    def __init__(
        self,
        goal_decomposer: GoalDecomposer,
        tool_planner: ToolPlanner,
        executor: ToolExecutor,
        validator: ResultValidator,
        config: Optional[AgentConfig] = None,
        # Legacy parameters for backward compatibility
        max_iterations: Optional[int] = None,
        max_execution_time: Optional[float] = None,
        llm: Optional[Any] = None,
        guardrail_config: Optional[GuardrailConfig] = None
    ):
        """Initialize the reasoning controller."""
        self.goal_decomposer = goal_decomposer
        self.tool_planner = tool_planner
        self.executor = executor
        self.validator = validator
        
        # Initialize configuration
        self.config = config or AgentConfig.from_env()
        
        # Apply legacy parameter overrides if provided
        if max_iterations is not None:
            self.config.reasoning.max_iterations = max_iterations
        if max_execution_time is not None:
            self.config.reasoning.max_execution_time = max_execution_time
        
        # Store commonly used values for convenience
        self.max_iterations = self.config.reasoning.max_iterations
        self.max_execution_time = self.config.reasoning.max_execution_time
        
        # Initialize execution guardrails
        if guardrail_config is not None:
            # Use legacy guardrail config if provided
            self.execution_guardrail = ExecutionGuardrail(guardrail_config)
        else:
            # Use guardrail config from unified configuration
            effective_guardrail_config = self.config.get_effective_guardrail_config()
            self.execution_guardrail = ExecutionGuardrail(effective_guardrail_config)
        
        self.logger = ReasoningUtils.setup_logger(__name__)
        self.state: Optional[ReasoningState] = None
    
    @traceable(name="autonomous_reasoning_process", metadata={"component": "reasoning_controller"})
    def reason(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete reasoning process for a goal.
        
        Args:
            goal: High-level goal to achieve (e.g., "Analyze fire safety compliance")
            context: Optional context information (session state, loaded data, etc.)
            
        Returns:
            Dict containing execution results, final state, and detailed logs
        """
        start_time = time.time()
        self.logger.info(f"Starting reasoning for goal: {goal}")
        
        # Log inputs for tracing
        self.logger.info(f"Reasoning inputs - Goal: {goal}, Context keys: {list(context.keys()) if context else []}")
        
        # Initialize reasoning state and reset guardrails
        self.state = ReasoningState(goal=goal, task_graph=TaskGraph(), context=context or {})
        self.execution_guardrail.reset()
        self.logger.info("Execution guardrails reset for new reasoning session")
        
        try:
            # PHASE 1: ANALYZE & PLAN
            self._analyze_and_plan(goal, context or {})
            
            # PHASE 2: EXECUTE
            execution_results = self._execute_reasoning_loop()
            
            # PHASE 3: FINALIZE
            final_results = self._finalize_results(execution_results)
            
            end_time = time.time()
            self.state.total_execution_time = end_time - start_time
            
            self.logger.info(f"Reasoning completed in {self.state.total_execution_time:.2f}s")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Reasoning failed: {e}")
            return {
                "status": "error",
                "message": f"Reasoning failed: {str(e)}",
                "error": ReasoningUtils.extract_error_info(e),
                "state": self.state,
                "execution_time": time.time() - start_time
            }
    
    @traceable(name="analyze_and_plan_phase")
    def _analyze_and_plan(self, goal: str, context: Dict[str, Any]) -> None:
        """Analyze the goal and create execution plan."""
        self.logger.info("Phase 1: Analyzing goal and creating plan")
        
        # Decompose goal into tasks
        decomposition_result = self.goal_decomposer.decompose(goal, context)
        if not decomposition_result.get("success", False):
            raise ValueError(f"Goal decomposition failed: {decomposition_result.get('message')}")
        
        tasks = decomposition_result["tasks"]
        
        # Add tasks to graph and validate dependencies
        successful_count, failed_task_ids = self.state.task_graph.add_tasks(tasks)
        self.logger.info(f"Added {successful_count}/{len(tasks)} tasks to task graph")
        
        if failed_task_ids:
            self.logger.warning(f"Failed to add tasks due to dependency issues: {failed_task_ids}")
        
        # Validate graph integrity
        issues = self.state.task_graph.validate_graph()
        if issues:
            self.logger.warning(f"Task graph validation issues: {issues}")
        
        # Log graph metrics
        metrics = self.state.task_graph.get_graph_metrics()
        self.logger.info(f"Task graph initialized: {metrics.total_tasks} tasks, {metrics.dependency_edges} dependencies")
        self.logger.info(f"Critical path length: {metrics.longest_path}")
        
        # Don't plan tools upfront - use just-in-time planning with full context
        # This allows each task to be planned with complete context from previously executed tasks
        ready_count = len(self.state.task_graph.get_ready_tasks())
        self.logger.info(f"Ready to execute: {ready_count} tasks")
    
    @traceable(name="execute_reasoning_loop")
    def _execute_reasoning_loop(self) -> List[ExecutionResult]:
        """Execute the main reasoning loop until completion or limits reached."""
        self.logger.info("Phase 2: Executing reasoning loop")
        
        all_results = []
        
        while (
            self.state.iteration < self.max_iterations and
            self.state.total_execution_time < self.max_execution_time and
            not self._is_goal_achieved()
        ):
            self.state.iteration += 1
            iteration_start = time.time()
            
            self.logger.info(f"Iteration {self.state.iteration}")
            
            try:
                # Check execution guardrails before proceeding
                self.execution_guardrail.record_execution_step()
            except GuardrailViolationError as e:
                self.logger.error(f"Execution guardrail violation: {e}")
                break
            
            # Find next ready task using task graph
            ready_tasks = self.state.task_graph.get_ready_tasks()
            if not ready_tasks:
                self.logger.info("No ready tasks - checking for blocked tasks")
                if self._handle_blocked_tasks():
                    continue
                else:
                    self.logger.info("No tasks can proceed - stopping")
                    break
            
            # Execute next task
            current_task = ready_tasks[0]
            self.state.current_task_id = current_task.id
            self.state.task_graph.update_task_status(current_task.id, TaskStatus.IN_PROGRESS)
            
            self.logger.info(f"Executing task: {current_task.name}")
            
            # Check task attempt guardrail
            try:
                self.execution_guardrail.record_task_attempt(current_task.id)
            except GuardrailViolationError as e:
                self.logger.error(f"Task attempt guardrail violation: {e}")
                self.state.task_graph.update_task_status(current_task.id, TaskStatus.FAILED)
                continue
            
            # JUST-IN-TIME PLANNING: Plan tools for this task with full context
            if not current_task.tool_sequence:
                self.logger.info(f"Planning tools for task '{current_task.name}' with current context")
                try:
                    planning_result = self.tool_planner.plan_tools(current_task, self.state.context)
                    if planning_result.get("success", False):
                        current_task.tool_sequence = planning_result["tool_sequence"]
                        current_task.metadata.update(planning_result.get("metadata", {}))
                        self.logger.info(f"Planned tools for '{current_task.name}': {current_task.tool_sequence}")
                    else:
                        self.logger.error(f"Tool planning failed for task '{current_task.name}': {planning_result.get('message')}")
                        self.state.task_graph.update_task_status(current_task.id, TaskStatus.FAILED)
                        continue
                except Exception as e:
                    self.logger.error(f"Error planning tools for task '{current_task.name}': {e}")
                    self.state.task_graph.update_task_status(current_task.id, TaskStatus.FAILED)
                    continue
            
            try:
                # Execute tools for this task
                execution_result = self._execute_task(current_task)
                all_results.append(execution_result)
                
                # Validate results
                validation_result = self.validator.validate_execution(
                    current_task, execution_result
                )
                
                if validation_result.get("success", False):
                    self.state.task_graph.update_task_status(current_task.id, TaskStatus.COMPLETED)
                    self.state.completed_tasks += 1
                    self.logger.info(f"Task completed successfully: {current_task.name}")
                else:
                    self.state.task_graph.update_task_status(current_task.id, TaskStatus.FAILED)
                    self.state.failed_tasks += 1
                    self.logger.warning(f"Task validation failed: {validation_result.get('message')}")
                    
                    # Try to recover or adapt
                    if not self._handle_task_failure(current_task, validation_result):
                        self.logger.error(f"Could not recover from task failure: {current_task.name}")
                
            except Exception as e:
                self.state.task_graph.update_task_status(current_task.id, TaskStatus.FAILED)
                self.state.failed_tasks += 1
                self.logger.error(f"Task execution failed: {current_task.name} - {e}")
                
                error_result = ExecutionResult(
                    success=False,
                    tool_name="task_execution",
                    output=None,
                    error_message=str(e)
                )
                all_results.append(error_result)
            
            # Update timing
            iteration_time = time.time() - iteration_start
            self.state.total_execution_time += iteration_time
            
            # Check if we should continue
            task_list = list(self.state.task_graph.tasks.values())
            progress = ReasoningUtils.calculate_task_progress(task_list)
            self.logger.info(f"Progress: {progress:.1f}% ({self.state.completed_tasks}/{len(task_list)} tasks)")
        
        return all_results
    
    @traceable(name="execute_single_task")
    def _execute_task(self, task: Task) -> ExecutionResult:
        """Execute a single task using its planned tool sequence."""
        if not task.tool_sequence:
            return ExecutionResult(
                success=False,
                tool_name="no_tools",
                output=None,
                error_message="No tools planned for task"
            )
        
        # For now, execute the first tool in the sequence
        # TODO: Support multi-tool sequences
        tool_name = task.tool_sequence[0]
        
        result = self.executor.execute_tool(
            tool_name=tool_name,
            task=task,
            context={
                "goal": self.state.goal, 
                "iteration": self.state.iteration,
                **self.state.context
            }
        )
        
        # Update context based on tool execution results
        if result.success:
            if tool_name == "load_building_data":
                self.state.context["building_data_loaded"] = True
                self.logger.info("Updated context: building_data_loaded = True")
                
                # No need to re-plan - just-in-time planning will handle this
                
            elif tool_name == "get_all_elements" and result.output:
                # Store element data in context for future tasks
                try:
                    # Extract element type from task description or metadata
                    element_type = None
                    if "door" in task.description.lower():
                        element_type = "doors"
                    elif "exit" in task.description.lower():
                        element_type = "exits"
                    elif "space" in task.description.lower():
                        element_type = "spaces"
                    elif "wall" in task.description.lower():
                        element_type = "walls"
                    
                    if element_type and result.output.get("data"):
                        self.state.context[f"{element_type}_data"] = result.output["data"]
                        self.logger.info(f"Updated context: {element_type}_data with {len(result.output['data'])} elements")
                        
                except Exception as e:
                    self.logger.warning(f"Could not update context from get_all_elements: {e}")
                    
            elif tool_name in ["calculate_distances", "calculate_areas"]:
                # Store calculation results in context if needed
                if result.output and result.output.get("data"):
                    calculation_key = f"last_{tool_name}_result"
                    self.state.context[calculation_key] = result.output["data"]
                    self.logger.info(f"Updated context: {calculation_key}")
                    
        self.logger.debug(f"Context after {tool_name}: {list(self.state.context.keys())}")
        
        return result
    
    def _is_goal_achieved(self) -> bool:
        """Check if the main goal has been achieved."""
        if not self.state.task_graph.tasks:
            return False
        
        # Goal is achieved when all tasks are completed
        completed_tasks = [
            task for task in self.state.task_graph.tasks.values()
            if task.status == TaskStatus.COMPLETED
        ]
        
        return len(completed_tasks) == len(self.state.task_graph.tasks)
    
    def _handle_blocked_tasks(self) -> bool:
        """Try to unblock tasks that are waiting for dependencies."""
        blocked_tasks = self.state.task_graph.get_blocked_tasks()
        
        if not blocked_tasks:
            return False
        
        # For now, just convert blocked to pending and retry
        # TODO: More sophisticated dependency resolution
        for task in blocked_tasks[:1]:  # Try one at a time
            self.state.task_graph.update_task_status(task.id, TaskStatus.PENDING)
            self.logger.info(f"Retrying blocked task: {task.name}")
            return True
        
        return False
    
    def _handle_task_failure(self, task: Task, validation_result: Dict[str, Any]) -> bool:
        """Try to handle task failure and recover."""
        # Check if replanning is recommended and allowed by guardrails
        if validation_result.get("should_replan", False):
            try:
                self.execution_guardrail.record_replanning_event()
                self.logger.info(f"Triggering replanning for failed task: {task.name}")
                # TODO: Implement actual replanning logic using Replanner component
                return True
            except GuardrailViolationError as e:
                self.logger.error(f"Replanning guardrail violation: {e}")
                return False
        
        # For now, just mark as failed without replanning
        self.logger.warning(f"Task failed without recovery: {task.name}")
        return False
    
    def _finalize_results(self, execution_results: List[ExecutionResult]) -> Dict[str, Any]:
        """Create final results summary."""
        success = self._is_goal_achieved()
        
        # Collect successful outputs
        outputs = []
        for result in execution_results:
            if result.success and result.output:
                outputs.append({
                    "tool": result.tool_name,
                    "output": result.output,
                    "execution_time": result.execution_time
                })
        
        # Create summary using task graph metrics
        graph_metrics = self.state.task_graph.get_graph_metrics()
        task_list = list(self.state.task_graph.tasks.values())
        
        summary = {
            "goal_achieved": success,
            "total_tasks": graph_metrics.total_tasks,
            "completed_tasks": graph_metrics.completed_tasks,
            "failed_tasks": graph_metrics.failed_tasks,
            "iterations": self.state.iteration,
            "execution_time": self.state.total_execution_time,
            "progress_percentage": ReasoningUtils.calculate_task_progress(task_list),
            "guardrails": self.execution_guardrail.get_status(),
            "graph_metrics": {
                "dependency_edges": graph_metrics.dependency_edges,
                "critical_path_length": graph_metrics.longest_path,
                "blocked_tasks": graph_metrics.blocked_tasks
            }
        }
        
        return {
            "status": "success" if success else "partial",
            "message": "Goal achieved" if success else "Goal partially achieved",
            "summary": summary,
            "outputs": outputs,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status.value,
                    "tools": task.tool_sequence
                }
                for task in self.state.task_graph.tasks.values()
            ],
            "execution_results": [
                {
                    "success": result.success,
                    "tool": result.tool_name,
                    "error": result.error_message,
                    "time": result.execution_time
                }
                for result in execution_results
            ]
        }