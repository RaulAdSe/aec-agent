"""
Reasoning Controller - Main orchestrator for the autonomous reasoning system.

The controller coordinates the reasoning loop: ANALYZE → PLAN → EXECUTE → VALIDATE → ITERATE
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .utils import ReasoningUtils, Task, TaskStatus, ExecutionResult
from .goal_decomposer import GoalDecomposer
from .tool_planner import ToolPlanner
from .executor import ToolExecutor
from .validator import ResultValidator


@dataclass
class ReasoningState:
    """Current state of the reasoning process."""
    goal: str
    tasks: List[Task]
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
        max_iterations: int = 20,
        max_execution_time: float = 300.0  # 5 minutes
    ):
        """Initialize the reasoning controller."""
        self.goal_decomposer = goal_decomposer
        self.tool_planner = tool_planner
        self.executor = executor
        self.validator = validator
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        
        self.logger = ReasoningUtils.setup_logger(__name__)
        self.state: Optional[ReasoningState] = None
    
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
        
        # Initialize reasoning state
        self.state = ReasoningState(goal=goal, tasks=[])
        
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
    
    def _analyze_and_plan(self, goal: str, context: Dict[str, Any]) -> None:
        """Analyze the goal and create execution plan."""
        self.logger.info("Phase 1: Analyzing goal and creating plan")
        
        # Decompose goal into tasks
        decomposition_result = self.goal_decomposer.decompose(goal, context)
        if not decomposition_result.get("success", False):
            raise ValueError(f"Goal decomposition failed: {decomposition_result.get('message')}")
        
        self.state.tasks = decomposition_result["tasks"]
        self.logger.info(f"Decomposed goal into {len(self.state.tasks)} tasks")
        
        # Plan tool execution for each task
        for task in self.state.tasks:
            planning_result = self.tool_planner.plan_tools(task, context)
            if planning_result.get("success", False):
                task.tool_sequence = planning_result["tool_sequence"]
                task.metadata.update(planning_result.get("metadata", {}))
            else:
                self.logger.warning(f"Tool planning failed for task {task.name}: {planning_result.get('message')}")
        
        self.logger.info(f"Task breakdown: {ReasoningUtils.format_task_summary(self.state.tasks)}")
    
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
            
            # Find next ready task
            ready_tasks = ReasoningUtils.find_ready_tasks(self.state.tasks)
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
            current_task.status = TaskStatus.IN_PROGRESS
            
            self.logger.info(f"Executing task: {current_task.name}")
            
            try:
                # Execute tools for this task
                execution_result = self._execute_task(current_task)
                all_results.append(execution_result)
                
                # Validate results
                validation_result = self.validator.validate_execution(
                    current_task, execution_result
                )
                
                if validation_result.get("success", False):
                    current_task.status = TaskStatus.COMPLETED
                    self.state.completed_tasks += 1
                    self.logger.info(f"Task completed successfully: {current_task.name}")
                else:
                    current_task.status = TaskStatus.FAILED
                    self.state.failed_tasks += 1
                    self.logger.warning(f"Task validation failed: {validation_result.get('message')}")
                    
                    # Try to recover or adapt
                    if not self._handle_task_failure(current_task, validation_result):
                        self.logger.error(f"Could not recover from task failure: {current_task.name}")
                
            except Exception as e:
                current_task.status = TaskStatus.FAILED
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
            progress = ReasoningUtils.calculate_task_progress(self.state.tasks)
            self.logger.info(f"Progress: {progress:.1f}% ({self.state.completed_tasks}/{len(self.state.tasks)} tasks)")
        
        return all_results
    
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
        
        return self.executor.execute_tool(
            tool_name=tool_name,
            task=task,
            context={"goal": self.state.goal, "iteration": self.state.iteration}
        )
    
    def _is_goal_achieved(self) -> bool:
        """Check if the main goal has been achieved."""
        if not self.state.tasks:
            return False
        
        # Goal is achieved when all tasks are completed
        completed_tasks = [
            task for task in self.state.tasks 
            if task.status == TaskStatus.COMPLETED
        ]
        
        return len(completed_tasks) == len(self.state.tasks)
    
    def _handle_blocked_tasks(self) -> bool:
        """Try to unblock tasks that are waiting for dependencies."""
        blocked_tasks = [
            task for task in self.state.tasks 
            if task.status == TaskStatus.BLOCKED
        ]
        
        if not blocked_tasks:
            return False
        
        # For now, just convert blocked to pending and retry
        # TODO: More sophisticated dependency resolution
        for task in blocked_tasks[:1]:  # Try one at a time
            task.status = TaskStatus.PENDING
            self.logger.info(f"Retrying blocked task: {task.name}")
            return True
        
        return False
    
    def _handle_task_failure(self, task: Task, validation_result: Dict[str, Any]) -> bool:
        """Try to handle task failure and recover."""
        # For now, just mark as failed
        # TODO: Implement retry strategies, alternative approaches
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
        
        # Create summary
        summary = {
            "goal_achieved": success,
            "total_tasks": len(self.state.tasks),
            "completed_tasks": self.state.completed_tasks,
            "failed_tasks": self.state.failed_tasks,
            "iterations": self.state.iteration,
            "execution_time": self.state.total_execution_time,
            "progress_percentage": ReasoningUtils.calculate_task_progress(self.state.tasks)
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
                for task in self.state.tasks
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