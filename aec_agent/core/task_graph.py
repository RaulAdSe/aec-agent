"""
Task Graph - Directed Acyclic Graph (DAG) for task dependency management.

Provides formal graph representation of task dependencies, enabling:
- Dependency validation and cycle detection
- Optimized task scheduling
- Clear visualization of execution flow
- Redundancy elimination
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass

from .reasoning_utils import Task, TaskStatus, Priority, ReasoningUtils


@dataclass 
class GraphMetrics:
    """Metrics about the task graph structure."""
    total_tasks: int
    ready_tasks: int
    completed_tasks: int
    failed_tasks: int
    blocked_tasks: int
    dependency_edges: int
    longest_path: int
    critical_path_tasks: List[str]


class TaskGraph:
    """
    Directed Acyclic Graph (DAG) for managing task dependencies.
    
    Provides formal graph operations for task orchestration:
    - Dependency validation and cycle detection
    - Topological ordering for execution planning
    - Critical path analysis
    - Graph visualization and debugging
    """
    
    def __init__(self, tasks: List[Task] = None):
        """Initialize task graph with optional initial tasks."""
        self.graph = nx.DiGraph()
        self.tasks: Dict[str, Task] = {}
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        if tasks:
            self.add_tasks(tasks)
            # Auto-mark optional tasks based on common patterns
            self.auto_mark_optional_tasks()
    
    def add_task(self, task: Task) -> bool:
        """
        Add a single task to the graph.
        
        Args:
            task: Task to add
            
        Returns:
            True if added successfully, False if would create cycle
        """
        # Add task to storage
        self.tasks[task.id] = task
        
        # Add node to graph
        self.graph.add_node(task.id, task=task)
        
        # Add dependency edges
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                self.logger.warning(f"Dependency {dep_id} not found for task {task.id}")
                continue
            
            # Add edge: dependency -> task
            self.graph.add_edge(dep_id, task.id)
        
        # Validate no cycles were created
        if not nx.is_directed_acyclic_graph(self.graph):
            self.logger.error(f"Adding task {task.id} would create cycle")
            # Remove the task and its edges
            self.graph.remove_node(task.id)
            del self.tasks[task.id]
            return False
        
        self.logger.debug(f"Added task {task.id} with {len(task.dependencies)} dependencies")
        return True
    
    def add_tasks(self, tasks: List[Task]) -> Tuple[int, List[str]]:
        """
        Add multiple tasks to the graph.
        
        Args:
            tasks: List of tasks to add
            
        Returns:
            Tuple of (successful_count, failed_task_ids)
        """
        successful = 0
        failed_ids = []
        
        for task in tasks:
            if self.add_task(task):
                successful += 1
            else:
                failed_ids.append(task.id)
        
        self.logger.info(f"Added {successful}/{len(tasks)} tasks to graph")
        if failed_ids:
            self.logger.warning(f"Failed to add tasks: {failed_ids}")
        
        return successful, failed_ids
    
    def get_ready_tasks(self) -> List[Task]:
        """
        Get all tasks that are ready to execute (dependencies satisfied).
        
        Returns:
            List of tasks ready for execution, sorted by priority
        """
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies_completed = True
            for dep_id in self.graph.predecessors(task_id):
                dep_task = self.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    dependencies_completed = False
                    break
            
            if dependencies_completed:
                ready_tasks.append(task)
        
        # Sort by priority (highest first)
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        self.logger.debug(f"Found {len(ready_tasks)} ready tasks")
        return ready_tasks
    
    def get_topological_order(self) -> List[str]:
        """
        Get topological ordering of all tasks.
        
        Returns:
            List of task IDs in valid execution order
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            self.logger.error(f"Cannot compute topological order: {e}")
            return []
    
    def find_critical_path(self) -> Tuple[List[str], int]:
        """
        Find the critical path (longest path) through the task graph.
        
        Returns:
            Tuple of (task_ids_in_critical_path, path_length)
        """
        if not self.graph.nodes:
            return [], 0
        
        # Find longest path using networkx
        try:
            path = nx.dag_longest_path(self.graph)
            return path, len(path)
        except nx.NetworkXError:
            return [], 0
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """
        Update task status and return whether graph state changed.
        
        Args:
            task_id: ID of task to update
            status: New status
            
        Returns:
            True if status was updated
        """
        if task_id not in self.tasks:
            self.logger.warning(f"Task {task_id} not found in graph")
            return False
        
        old_status = self.tasks[task_id].status
        self.tasks[task_id].status = status
        
        self.logger.debug(f"Updated task {task_id}: {old_status.value} -> {status.value}")
        return True
    
    def get_blocked_tasks(self) -> List[Task]:
        """Get tasks that are blocked by failed dependencies (direct or transitive)."""
        blocked_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if any ancestor task failed
            has_failed_dependency = False
            ancestors = self.get_task_ancestors(task_id)
            
            for ancestor_id in ancestors:
                ancestor_task = self.tasks[ancestor_id]
                if ancestor_task.status == TaskStatus.FAILED:
                    has_failed_dependency = True
                    break
            
            if has_failed_dependency:
                blocked_tasks.append(task)
        
        return blocked_tasks
    
    def get_task_descendants(self, task_id: str) -> Set[str]:
        """Get all tasks that depend on the given task (directly or indirectly)."""
        if task_id not in self.graph:
            return set()
        
        return set(nx.descendants(self.graph, task_id))
    
    def get_task_ancestors(self, task_id: str) -> Set[str]:
        """Get all tasks that the given task depends on (directly or indirectly)."""
        if task_id not in self.graph:
            return set()
        
        return set(nx.ancestors(self.graph, task_id))
    
    def remove_redundant_tasks(self) -> List[str]:
        """
        Remove tasks that are redundant or no longer needed.
        
        Returns:
            List of removed task IDs
        """
        # For now, just identify obviously redundant tasks
        # TODO: More sophisticated redundancy detection
        redundant = []
        
        # Find duplicate task names/descriptions
        seen_descriptions = {}
        for task_id, task in self.tasks.items():
            desc_key = (task.name.lower(), task.description.lower())
            if desc_key in seen_descriptions:
                # Mark later task as redundant
                redundant.append(task_id)
                self.logger.info(f"Identified redundant task: {task.name}")
            else:
                seen_descriptions[desc_key] = task_id
        
        # Remove redundant tasks
        for task_id in redundant:
            self.graph.remove_node(task_id)
            del self.tasks[task_id]
        
        return redundant
    
    def mark_task_optional(self, task_id: str):
        """Mark task as optional - failure won't block dependents."""
        task = self.tasks.get(task_id)
        if task:
            task.metadata["optional"] = True
            self.logger.info(f"Marked task as optional: {task.name}")
            
    def handle_optional_task_failure(self, failed_task_id: str):
        """Remove optional task dependencies when task fails."""
        failed_task = self.tasks.get(failed_task_id)
        
        if failed_task and failed_task.metadata.get("optional", False):
            # Find tasks that depend on this failed optional task
            for task in self.tasks.values():
                if failed_task_id in task.dependencies:
                    task.dependencies.remove(failed_task_id)
                    self.logger.info(f"Removed optional dependency {failed_task_id} from {task.name}")
                    # Unblock if no other dependencies 
                    if task.status == TaskStatus.BLOCKED and self._all_dependencies_met(task):
                        task.status = TaskStatus.PENDING
                        self.logger.info(f"Unblocked task: {task.name}")
                        
    def _all_dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies for a task are met."""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
                    
    def auto_mark_optional_tasks(self):
        """Automatically identify and mark optional tasks."""
        optional_keywords = ["spatial", "relationship", "chart", "visualization", "graph"]
        
        for task in self.tasks.values():
            task_text = f"{task.name.lower()} {task.description.lower()}"
            if any(keyword in task_text for keyword in optional_keywords):
                self.mark_task_optional(task.id)
    
    def get_graph_metrics(self) -> GraphMetrics:
        """Get comprehensive metrics about the graph state."""
        status_counts = defaultdict(int)
        for task in self.tasks.values():
            status_counts[task.status] += 1
        
        critical_path, path_length = self.find_critical_path()
        
        return GraphMetrics(
            total_tasks=len(self.tasks),
            ready_tasks=len(self.get_ready_tasks()),
            completed_tasks=status_counts[TaskStatus.COMPLETED],
            failed_tasks=status_counts[TaskStatus.FAILED], 
            blocked_tasks=len(self.get_blocked_tasks()),
            dependency_edges=self.graph.number_of_edges(),
            longest_path=path_length,
            critical_path_tasks=critical_path
        )
    
    def visualize_graph(self, filename: Optional[str] = None) -> str:
        """
        Generate text-based visualization of the task graph.
        
        Args:
            filename: Optional file to save visualization
            
        Returns:
            Text representation of the graph
        """
        if not self.tasks:
            return "Empty task graph"
        
        lines = []
        lines.append("Task Graph Visualization")
        lines.append("=" * 50)
        
        # Show tasks in topological order
        topo_order = self.get_topological_order()
        
        for i, task_id in enumerate(topo_order):
            task = self.tasks[task_id]
            
            # Task info
            status_icon = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.IN_PROGRESS: "🔄", 
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.BLOCKED: "🚫"
            }.get(task.status, "❓")
            
            priority_icon = {
                Priority.LOW: "🔸",
                Priority.MEDIUM: "🔷",
                Priority.HIGH: "🔺",
                Priority.CRITICAL: "🚨"
            }.get(task.priority, "🔷")
            
            lines.append(f"{i+1:2d}. {status_icon} {priority_icon} {task.name}")
            
            # Dependencies
            deps = list(self.graph.predecessors(task_id))
            if deps:
                dep_names = [self.tasks[dep_id].name for dep_id in deps]
                lines.append(f"     ⬅️ Depends on: {', '.join(dep_names)}")
            
            # Dependents
            dependents = list(self.graph.successors(task_id))
            if dependents:
                dep_names = [self.tasks[dep_id].name for dep_id in dependents]
                lines.append(f"     ➡️ Required by: {', '.join(dep_names)}")
        
        # Summary
        metrics = self.get_graph_metrics()
        lines.append("\nGraph Metrics")
        lines.append("-" * 20)
        lines.append(f"Total tasks: {metrics.total_tasks}")
        lines.append(f"Ready: {metrics.ready_tasks}, Completed: {metrics.completed_tasks}")
        lines.append(f"Failed: {metrics.failed_tasks}, Blocked: {metrics.blocked_tasks}")
        lines.append(f"Dependencies: {metrics.dependency_edges}")
        lines.append(f"Critical path length: {metrics.longest_path}")
        
        visualization = "\n".join(lines)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(visualization)
            self.logger.info(f"Graph visualization saved to {filename}")
        
        return visualization
    
    def validate_graph(self) -> Dict[str, List[str]]:
        """
        Validate graph integrity and return any issues.
        
        Returns:
            Dict with issue types as keys and issue descriptions as values
        """
        issues = defaultdict(list)
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle = nx.find_cycle(self.graph)
                issues["cycles"].append(f"Cycle detected: {cycle}")
            except nx.NetworkXNoCycle:
                pass
        
        # Check for orphaned tasks (no dependencies but not root-level)
        for task_id, task in self.tasks.items():
            if (not task.dependencies and 
                task_id not in self.graph.nodes() or 
                self.graph.in_degree(task_id) == 0):
                
                # Check if it's truly independent or missing dependencies
                if "Load" not in task.name and "load" not in task.description.lower():
                    issues["potential_orphans"].append(f"Task {task.name} has no dependencies")
        
        # Check for missing dependencies
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    issues["missing_dependencies"].append(
                        f"Task {task.name} depends on missing task {dep_id}"
                    )
        
        return dict(issues)