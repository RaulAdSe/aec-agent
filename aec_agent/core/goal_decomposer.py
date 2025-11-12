"""
Goal Decomposer - Breaks complex goals into actionable subtasks.

This module analyzes high-level goals and creates structured task breakdowns
optimized for AEC compliance workflows using deterministic pattern matching.
"""

import uuid
from typing import Dict, Any, List, Optional

from .reasoning_utils import ReasoningUtils, Task, Priority


class GoalDecomposer:
    """
    Breaks complex goals into actionable subtasks with dependencies.
    
    Uses deterministic pattern matching for AEC compliance workflows.
    No fallbacks - explicit failure for unmatched patterns.
    """
    
    def __init__(self):
        """Initialize the goal decomposer."""
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # AEC-specific task patterns
        self.aec_patterns = {
            "fire_safety": [
                "Load building data",
                "Identify fire-related elements",
                "Retrieve fire safety regulations", 
                "Validate fire safety compliance"
            ],
            "accessibility": [
                "Load building data",
                "Identify accessibility elements",
                "Retrieve accessibility standards",
                "Validate accessibility compliance"
            ],
            "structural": [
                "Load building data",
                "Identify structural elements",
                "Calculate structural metrics",
                "Validate structural requirements"
            ],
            "energy": [
                "Load building data",
                "Identify energy-related elements",
                "Calculate energy metrics",
                "Validate energy efficiency requirements"
            ],
            "simple_analysis": [
                "Load and parse building data",
                "Extract requested information",
                "Present results"
            ]
        }
    
    def decompose(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose a goal into actionable subtasks.
        
        Args:
            goal: High-level goal (e.g., "Analyze fire safety compliance")
            context: Current context (loaded data, session state, etc.)
            
        Returns:
            Dict containing success status, tasks list, and metadata
        """
        self.logger.info(f"Decomposing goal: {goal}")
        
        try:
            # Use only AEC pattern matching - no fallbacks
            aec_tasks = self._try_aec_pattern_match(goal, context)
            if aec_tasks:
                self.logger.info(f"Used AEC pattern for decomposition")
                return {
                    "success": True,
                    "tasks": aec_tasks,
                    "method": "aec_pattern",
                    "message": f"Decomposed using AEC pattern into {len(aec_tasks)} tasks"
                }
            
            # No pattern match - fail explicitly
            self.logger.error(f"No AEC pattern matches goal: {goal}")
            return {
                "success": False,
                "tasks": [],
                "message": f"Goal does not match any known AEC patterns: {goal}",
                "available_patterns": list(self.aec_patterns.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Goal decomposition failed: {e}")
            return {
                "success": False,
                "tasks": [],
                "message": f"Decomposition failed: {str(e)}",
                "error": ReasoningUtils.extract_error_info(e)
            }
    
    def _try_aec_pattern_match(self, goal: str, context: Dict[str, Any]) -> Optional[List[Task]]:
        """Try to match goal against known AEC patterns."""
        goal_lower = goal.lower()
        
        # Detect AEC domain - be more specific to avoid false matches
        if any(term in goal_lower for term in ["fire safety", "fire compliance", "emergency exit", "fire protection"]):
            pattern = "fire_safety"
        elif any(term in goal_lower for term in ["accessibility compliance", "accessible design", "ada compliance"]):
            pattern = "accessibility"
        elif any(term in goal_lower for term in ["structural analysis", "structural compliance", "load calculation", "beam analysis"]):
            pattern = "structural"
        elif any(term in goal_lower for term in ["energy efficiency", "thermal analysis", "hvac compliance"]):
            pattern = "energy"
        elif any(term in goal_lower for term in ["count", "list", "extract", "show", "get all", "find all", "query"]):
            pattern = "simple_analysis"
        else:
            return None
        
        # Get pattern and create tasks
        task_names = self.aec_patterns.get(pattern, [])
        if not task_names:
            return None
        
        tasks = []
        prev_task_id = None
        
        for i, name in enumerate(task_names):
            task_id = str(uuid.uuid4())
            
            # Create task with appropriate dependencies
            task = Task(
                id=task_id,
                name=name,
                description=self._generate_task_description(name, pattern, goal),
                priority=Priority.HIGH if i == 0 else Priority.MEDIUM,
                dependencies=[prev_task_id] if prev_task_id else [],
                metadata={
                    "pattern": pattern,
                    "original_goal": goal,
                    "sequence_order": i
                }
            )
            
            tasks.append(task)
            prev_task_id = task_id
        
        return tasks
    
    def _generate_task_description(self, task_name: str, pattern: str, goal: str) -> str:
        """Generate detailed description for a task."""
        
        # Extract file path from goal if present
        file_path = self._extract_file_path_from_goal(goal)
        
        descriptions = {
            "Load building data": f"Load and validate building data from {file_path} required for {pattern} analysis" if file_path else f"Load and validate building data required for {pattern} analysis",
            "Load and parse building data": f"Load and validate building data from {file_path} required for {pattern} analysis" if file_path else f"Load and validate building data required for {pattern} analysis",
            "Identify fire-related elements": "Find all fire doors, exits, zones, and safety equipment",
            "Identify accessibility elements": "Find all doors, ramps, elevators, and accessibility features",
            "Identify structural elements": "Find all beams, columns, slabs, and structural components",
            "Identify energy-related elements": "Find all HVAC, lighting, and energy systems",
            "Extract requested information": "Extract and process the specific information requested in the goal",
            "Present results": "Format and present the extracted information in a clear summary",
            "Retrieve fire safety regulations": "Search compliance documents for fire safety requirements",
            "Retrieve accessibility standards": "Search compliance documents for accessibility standards",
            "Calculate structural metrics": "Calculate loads, stresses, and structural performance metrics",
            "Calculate energy metrics": "Calculate energy consumption and efficiency metrics",
            "Validate fire safety compliance": "Check all fire elements against safety regulations",
            "Validate accessibility compliance": "Check all accessibility elements against standards",
            "Validate structural requirements": "Check all structural elements against building codes",
            "Validate energy efficiency requirements": "Check energy systems against efficiency standards"
        }
        
        return descriptions.get(task_name, f"Complete {task_name.lower()} for {goal}")
    
    def _extract_file_path_from_goal(self, goal: str) -> Optional[str]:
        """Extract file path from goal text."""
        words = goal.split()
        for word in words:
            if '.json' in word or 'data/' in word:
                return word.strip('",.')
        return None