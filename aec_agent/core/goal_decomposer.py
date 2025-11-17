"""
Goal Decomposer - Breaks complex goals into actionable subtasks.

This module analyzes high-level goals and creates structured task breakdowns
optimized for AEC compliance workflows using deterministic pattern matching.
"""

import uuid
from typing import Dict, Any, List, Optional

from .reasoning_utils import ReasoningUtils, Task, Priority

# Import LangSmith tracing
from langsmith import traceable

# Import LLM components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


class GoalDecomposer:
    """
    Breaks complex goals into actionable subtasks with dependencies.
    
    Uses LLM-based reasoning with AEC domain knowledge for intelligent task decomposition.
    """
    
    def __init__(self, llm: ChatOpenAI = None):
        """Initialize the goal decomposer."""
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Setup LLM for reasoning
        if llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=2000
            )
        else:
            self.llm = llm
        
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
    
    @traceable(name="goal_decomposition", metadata={"component": "goal_decomposer"})
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
            # First try LLM-based decomposition for intelligent analysis
            llm_tasks = self._llm_decompose_goal(goal, context)
            if llm_tasks:
                self.logger.info(f"Used LLM for goal decomposition")
                return {
                    "success": True,
                    "tasks": llm_tasks,
                    "method": "llm_reasoning",
                    "message": f"Decomposed using LLM reasoning into {len(llm_tasks)} tasks"
                }
            
            # Fallback to pattern matching if LLM fails
            aec_tasks = self._try_aec_pattern_match(goal, context)
            if aec_tasks:
                self.logger.info(f"Used AEC pattern fallback for decomposition")
                return {
                    "success": True,
                    "tasks": aec_tasks,
                    "method": "aec_pattern_fallback",
                    "message": f"Decomposed using AEC pattern fallback into {len(aec_tasks)} tasks"
                }
            
            # No decomposition possible
            self.logger.error(f"Both LLM and pattern matching failed for goal: {goal}")
            return {
                "success": False,
                "tasks": [],
                "message": f"Goal decomposition failed: {goal}",
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
    
    @traceable(name="llm_goal_decomposition")
    def _llm_decompose_goal(self, goal: str, context: Dict[str, Any]) -> Optional[List[Task]]:
        """Use LLM to intelligently decompose goals into tasks."""
        
        # Create decomposition prompt
        decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AEC (Architecture, Engineering, Construction) compliance analyst.
Your job is to break down complex building analysis goals into specific, actionable tasks.

Available tools for task execution:
- load_building_data: Load IFC JSON building data files
- get_all_elements: Get all elements of a specific type (spaces, doors, walls, etc.)
- get_element_properties: Get detailed properties of specific elements  
- query_elements: Filter elements with specific criteria (JSON input)
- calculate_metrics: Perform calculations (areas, volumes, counts, etc.)
- find_related_elements: Find spatial relationships between elements
- validate_compliance_rule: Check elements against compliance rules
- search_compliance_documents: Search building codes and regulations

Context information: {context}

Break down the goal into 3-6 specific tasks that:
1. Follow logical dependencies (load data before analysis)
2. Are concrete and actionable
3. **Each task must be achievable with exactly ONE tool call** - this is a critical constraint
4. Use the available tools effectively
5. Lead to achieving the stated goal

**IMPORTANT CONSTRAINT**: Each task you create must be accomplishable by executing exactly ONE tool. If a goal requires multiple tool calls, break it into separate tasks with dependencies. For example:
- ❌ Bad: "Get all doors and calculate their distances" (requires 2 tools: get_all_elements + calculate_distances)
- ✅ Good: "Get all door elements" → Task 1 (uses get_all_elements)
           "Calculate distances between doors" → Task 2 (uses calculate_distances, depends on Task 1)

Return ONLY a JSON list of task objects like this:
[
  {{"name": "Load building data", "description": "Load and validate building data from the specified file", "priority": "HIGH"}},
  {{"name": "Task name", "description": "Detailed description", "priority": "MEDIUM"}},
  ...
]

Priority levels: HIGH, MEDIUM, LOW"""),
            ("human", "Goal to decompose: {goal}")
        ])
        
        try:
            # Execute LLM decomposition
            chain = decomposition_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "goal": goal,
                "context": str(context) if context else "No additional context"
            })
            
            # Parse JSON response
            import json
            task_data = json.loads(response.strip())
            
            if not isinstance(task_data, list) or len(task_data) == 0:
                self.logger.warning("LLM returned invalid task list format")
                return None
            
            # Convert to Task objects
            tasks = []
            prev_task_id = None
            
            for i, task_info in enumerate(task_data):
                task_id = str(uuid.uuid4())
                
                # Map priority strings to Priority enum
                priority_map = {
                    "HIGH": Priority.HIGH,
                    "MEDIUM": Priority.MEDIUM, 
                    "LOW": Priority.LOW
                }
                priority = priority_map.get(task_info.get("priority", "MEDIUM"), Priority.MEDIUM)
                
                # Extract file path from goal for first task
                metadata = {
                    "method": "llm_reasoning", 
                    "original_goal": goal,
                    "sequence_order": i
                }
                
                # If this is a data loading task and goal contains file path, extract it
                if i == 0 and any(term in task_info.get("name", "").lower() for term in ["load", "data"]):
                    file_path = self._extract_file_path_from_goal(goal)
                    if file_path:
                        metadata["file_path"] = file_path
                
                # Create task with dependencies
                task = Task(
                    id=task_id,
                    name=task_info.get("name", f"Task {i+1}"),
                    description=task_info.get("description", ""),
                    priority=priority,
                    dependencies=[prev_task_id] if prev_task_id and i > 0 else [],
                    metadata=metadata
                )
                
                tasks.append(task)
                prev_task_id = task_id
            
            self.logger.info(f"LLM successfully decomposed goal into {len(tasks)} tasks")
            return tasks
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM response was not valid JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"LLM decomposition failed: {e}")
            return None
    
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