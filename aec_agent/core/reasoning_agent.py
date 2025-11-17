"""
Reasoning Agent - Autonomous AEC compliance agent with goal-driven reasoning.

This module provides the main entry point for the autonomous reasoning agent
that can break down complex AEC compliance goals into tasks and execute them
intelligently using available tools.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.tracers.langchain import LangChainTracer

# Import reasoning components
from .reasoning_controller import ReasoningController
from .goal_decomposer import GoalDecomposer
from .tool_planner import ToolPlanner
from .executor import ToolExecutor
from .validator import ResultValidator
from .reasoning_utils import ReasoningUtils

# Import tool system
from ..tools.building_data_toolkit import (
    load_building_data,
    get_all_elements,
    get_all_properties,
    query_elements,
    calculate,
    find_related,
    validate_rule
)
from ..tools.compliance_search import search_compliance_docs

# Import memory system (reuse existing)
from ..memory import MemoryManager, MemoryManagerConfig

# Import unified configuration
from ..config import AgentConfig


class ReasoningAgent:
    """
    Autonomous AEC compliance agent with goal-driven reasoning.
    
    Features:
    - Autonomous goal decomposition and task planning
    - Intelligent tool selection and sequencing
    - Real-time execution monitoring and validation
    - Memory-driven learning and optimization
    - Error recovery and adaptive planning
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        # Legacy parameters for backward compatibility
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        verbose: bool = True,
        enable_memory: Optional[bool] = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        session_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        max_execution_time: Optional[float] = None
    ):
        """
        Initialize the reasoning agent.
        
        Args:
            config: Unified agent configuration (preferred)
            model_name: LLM model to use (legacy, overrides config)
            temperature: LLM temperature (legacy, overrides config)
            verbose: Enable verbose logging
            enable_memory: Enable memory system (legacy, overrides config)
            memory_config: Memory system configuration
            session_id: Session identifier
            max_iterations: Maximum reasoning iterations (legacy, overrides config)
            max_execution_time: Maximum execution time in seconds (legacy, overrides config)
        """
        # Initialize configuration
        self.config = config or AgentConfig.from_env()
        
        # Apply legacy parameter overrides if provided
        if model_name is not None:
            self.config.llm.model_name = model_name
        if temperature is not None:
            self.config.llm.temperature = temperature
        if enable_memory is not None:
            self.config.memory.enable_short_term_memory = enable_memory
        if max_iterations is not None:
            self.config.reasoning.max_iterations = max_iterations
        if max_execution_time is not None:
            self.config.reasoning.max_execution_time = max_execution_time
        if session_id is not None:
            self.config.session_id = session_id
        
        # Store commonly used values for convenience
        self.model_name = self.config.llm.model_name
        self.temperature = self.config.llm.temperature
        self.verbose = verbose
        self.enable_memory = self.config.memory.enable_short_term_memory
        self.max_iterations = self.config.reasoning.max_iterations
        self.max_execution_time = self.config.reasoning.max_execution_time
        
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Setup memory system
        if enable_memory:
            self._setup_memory(memory_config, session_id)
        else:
            self.memory_manager = None
        
        # Setup LangSmith tracing
        self._setup_langsmith_tracing()
        
        # Initialize LLM
        self._setup_llm()
        
        # Initialize tool registry
        self._setup_tool_registry()
        
        # Initialize reasoning components
        self._setup_reasoning_components()
        
        self.logger.info(f"Reasoning agent initialized with model {model_name}")
    
    def _setup_memory(self, memory_config: Optional[MemoryManagerConfig], session_id: Optional[str]):
        """Setup the memory management system."""
        if memory_config is None:
            persistence_path = Path.cwd() / "data" / "sessions"
            memory_config = MemoryManagerConfig(
                session_persistence_path=persistence_path,
                enable_persistence=True,
                auto_save_interval=5
            )
        
        self.memory_manager = MemoryManager(
            config=memory_config,
            session_id=session_id
        )
        
        self.logger.info(f"Memory system initialized with session_id={self.memory_manager.get_session_id()}")
    
    def _setup_langsmith_tracing(self):
        """Setup LangSmith tracing for monitoring."""
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = "AEC-Reasoning-Agent"
            
            tracer = LangChainTracer(project_name="AEC-Reasoning-Agent")
            self.callback_manager = CallbackManager([tracer])
            
            self.logger.info("LangSmith tracing enabled for project: AEC-Reasoning-Agent")
        else:
            self.callback_manager = None
            self.logger.warning("LANGSMITH_API_KEY not found - tracing disabled")
    
    def _setup_llm(self):
        """Setup the language model."""
        llm_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": 4000,
            "model_kwargs": {
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        }
        
        if self.callback_manager:
            llm_kwargs["callbacks"] = self.callback_manager
        
        self.llm = ChatOpenAI(**llm_kwargs)
    
    def _setup_tool_registry(self):
        """Setup the tool registry with all available tools."""
        self.tool_registry = {
            "load_building_data": self._create_building_data_wrapper(),
            "get_all_elements": get_all_elements,
            "get_element_properties": get_all_properties,
            "query_elements": self._create_query_wrapper(),
            "calculate_distances": self._create_distances_wrapper(),
            "calculate_areas": self._create_areas_wrapper(),
            "find_related_elements": self._create_related_wrapper(),
            "validate_compliance_rule": self._create_validation_wrapper(),
            "search_compliance_documents": search_compliance_docs,
            "document_findings": self._create_documentation_wrapper()
        }
    
    def _create_building_data_wrapper(self):
        """Create wrapper for building data loading that integrates with memory."""
        def wrapper(input_param: str) -> Dict[str, Any]:
            # Extract file path from input parameter or task description
            import re
            import os
            
            # If input_param looks like a file path, use it directly
            if input_param and (input_param.endswith('.json') or os.path.exists(input_param)):
                file_path = input_param
            else:
                # Try to extract file path from the input text
                # Look for patterns like "data/out/FM-ARC_v2.json" or similar
                path_patterns = [
                    r'["\']?([^"\']*\.json)["\']?',  # JSON files in quotes or not
                    r'data/[^"\s]*\.json',            # Paths starting with data/
                    r'[^"\s]*FM-ARC[^"\s]*\.json'     # Specific file pattern
                ]
                
                file_path = None
                for pattern in path_patterns:
                    matches = re.findall(pattern, input_param)
                    if matches:
                        file_path = matches[0]
                        break
                
                # If no path found in input, try common default locations
                if not file_path:
                    default_paths = [
                        "data/out/FM-ARC_v2.json",
                        "data/blueprints/FM-ARC_v2.json",
                        "data/FM-ARC_v2.json"
                    ]
                    for default_path in default_paths:
                        if os.path.exists(default_path):
                            file_path = default_path
                            break
                
                if not file_path:
                    return {
                        "status": "error",
                        "message": f"Could not extract file path from input: {input_param}. Please provide a valid file path."
                    }
            
            result = load_building_data(file_path)
            
            # Track in memory if available
            if self.memory_manager and result.get("status") == "success":
                data = result.get("data", {})
                file_info = data.get("file_info", {})
                
                # Extract summary for memory
                context_data = {
                    "project_name": file_info.get("project_name", "Unknown"),
                    "total_elements": file_info.get("total_elements", 0),
                    "schema": file_info.get("schema", "Unknown"),
                    "available_element_types": {
                        "spaces": len(data.get("spaces", [])),
                        "doors": len(data.get("doors", [])),
                        "walls": len(data.get("walls", [])),
                        "slabs": len(data.get("slabs", [])),
                        "stairs": len(data.get("stairs", []))
                    }
                }
                
                self.memory_manager.set_building_data_context(path, context_data)
                self.memory_manager.track_active_file(path)
                
                self.memory_manager.record_tool_execution(
                    tool_name="load_building_data",
                    arguments={"path": path},
                    success=True,
                    result_summary=f"Loaded building data: {context_data['total_elements']} elements"
                )
            
            return result
        
        return wrapper
    
    def _create_query_wrapper(self):
        """Create wrapper for query_elements with JSON parsing."""
        def wrapper(input_str: str) -> Dict[str, Any]:
            import json
            try:
                if isinstance(input_str, str):
                    params = json.loads(input_str)
                else:
                    params = input_str
                
                element_type = params.get("element_type")
                filters = params.get("filters", {})
                
                if not element_type:
                    return {
                        "status": "error",
                        "message": "Missing required 'element_type' parameter"
                    }
                
                return query_elements(element_type, filters)
                
            except Exception as e:
                return {
                    "status": "error", 
                    "message": f"Query execution failed: {str(e)}"
                }
        
        return wrapper
    
    def _create_calculate_wrapper(self):
        """Create wrapper for calculate with JSON parsing."""
        def wrapper(input_str: str) -> Dict[str, Any]:
            import json
            try:
                if isinstance(input_str, str):
                    params = json.loads(input_str)
                else:
                    params = input_str
                
                operation = params.pop("operation", None)
                if not operation:
                    return {
                        "status": "error",
                        "message": "Missing required 'operation' parameter"
                    }
                
                return calculate(operation, **params)
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Calculation failed: {str(e)}"
                }
        
        return wrapper
    
    def _create_distances_wrapper(self):
        """Create wrapper for distance calculations only."""
        def wrapper(input_str: str) -> Dict[str, Any]:
            import json
            try:
                if isinstance(input_str, str):
                    params = json.loads(input_str)
                else:
                    params = input_str
                
                # For distance calculations, determine the operation based on parameters
                if "element1_id" in params and "element2_id" in params:
                    operation = "distance_between_elements"
                elif "point1" in params and "point2" in params:
                    # Check if 3D or 2D based on point dimensions
                    point1 = params.get("point1", [])
                    if len(point1) >= 3:
                        operation = "distance_3d"
                    else:
                        operation = "distance_2d"
                else:
                    return {
                        "status": "error",
                        "message": "Distance calculation requires either element IDs (element1_id, element2_id) or coordinates (point1, point2)"
                    }
                
                return calculate(operation, **params)
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Distance calculation failed: {str(e)}"
                }
        
        return wrapper
    
    def _create_areas_wrapper(self):
        """Create wrapper for area/volume calculations only."""
        def wrapper(input_str: str) -> Dict[str, Any]:
            import json
            try:
                if isinstance(input_str, str):
                    params = json.loads(input_str)
                else:
                    params = input_str
                
                # Determine operation based on request
                calculation_type = params.pop("calculation_type", "area")
                
                if calculation_type in ["volume", "volumes"]:
                    operation = "volume_sum"
                else:
                    operation = "area_sum"  # Default to area calculation
                
                return calculate(operation, **params)
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Area/volume calculation failed: {str(e)}"
                }
        
        return wrapper
    
    def _create_related_wrapper(self):
        """Create wrapper for find_related with JSON parsing."""
        def wrapper(input_str: str) -> Dict[str, Any]:
            import json
            try:
                if isinstance(input_str, str):
                    params = json.loads(input_str)
                else:
                    params = input_str
                
                element_id = params.pop("element_id", None)
                relationship_type = params.pop("relationship_type", None)
                
                if not element_id or not relationship_type:
                    return {
                        "status": "error",
                        "message": "Missing required parameters: element_id, relationship_type"
                    }
                
                return find_related(element_id, relationship_type, **params)
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Find related failed: {str(e)}"
                }
        
        return wrapper
    
    def _create_validation_wrapper(self):
        """Create wrapper for validate_rule with JSON parsing."""
        def wrapper(input_str: str) -> Dict[str, Any]:
            import json
            try:
                if isinstance(input_str, str):
                    params = json.loads(input_str)
                else:
                    params = input_str
                
                rule_type = params.pop("rule_type", None)
                element_id = params.pop("element_id", None) 
                criteria = params.pop("criteria", {})
                
                if not rule_type or not element_id:
                    return {
                        "status": "error",
                        "message": "Missing required parameters: rule_type, element_id"
                    }
                
                return validate_rule(rule_type, element_id, criteria)
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Validation failed: {str(e)}"
                }
        
        return wrapper
    
    def _create_documentation_wrapper(self):
        """Create wrapper for documenting findings and results."""
        def wrapper(input_str: str) -> Dict[str, Any]:
            import json
            from datetime import datetime
            
            try:
                if isinstance(input_str, str):
                    try:
                        params = json.loads(input_str)
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        params = {"content": input_str}
                else:
                    params = input_str
                
                content = params.get("content", "")
                document_type = params.get("type", "findings")
                title = params.get("title", "Analysis Results")
                
                if not content:
                    return {
                        "status": "error",
                        "message": "No content provided to document"
                    }
                
                # Create structured documentation
                documentation = {
                    "title": title,
                    "type": document_type,
                    "timestamp": datetime.now().isoformat(),
                    "content": content,
                    "sections": params.get("sections", []),
                    "summary": params.get("summary", ""),
                    "recommendations": params.get("recommendations", [])
                }
                
                # Store in memory if available
                if self.memory_manager:
                    self.memory_manager.record_tool_execution(
                        tool_name="document_findings",
                        arguments={"title": title, "type": document_type},
                        success=True,
                        result_summary=f"Documented {document_type}: {title}"
                    )
                
                return {
                    "status": "success",
                    "message": f"Successfully documented {document_type}: {title}",
                    "data": documentation,
                    "document_id": f"{document_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }
                
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Documentation failed: {str(e)}"
                }
        
        return wrapper
    
    def _setup_reasoning_components(self):
        """Initialize all reasoning components."""
        # Initialize individual components with LLM support
        self.goal_decomposer = GoalDecomposer(llm=self.llm)
        self.tool_planner = ToolPlanner(llm=self.llm)
        self.executor = ToolExecutor(
            tool_registry=self.tool_registry,
            timeout=60.0
        )
        self.validator = ResultValidator(llm=self.llm)
        
        # Create the main reasoning controller
        self.reasoning_controller = ReasoningController(
            goal_decomposer=self.goal_decomposer,
            tool_planner=self.tool_planner,
            executor=self.executor,
            validator=self.validator,
            max_iterations=self.max_iterations,
            max_execution_time=self.max_execution_time,
            llm=self.llm
        )
    
    @traceable(name="reasoning_agent_process", project_name="AEC-Reasoning-Agent")
    def process_goal(self, goal: str, clear_history: bool = False) -> Dict[str, Any]:
        """
        Process a high-level goal using autonomous reasoning.
        
        Args:
            goal: High-level goal description (e.g., "Analyze fire safety compliance")
            clear_history: Whether to clear conversation history
            
        Returns:
            Comprehensive results including task breakdown, execution results, and analysis
        """
        try:
            # Clear memory if requested
            if clear_history and self.memory_manager:
                self.memory_manager.clear_conversation_memory()
                self.logger.info("Conversation history cleared")
            
            # Set session goal in memory
            if self.memory_manager:
                self.memory_manager.set_session_goal(goal)
                # Get context from memory manager for reasoning
                context = {
                    "session_summary": self.memory_manager.get_session_summary(),
                    "active_files": self.memory_manager.get_active_files(),
                    "recent_tools": self.memory_manager.get_recent_tool_executions(5)
                }
            else:
                context = {}
            
            # Execute autonomous reasoning
            reasoning_result = self.reasoning_controller.reason(goal, context)
            
            # Store conversation turn in memory
            if self.memory_manager:
                summary = reasoning_result.get("summary", {})
                response_text = f"Goal: {goal}\nResult: {reasoning_result.get('message', 'Processing completed')}\nTasks completed: {summary.get('completed_tasks', 0)}/{summary.get('total_tasks', 0)}"
                self.memory_manager.add_conversation_turn(goal, response_text)
            
            # Add memory summary to response
            response = {
                "status": reasoning_result.get("status", "unknown"),
                "message": reasoning_result.get("message", ""),
                "goal": goal,
                "reasoning_result": reasoning_result
            }
            
            if self.memory_manager:
                response["session_summary"] = self.memory_manager.get_session_summary()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing goal: {e}")
            error_response = {
                "status": "error", 
                "message": f"Goal processing failed: {str(e)}",
                "goal": goal
            }
            
            # Still store failed conversation if memory is enabled
            if self.memory_manager:
                self.memory_manager.add_conversation_turn(goal, f"Error: {str(e)}")
            
            return error_response
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        status = {
            "name": "AEC Reasoning Agent",
            "status": "ready",
            "model": self.model_name,
            "reasoning_enabled": True,
            "memory_enabled": self.enable_memory,
            "tools": len(self.tool_registry),
            "tool_list": list(self.tool_registry.keys()),
            "max_iterations": self.max_iterations,
            "max_execution_time": self.max_execution_time
        }
        
        if self.memory_manager:
            status["session_id"] = self.memory_manager.get_session_id()
            status["memory_summary"] = self.memory_manager.get_session_summary()
        
        return status
    
    # Memory management methods
    def set_session_goal(self, goal: str, context: str = "") -> None:
        """Set the main goal for this session."""
        if self.memory_manager:
            self.memory_manager.set_session_goal(goal, context)
            self.logger.info(f"Session goal set: {goal}")
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """Get session summary."""
        if self.memory_manager:
            return self.memory_manager.get_session_summary()
        return None
    
    def save_session(self) -> None:
        """Save the current session to disk."""
        if self.memory_manager:
            self.memory_manager.save_session()
            self.logger.info("Session saved")
    
    def clear_memory(self, conversation_only: bool = True) -> None:
        """Clear memory."""
        if self.memory_manager:
            if conversation_only:
                self.memory_manager.clear_conversation_memory()
                self.logger.info("Conversation memory cleared")
            else:
                self.memory_manager.clear_all_memory()
                self.logger.info("All memory cleared")


def create_reasoning_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    verbose: bool = True,
    enable_memory: bool = True,
    memory_config: Optional[MemoryManagerConfig] = None,
    session_id: Optional[str] = None,
    max_iterations: int = 20,
    max_execution_time: float = 300.0
) -> ReasoningAgent:
    """
    Create a reasoning agent with specified configuration.
    
    Args:
        model_name: LLM model to use
        temperature: LLM temperature
        verbose: Enable verbose logging
        enable_memory: Enable memory system
        memory_config: Memory system configuration
        session_id: Session identifier
        max_iterations: Maximum reasoning iterations
        max_execution_time: Maximum execution time
        
    Returns:
        Configured reasoning agent
    """
    return ReasoningAgent(
        model_name=model_name,
        temperature=temperature,
        verbose=verbose,
        enable_memory=enable_memory,
        memory_config=memory_config,
        session_id=session_id,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time
    )