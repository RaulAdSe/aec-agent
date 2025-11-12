"""
Reasoning Agent - Autonomous AEC compliance agent with goal-driven reasoning.

This module provides the main entry point for the autonomous reasoning agent
that can break down complex AEC compliance goals into tasks and execute them
intelligently using available tools.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer

# Import reasoning components
from .core.reasoning_controller import ReasoningController
from .core.goal_decomposer import GoalDecomposer
from .core.tool_planner import ToolPlanner
from .core.executor import ToolExecutor
from .core.validator import ResultValidator
from .core.reasoning_utils import ReasoningUtils

# Import tool system
from .tools.building_data_toolkit import (
    load_building_data,
    get_all_elements,
    get_all_properties,
    query_elements,
    calculate,
    find_related,
    validate_rule
)
from .tools.compliance_search import search_compliance_docs

# Import memory system (reuse existing)
from .memory import MemoryManager, MemoryManagerConfig


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
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        verbose: bool = True,
        enable_memory: bool = True,
        memory_config: Optional[MemoryManagerConfig] = None,
        session_id: Optional[str] = None,
        max_iterations: int = 20,
        max_execution_time: float = 300.0
    ):
        """
        Initialize the reasoning agent.
        
        Args:
            model_name: LLM model to use
            temperature: LLM temperature
            verbose: Enable verbose logging
            enable_memory: Enable memory system
            memory_config: Memory system configuration
            session_id: Session identifier
            max_iterations: Maximum reasoning iterations
            max_execution_time: Maximum execution time in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.enable_memory = enable_memory
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        
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
            session_id=session_id,
            enable_persistence=memory_config.enable_persistence,
            persistence_path=memory_config.session_persistence_path,
            auto_save_interval=memory_config.auto_save_interval
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
            "calculate_metrics": self._create_calculate_wrapper(),
            "find_related_elements": self._create_related_wrapper(),
            "validate_compliance_rule": self._create_validation_wrapper(),
            "search_compliance_documents": search_compliance_docs
        }
    
    def _create_building_data_wrapper(self):
        """Create wrapper for building data loading that integrates with memory."""
        def wrapper(path: str) -> Dict[str, Any]:
            result = load_building_data(path)
            
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
                self.memory_manager.track_active_file(path, "building_data")
                
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
    
    def _setup_reasoning_components(self):
        """Initialize all reasoning components."""
        # Initialize individual components
        self.goal_decomposer = GoalDecomposer(llm=self.llm)
        self.tool_planner = ToolPlanner(llm=self.llm)
        self.executor = ToolExecutor(
            tool_registry=self.tool_registry,
            timeout=60.0
        )
        self.validator = ResultValidator()
        
        # Create the main reasoning controller
        self.reasoning_controller = ReasoningController(
            goal_decomposer=self.goal_decomposer,
            tool_planner=self.tool_planner,
            executor=self.executor,
            validator=self.validator,
            max_iterations=self.max_iterations,
            max_execution_time=self.max_execution_time
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
                context = self.memory_manager.get_reasoning_context()
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