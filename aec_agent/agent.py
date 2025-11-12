"""
Simple AEC Compliance Agent with access to all tools.

One agent that can handle building data analysis and compliance search.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer

from .memory import MemoryManager, MemoryManagerConfig, TaskStatus
from .tools.building_data_toolkit import (
    load_building_data,
    get_all_elements,
    get_all_properties,
    query_elements,
    calculate,
    find_related,
    validate_rule
)
from .tools.compliance_search import (
    search_compliance_docs
)


class ComplianceAgent:
    """
    Simple compliance agent with access to building data and compliance tools.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.1, 
        verbose: bool = True,
        enable_memory: bool = True,
        memory_config: Optional[MemoryManagerConfig] = None,
        session_id: Optional[str] = None
    ):
        """Initialize the compliance agent."""
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.enable_memory = enable_memory
        self.logger = logging.getLogger(__name__)
        
        # Setup memory system
        if self.enable_memory:
            self._setup_memory(memory_config, session_id)
        else:
            self.memory_manager = None
        
        # Setup LangSmith tracing
        self._setup_langsmith_tracing()
        
        # Initialize components
        self._setup_llm()
        self._setup_tools()
        self._setup_agent()
        
        memory_status = "enabled" if self.enable_memory else "disabled"
        self.logger.info(f"Compliance agent initialized with LangSmith tracing and memory {memory_status}")
    
    def _setup_memory(self, memory_config: Optional[MemoryManagerConfig], session_id: Optional[str]):
        """Setup the memory management system."""
        # Create default memory config if not provided
        if memory_config is None:
            # Setup persistence path in a data directory
            persistence_path = Path.cwd() / "data" / "sessions"
            memory_config = MemoryManagerConfig(
                session_persistence_path=persistence_path,
                enable_persistence=True,
                auto_save_interval=5  # Save every 5 operations
            )
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(config=memory_config, session_id=session_id)
        self.logger.info(f"Memory system initialized with session_id={self.memory_manager.get_session_id()}")
    
    def _setup_langsmith_tracing(self):
        """Setup LangSmith tracing for monitoring agent behavior."""
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if langsmith_api_key:
            # Configure LangSmith environment
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = "AEC-Compliance-Agent"
            
            # Setup callback manager with LangChain tracer
            tracer = LangChainTracer(project_name="AEC-Compliance-Agent")
            self.callback_manager = CallbackManager([tracer])
            
            self.logger.info("LangSmith tracing enabled for project: AEC-Compliance-Agent")
        else:
            self.callback_manager = None
            self.logger.warning("LANGSMITH_API_KEY not found - tracing disabled")
    
    def _setup_llm(self):
        """Set up the language model."""
        llm_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": 4000,
            "model_kwargs": {
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
        }
        
        # Add callbacks if LangSmith is enabled
        if self.callback_manager:
            llm_kwargs["callbacks"] = self.callback_manager
            
        self.llm = ChatOpenAI(**llm_kwargs)
    
    def _setup_tools(self):
        """Set up all available tools."""
        self.tools = [
            # Building Data Tools
            Tool(
                name="load_building_data", 
                description="Load building data from IFC JSON file. Input: file path.",
                func=self._load_building_data_wrapper
            ),
            Tool(
                name="get_all_elements",
                description="Get all elements of a type. Input: element type (spaces, doors, walls, slabs, stairs).",
                func=lambda element_type: get_all_elements(element_type)
            ),
            Tool(
                name="get_element_properties", 
                description="Get all properties for an element. Input: element ID.",
                func=lambda element_id: get_all_properties(element_id)
            ),
            Tool(
                name="query_elements",
                description="Query elements with filters. Input: JSON with element_type and optional filters.",
                func=self._query_elements_wrapper
            ),
            Tool(
                name="calculate_metrics",
                description="Perform calculations. Input: JSON with operation and parameters.",
                func=self._calculate_wrapper
            ),
            Tool(
                name="find_related_elements",
                description="Find related elements. Input: JSON with element_id, relationship_type, and parameters.",
                func=self._find_related_wrapper
            ),
            Tool(
                name="validate_compliance_rule",
                description="Validate compliance rules. Input: JSON with rule_type, element_id, and criteria.",
                func=self._validate_rule_wrapper
            ),
            
            # Compliance Search Tools
            Tool(
                name="search_compliance_documents",
                description="Search compliance documents. Input: query text.",
                func=lambda query: search_compliance_docs(query)
            ),
            # Temporarily disabled due to tool wrapper issue
            # Tool(
            #     name="check_knowledge_base_status",
            #     description="Check knowledge base status. No input needed.",
            #     func=lambda x: check_knowledge_base_status()
            # )
        ]
    
    def _load_building_data_wrapper(self, path: str) -> Dict[str, Any]:
        """
        Wrapper for load_building_data that returns full JSON data + summary.
        
        Returns the complete building data to the agent, while storing only
        the summary in memory to keep context window lean.
        """
        result = load_building_data(path)
        
        # Track tool execution in memory
        if self.memory_manager:
            success = result.get("status") == "success"
            
            if success:
                data = result.get("data", {})
                file_info = data.get("file_info", {})
                
                # Extract summary for memory (not the full JSON)
                summary_context = {
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
                
                # Store only summary in memory, not full JSON
                self.memory_manager.set_building_data_context(path, summary_context)
                self.memory_manager.track_active_file(path)
                
                result_summary = f"Loaded building data from {path} ({summary_context['total_elements']} elements)"
            else:
                result_summary = f"Failed to load {path}"
            
            self.memory_manager.record_tool_execution(
                tool_name="load_building_data",
                arguments={"path": path},
                success=success,
                result_summary=result_summary
            )
        
        # Return full JSON data + summary to the agent
        if result.get("status") == "success":
            data = result.get("data", {})
            file_info = data.get("file_info", {})
            
            # Return full data with summary included
            return {
                "status": "success",
                "data": data,  # Full JSON data
                "summary": {  # Brief summary for quick reference
                    "file_loaded": path,
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
                },
                "logs": result.get("logs", [])
            }
        else:
            return result
    
    def _query_elements_wrapper(self, input_str: str) -> Dict[str, Any]:
        """Wrapper for query_elements."""
        import json
        try:
            params = json.loads(input_str)
            return query_elements(params.get("element_type"), params.get("filters"))
        except Exception as e:
            return {"status": "error", "logs": [f"Invalid input: {e}"]}
    
    def _calculate_wrapper(self, input_str: str) -> Dict[str, Any]:
        """Wrapper for calculate."""
        import json
        try:
            params = json.loads(input_str)
            operation = params.pop("operation")
            return calculate(operation, **params)
        except Exception as e:
            return {"status": "error", "logs": [f"Invalid input: {e}"]}
    
    def _find_related_wrapper(self, input_str: str) -> Dict[str, Any]:
        """Wrapper for find_related."""
        import json
        try:
            params = json.loads(input_str)
            element_id = params.pop("element_id")
            relationship_type = params.pop("relationship_type")
            return find_related(element_id, relationship_type, **params)
        except Exception as e:
            return {"status": "error", "logs": [f"Invalid input: {e}"]}
    
    def _validate_rule_wrapper(self, input_str: str) -> Dict[str, Any]:
        """Wrapper for validate_rule."""
        import json
        try:
            params = json.loads(input_str)
            rule_type = params.pop("rule_type")
            element_id = params.pop("element_id")
            criteria = params.pop("criteria")
            return validate_rule(rule_type, element_id, criteria)
        except Exception as e:
            return {"status": "error", "logs": [f"Invalid input: {e}"]}
    
    def _setup_agent(self):
        """Set up the LangChain ReAct agent."""
        
        # Create ReAct prompt template with memory context
        if self.enable_memory:
            react_prompt = PromptTemplate.from_template("""
You are an AEC compliance expert assistant. You have access to tools for building analysis and compliance checking.

{full_memory_context}

AVAILABLE TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES:
1. For simple conversational questions (greetings, capability questions, general chat), skip tools entirely
2. When you don't need tools, go DIRECTLY to "Final Answer:" without any Action/Action Input
3. Only use tools for: loading building data, analyzing elements, compliance searches
4. If unsure whether to use tools, don't use them - just provide a direct answer
5. Consider the session context and conversation history when responding
6. Remember your current goals and active files from the session context

EXAMPLES:
- "Hello" → Thought: Simple greeting, no tools needed. Final Answer: Hello! I'm an AEC compliance expert...
- "What tools do you have?" → Thought: General question, no tools needed. Final Answer: I have tools for...
- "Load data from X.json" → Thought: Need to load data. Action: load_building_data...

Question: {input}
Thought: {agent_scratchpad}""")
        else:
            react_prompt = PromptTemplate.from_template("""
You are an AEC compliance expert assistant. You have access to tools for building analysis and compliance checking.

AVAILABLE TOOLS:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES:
1. For simple conversational questions (greetings, capability questions, general chat), skip tools entirely
2. When you don't need tools, go DIRECTLY to "Final Answer:" without any Action/Action Input
3. Only use tools for: loading building data, analyzing elements, compliance searches
4. If unsure whether to use tools, don't use them - just provide a direct answer

EXAMPLES:
- "Hello" → Thought: Simple greeting, no tools needed. Final Answer: Hello! I'm an AEC compliance expert...
- "What tools do you have?" → Thought: General question, no tools needed. Final Answer: I have tools for...
- "Load data from X.json" → Thought: Need to load data. Action: load_building_data...

Question: {input}
Thought: {agent_scratchpad}""")
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        executor_kwargs = {
            "agent": agent,
            "tools": self.tools,
            "verbose": self.verbose,
            "max_iterations": 15,
            "handle_parsing_errors": True
        }
        
        # Add callbacks if LangSmith is enabled
        if self.callback_manager:
            executor_kwargs["callbacks"] = self.callback_manager
            
        self.agent_executor = AgentExecutor(**executor_kwargs)
    
    @traceable(name="compliance_agent_query", project_name="AEC-Compliance-Agent")
    def process(self, query: str, clear_history: bool = False) -> Dict[str, Any]:
        """
        Process a query with the agent.
        
        Args:
            query: Natural language query
            clear_history: Whether to clear conversation history before processing
            
        Returns:
            Response with status, message, and result
        """
        try:
            # Clear memory if requested
            if clear_history and self.memory_manager:
                self.memory_manager.clear_conversation_memory()
                self.logger.info("Conversation history cleared")
            
            # Prepare invoke arguments
            invoke_kwargs = {"input": query}
            
            # Add memory context if enabled
            if self.memory_manager:
                memory_context = self.memory_manager.get_full_context_for_prompt()
                invoke_kwargs["full_memory_context"] = memory_context
            
            # Process the query with metadata for LangSmith
            run_metadata = {
                "model": self.model_name,
                "temperature": self.temperature,
                "clear_history": clear_history,
                "memory_enabled": self.enable_memory
            }
            
            if self.callback_manager:
                invoke_kwargs["config"] = {"metadata": run_metadata}
                
            result = self.agent_executor.invoke(invoke_kwargs)
            response = result.get("output", "")
            
            # Store conversation turn in memory
            if self.memory_manager and not clear_history:
                self.memory_manager.add_conversation_turn(query, response)
            
            response_data = {
                "status": "success",
                "message": "Query processed successfully",
                "response": response,
                "raw_result": result
            }
            
            # Add memory summary to response if enabled
            if self.memory_manager:
                response_data["session_summary"] = self.memory_manager.get_session_summary()
            
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            error_response = {
                "status": "error", 
                "message": f"Error: {str(e)}",
                "response": ""
            }
            
            # Still store failed conversation if memory is enabled
            if self.memory_manager and not clear_history:
                self.memory_manager.add_conversation_turn(query, f"Error: {str(e)}")
            
            return error_response
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        status = {
            "name": "AEC Compliance Agent",
            "status": "ready",
            "model": self.model_name,
            "tools": len(self.tools),
            "tool_list": [tool.name for tool in self.tools],
            "memory_enabled": self.enable_memory
        }
        
        # Add memory-specific status if enabled
        if self.memory_manager:
            status["session_id"] = self.memory_manager.get_session_id()
            status["memory_summary"] = self.memory_manager.get_session_summary()
        
        return status
    
    # Memory Management Methods
    def set_session_goal(self, goal: str, context: str = "") -> None:
        """Set the main goal for this session."""
        if self.memory_manager:
            self.memory_manager.set_session_goal(goal, context)
            self.logger.info(f"Session goal set: {goal}")
        else:
            self.logger.warning("Memory not enabled - cannot set session goal")
    
    def add_subtask(self, name: str, dependencies: Optional[list] = None) -> Optional[str]:
        """Add a subtask to track."""
        if self.memory_manager:
            return self.memory_manager.add_subtask(name, dependencies)
        else:
            self.logger.warning("Memory not enabled - cannot add subtask")
            return None
    
    def update_subtask_status(self, subtask_id: str, status: TaskStatus, notes: Optional[str] = None) -> bool:
        """Update subtask status."""
        if self.memory_manager:
            return self.memory_manager.update_subtask_status(subtask_id, status, notes)
        else:
            self.logger.warning("Memory not enabled - cannot update subtask")
            return False
    
    def get_memory_summary(self) -> Optional[Dict[str, Any]]:
        """Get a summary of the current memory state."""
        if self.memory_manager:
            return self.memory_manager.get_session_summary()
        else:
            return None
    
    def save_session(self) -> None:
        """Save the current session to disk."""
        if self.memory_manager:
            self.memory_manager.save_session()
            self.logger.info("Session saved")
        else:
            self.logger.warning("Memory not enabled - cannot save session")
    
    def clear_memory(self, conversation_only: bool = True) -> None:
        """Clear memory - conversation only or all memory."""
        if self.memory_manager:
            if conversation_only:
                self.memory_manager.clear_conversation_memory()
                self.logger.info("Conversation memory cleared")
            else:
                self.memory_manager.clear_all_memory()
                self.logger.info("All memory cleared")
        else:
            self.logger.warning("Memory not enabled - cannot clear memory")


def create_agent(
    model_name: str = "gpt-4o-mini", 
    temperature: float = 0.1, 
    verbose: bool = True,
    enable_memory: bool = True,
    memory_config: Optional[MemoryManagerConfig] = None,
    session_id: Optional[str] = None
) -> ComplianceAgent:
    """Create a compliance agent with optional memory capabilities."""
    return ComplianceAgent(
        model_name=model_name, 
        temperature=temperature, 
        verbose=verbose,
        enable_memory=enable_memory,
        memory_config=memory_config,
        session_id=session_id
    )