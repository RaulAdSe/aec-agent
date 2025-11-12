"""
Simple AEC Compliance Agent with access to all tools.

One agent that can handle building data analysis and compliance search.
"""

import logging
import os
from typing import Dict, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer

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
from .memory import ConversationHistory


class ComplianceAgent:
    """
    Simple compliance agent with access to building data and compliance tools.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1, verbose: bool = True):
        """Initialize the compliance agent."""
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory management
        self.memory = ConversationHistory(max_entries=50)
        
        # Setup LangSmith tracing
        self._setup_langsmith_tracing()
        
        # Initialize components
        self._setup_llm()
        self._setup_tools()
        self._setup_agent()
        
        self.logger.info("Compliance agent initialized with LangSmith tracing and sliding window memory")
    
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
        """Wrapper for load_building_data that returns only summary info."""
        result = load_building_data(path)
        if result.get("status") == "success":
            data = result.get("data", {})
            file_info = data.get("file_info", {})
            
            # Return only summary information
            summary = {
                "status": "success",
                "data": {
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
                "logs": [f"Building data loaded successfully from {path}"]
            }
            return summary
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
        
        # Create ReAct prompt template
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

IMPORTANT GUIDELINES:
1. For simple greetings or general questions about capabilities, provide a direct answer without using tools
2. For building analysis, load building data first using load_building_data with a valid file path
3. Only use tools when you have specific building data to analyze or compliance questions to research
4. If no building data is loaded and user asks about building elements, explain that data needs to be loaded first

EXAMPLES:
- "Hello" → Direct answer about capabilities
- "What can you do?" → Direct answer listing capabilities
- "Load building data from X" → Use load_building_data tool
- "How many spaces are there?" → Use get_all_elements if data is loaded

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
            # Clear history if requested
            if clear_history:
                self.memory.clear()
            
            # Process the query with metadata for LangSmith
            run_metadata = {
                "model": self.model_name,
                "temperature": self.temperature,
                "clear_history": clear_history,
                "memory_entries": len(self.memory.entries)
            }
            
            invoke_kwargs = {"input": query}
            if self.callback_manager:
                invoke_kwargs["config"] = {"metadata": run_metadata}
                
            result = self.agent_executor.invoke(invoke_kwargs)
            response = result.get("output", "")
            
            # Store the interaction in memory
            self.memory.add_interaction(query, response)
            
            return {
                "status": "success",
                "message": "Query processed successfully",
                "response": response,
                "raw_result": result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            # Still store the failed interaction
            self.memory.add_interaction(query, f"Error: {str(e)}")
            
            return {
                "status": "error", 
                "message": f"Error: {str(e)}",
                "response": ""
            }
    
    def clear_history(self):
        """Clear the agent's conversation history."""
        self.memory.clear()
        self.logger.info("Agent conversation history cleared")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get current memory summary."""
        return self.memory.get_summary()
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "name": "AEC Compliance Agent",
            "status": "ready",
            "model": self.model_name,
            "tools": len(self.tools),
            "tool_list": [tool.name for tool in self.tools]
        }


def create_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.1, verbose: bool = True) -> ComplianceAgent:
    """Create a compliance agent."""
    return ComplianceAgent(model_name=model_name, temperature=temperature, verbose=verbose)