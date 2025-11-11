"""
Simple AEC Compliance Agent with access to all tools.

One agent that can handle building data analysis and compliance search.
"""

import logging
from typing import Dict, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

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
    
    def __init__(self, model_name: str = "gpt-5-mini", temperature: float = 1.0, verbose: bool = True):
        """Initialize the compliance agent."""
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_llm()
        self._setup_tools()
        self._setup_agent()
        
        self.logger.info("Compliance agent initialized with all tools")
    
    def _setup_llm(self):
        """Set up the language model."""
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_completion_tokens=1000
        )
    
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

WORKFLOW GUIDELINES:
1. For building analysis, load building data first using load_building_data
2. Use get_all_elements to see available element types
3. Use get_element_properties to discover property names
4. Use other tools for detailed analysis and compliance checking

Question: {input}
Thought: {agent_scratchpad}""")
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a query with the agent.
        
        Args:
            query: Natural language query
            
        Returns:
            Response with status, message, and result
        """
        try:
            result = self.agent_executor.invoke({"input": query})
            
            return {
                "status": "success",
                "message": "Query processed successfully",
                "response": result.get("output", ""),
                "raw_result": result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "status": "error", 
                "message": f"Error: {str(e)}",
                "response": ""
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "name": "AEC Compliance Agent",
            "status": "ready",
            "model": self.model_name,
            "tools": len(self.tools),
            "tool_list": [tool.name for tool in self.tools]
        }


def create_agent(model_name: str = "gpt-5-mini", temperature: float = 1.0, verbose: bool = True) -> ComplianceAgent:
    """Create a compliance agent."""
    return ComplianceAgent(model_name=model_name, temperature=temperature, verbose=verbose)