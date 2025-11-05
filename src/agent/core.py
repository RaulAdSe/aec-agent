"""
Core AEC Compliance Agent with TOON optimization.

A clean, minimal agent implementation focused on building code compliance
verification using TOON format for optimal LLM token efficiency.
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ..utils import ToonConverter
from .tools import ComplianceToolkit
from .memory import AgentMemory


class AgentConfig(BaseModel):
    """Configuration for the AEC Compliance Agent."""
    
    model_name: str = Field(
        default="gemini-2.0-flash-exp",
        description="LLM model to use"
    )
    temperature: float = Field(
        default=0.1,
        description="LLM temperature for consistency"
    )
    max_tokens: int = Field(
        default=8192,
        description="Maximum tokens per response"
    )
    use_toon: bool = Field(
        default=True,
        description="Whether to use TOON format for data exchange"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )


class ComplianceAgent:
    """
    Clean AEC Compliance Agent for building code verification.
    
    Features:
    - TOON format optimization for token efficiency
    - Building data analysis from IFC files
    - Spanish building code compliance checking
    - Geometric calculations and spatial analysis
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the compliance agent."""
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.toon_converter = ToonConverter()
        self.memory = AgentMemory()
        self.toolkit = ComplianceToolkit(toon_converter=self.toon_converter)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        # Create agent executor
        self._create_agent()
        
        self.logger.info(f"ComplianceAgent initialized with {self.config.model_name}")
    
    def _create_agent(self):
        """Create the agent executor with tools and prompts."""
        
        # System prompt optimized for compliance verification
        system_prompt = """You are a specialized AEC (Architecture, Engineering, Construction) compliance agent.

Your role is to analyze building data and verify compliance with Spanish building codes (CTE DB-SI, CTE DB-SUA).

Key capabilities:
- Analyze IFC building data in TOON format (optimized for token efficiency)
- Perform geometric calculations for spatial compliance
- Check fire safety and evacuation requirements
- Verify accessibility and safety regulations

Data format: Building data is provided in TOON format to minimize token usage.
Always prioritize accuracy and cite specific building code references.

Available tools: {tools}
Tool names: {tool_names}

Use this format:
Thought: I need to analyze the building data for compliance
Action: [tool_name]
Action Input: [input_parameters]
Observation: [tool_output]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to provide the compliance analysis
Final Answer: [comprehensive compliance assessment]"""

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # Create agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.llm,
            tools=self.toolkit.get_tools(),
            prompt=self.prompt,
            verbose=self.config.verbose,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def analyze_building(self, building_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze building data for compliance.
        
        Args:
            building_data: Building data (will be converted to TOON if enabled)
            
        Returns:
            Compliance analysis results
        """
        try:
            # Convert to TOON format if enabled
            if self.config.use_toon:
                building_toon = self.toon_converter.json_to_toon(building_data)
                input_data = f"Building data (TOON format):\n```toon\n{building_toon}\n```"
            else:
                input_data = f"Building data (JSON):\n{building_data}"
            
            # Store in memory
            self.memory.add_building_data(building_data)
            
            # Run analysis
            result = self.agent_executor.invoke({
                "input": f"""Analyze this building for compliance with Spanish building codes.

{input_data}

Please provide:
1. Fire safety compliance analysis (CTE DB-SI)
2. Accessibility compliance analysis (CTE DB-SUA)  
3. Spatial and geometric compliance checks
4. Any non-compliance issues found
5. Recommendations for compliance improvements"""
            })
            
            # Store results in memory
            self.memory.add_analysis_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Building analysis failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def verify_specific_compliance(self, building_data: Dict[str, Any], 
                                 compliance_type: str) -> Dict[str, Any]:
        """
        Verify specific compliance requirements.
        
        Args:
            building_data: Building data to analyze
            compliance_type: Type of compliance ("fire_safety", "accessibility", "structural")
            
        Returns:
            Specific compliance verification results
        """
        compliance_prompts = {
            "fire_safety": """Verify fire safety compliance according to CTE DB-SI:
- Fire compartmentation and sector analysis
- Evacuation route calculations
- Emergency exit requirements
- Fire resistance ratings""",
            
            "accessibility": """Verify accessibility compliance according to CTE DB-SUA:
- Accessible route analysis
- Doorway width requirements
- Ramp slope calculations
- Elevator accessibility""",
            
            "structural": """Verify structural compliance:
- Load path analysis
- Clear span requirements
- Height restrictions
- Structural member sizing"""
        }
        
        if compliance_type not in compliance_prompts:
            return {"error": f"Unknown compliance type: {compliance_type}"}
        
        try:
            # Convert to TOON if enabled
            if self.config.use_toon:
                building_toon = self.toon_converter.json_to_toon(building_data)
                input_data = f"Building data (TOON format):\n```toon\n{building_toon}\n```"
            else:
                input_data = f"Building data (JSON):\n{building_data}"
            
            result = self.agent_executor.invoke({
                "input": f"""{compliance_prompts[compliance_type]}

{input_data}

Provide detailed analysis with specific code references and compliance status."""
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Compliance verification failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of agent memory and recent analyses."""
        return self.memory.get_summary()
    
    def clear_memory(self):
        """Clear agent memory for fresh start."""
        self.memory.clear()
        self.logger.info("Agent memory cleared")


# Example usage
if __name__ == "__main__":
    # Example building data for testing
    sample_building = {
        "metadata": {
            "project_name": "Test Building",
            "total_area": 500.0,
            "levels": 2
        },
        "rooms": [
            {
                "id": "R001",
                "name": "Office",
                "area": 25.0,
                "use": "office",
                "occupancy_load": 5
            }
        ],
        "doors": [
            {
                "id": "D001", 
                "width": 0.9,
                "fire_rated": True
            }
        ]
    }
    
    # Initialize agent
    agent = ComplianceAgent(AgentConfig(verbose=True))
    
    # Run analysis
    try:
        result = agent.analyze_building(sample_building)
        print(f"Analysis result: {result}")
    except Exception as e:
        print(f"Test failed: {e}")