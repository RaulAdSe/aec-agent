"""AEC Compliance Agent implementation."""

from typing import Dict, Any, Optional

from ..base_agent import BaseAgent
from .config import ComplianceAgentConfig
from .prompts import get_analysis_prompt, get_tool_prompt

from ...memory.conversation_history import ConversationHistory
from ...tools.compliance_toolkit import ComplianceToolkit
from ...utils.toon_converter import ToonConverter


class ComplianceAgent(BaseAgent):
    """
    AEC Compliance Agent for building code verification.
    
    Specialized agent for analyzing building data and verifying compliance
    with Spanish building codes (CTE DB-SI, CTE DB-SUA).
    """
    
    def __init__(self, config: Optional[ComplianceAgentConfig] = None):
        """Initialize the compliance agent."""
        self.compliance_config = config or ComplianceAgentConfig()
        
        # Initialize TOON converter before calling super().__init__
        self.toon_converter = ToonConverter() if self.compliance_config.use_toon else None
        
        super().__init__(self.compliance_config)
    
    def _initialize_memory(self) -> None:
        """Initialize agent memory systems."""
        self.memory = ConversationHistory(max_entries=100)
        self.logger.info("Initialized memory systems")
    
    def _initialize_tools(self) -> None:
        """Initialize compliance analysis tools."""
        self.toolkit = ComplianceToolkit(toon_converter=self.toon_converter)
        self.logger.info(f"Initialized {len(self.toolkit.get_tools())} tools")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process building data for compliance analysis.
        
        Args:
            input_data: Building data to analyze
            
        Returns:
            Compliance analysis results
        """
        try:
            # Store input in memory
            self.memory.add_building_data(input_data)
            
            # Convert to TOON format if enabled
            if self.compliance_config.use_toon and self.toon_converter:
                building_toon = self.toon_converter.json_to_toon(input_data)
                data_for_analysis = f"Building data (TOON format):\n```toon\n{building_toon}\n```"
            else:
                data_for_analysis = f"Building data (JSON):\n{input_data}"
            
            # Perform analysis (placeholder - implement your logic here)
            result = self._analyze_building(data_for_analysis)
            
            # Store results in memory
            self.memory.add_analysis_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def analyze_specific_compliance(self, building_data: Dict[str, Any], 
                                  compliance_type: str) -> Dict[str, Any]:
        """
        Analyze specific compliance requirements.
        
        Args:
            building_data: Building data to analyze
            compliance_type: Type of compliance ("fire_safety", "accessibility", "general")
            
        Returns:
            Specific compliance analysis results
        """
        # Convert data format if needed
        if self.compliance_config.use_toon and self.toon_converter:
            building_toon = self.toon_converter.json_to_toon(building_data)
            data_str = building_toon
        else:
            data_str = str(building_data)
        
        # Get appropriate prompt
        prompt = get_analysis_prompt(compliance_type, data_str)
        
        # Placeholder for actual LLM integration
        result = {
            "compliance_type": compliance_type,
            "status": "success",
            "message": "Specific analysis not implemented yet",
            "prompt_used": prompt[:100] + "...",  # First 100 chars for reference
            "data": building_data
        }
        
        return result
    
    def _analyze_building(self, building_data_str: str) -> Dict[str, Any]:
        """
        Internal method to analyze building data.
        
        This is where you'll implement your actual analysis logic.
        """
        # Placeholder implementation
        return {
            "status": "success",
            "message": "Analysis completed (placeholder)",
            "compliance_summary": {
                "fire_safety": "Not implemented",
                "accessibility": "Not implemented",
                "general": "Not implemented"
            },
            "recommendations": [
                "Implement actual analysis logic",
                "Integrate with LLM for real compliance checking",
                "Add specific building code verification"
            ]
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get agent memory summary."""
        return self.memory.get_summary()
    
    def clear_memory(self) -> None:
        """Clear agent memory."""
        self.memory.clear()
        self.logger.info("Agent memory cleared")