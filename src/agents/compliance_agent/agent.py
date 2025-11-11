"""AEC Compliance Agent implementation."""

from typing import Dict, Any, Optional


from ..base_agent import BaseAgent
from .config import ComplianceAgentConfig
from .prompts import get_analysis_prompt, get_tool_prompt

from ...memory.conversation_history import ConversationHistory
from ...tools.compliance_toolkit import ComplianceToolkit
from ...utils.toon_converter import ToonConverter
from ...services.ai_client import AIClient
from ...core.config import config


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
        self.ai_client = AIClient()
        self.logger.info(f"Initialized {len(self.toolkit.get_tools())} tools and OpenAI client")
    
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
        
        # Perform AI analysis
        try:
            ai_response = self.ai_client.analyze_compliance(prompt)
            
            if ai_response.get("status") == "success":
                return {
                    "compliance_type": compliance_type,
                    "status": "success",
                    "analysis": ai_response.get("analysis"),
                    "model_used": ai_response.get("model_used"),
                    "usage": ai_response.get("usage", {}),
                    "building_data": building_data
                }
            else:
                return {
                    "compliance_type": compliance_type,
                    "status": "failed",
                    "error": ai_response.get("error"),
                    "building_data": building_data
                }
        except Exception as e:
            return {
                "compliance_type": compliance_type,
                "status": "failed",
                "error": str(e),
                "building_data": building_data
            }
    
    def _analyze_building(self, building_data_str: str) -> Dict[str, Any]:
        """
        Internal method to analyze building data using AI.
        """
        try:
            # Create comprehensive prompt for general analysis
            prompt = get_analysis_prompt("general", building_data_str)
            
            # Get AI analysis
            ai_response = self.ai_client.analyze_compliance(prompt)
            
            if ai_response.get("status") == "success":
                analysis_text = ai_response.get("analysis", "")
                
                return {
                    "status": "success",
                    "analysis": analysis_text,
                    "model_used": ai_response.get("model_used"),
                    "usage": ai_response.get("usage", {}),
                    "compliance_summary": {
                        "general": "Analysis completed with AI",
                        "details": analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
                    }
                }
            else:
                # Fallback if AI fails
                self.logger.error(f"AI analysis failed: {ai_response.get('error')}")
                return {
                    "status": "partial_success",
                    "analysis": "AI analysis unavailable, using basic checks",
                    "error": ai_response.get("error"),
                    "compliance_summary": {
                        "general": "Basic checks only - AI unavailable"
                    }
                }
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "compliance_summary": {
                    "general": "Analysis failed"
                }
            }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get agent memory summary."""
        return self.memory.get_summary()
    
    def clear_memory(self) -> None:
        """Clear agent memory."""
        self.memory.clear()
        self.logger.info("Agent memory cleared")