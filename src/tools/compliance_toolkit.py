"""Compliance analysis tools for AEC building verification."""

import logging
from typing import Dict, List, Any, Optional

from ..utils.toon_converter import ToonConverter


class ComplianceToolkit:
    """
    Toolkit for AEC compliance analysis tools.
    
    Provides specialized tools for building code verification,
    geometric calculations, and compliance checking.
    """
    
    def __init__(self, toon_converter: Optional[ToonConverter] = None):
        """Initialize the compliance toolkit."""
        self.toon_converter = toon_converter or ToonConverter()
        self.logger = logging.getLogger(__name__)
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Create the compliance analysis tools."""
        tools = [
            {
                "name": "example_compliance_tool",
                "description": "Example tool for compliance analysis - replace with your implementation",
                "func": self._example_compliance_tool
            },
            {
                "name": "convert_data_format", 
                "description": "Convert building data between JSON and TOON formats",
                "func": self._convert_data_format_tool
            }
        ]
        return tools
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools."""
        return self.tools
    
    def _example_compliance_tool(self, input_params: str) -> str:
        """
        Example compliance tool - implement your own tools here.
        
        This is where you'll add tools like:
        - Door width analysis
        - Evacuation route calculation
        - Fire compartment analysis
        - Accessibility compliance checking
        """
        return f"Example compliance tool executed with: {input_params}"
    
    def _convert_data_format_tool(self, input_params: str) -> str:
        """Tool for converting between JSON and TOON formats."""
        try:
            params = self._parse_tool_input(input_params)
            
            data = params.get("data")
            target_format = params.get("format", "toon").lower()
            
            if not data:
                return "Error: No data provided for conversion"
            
            if target_format == "toon":
                result = self.toon_converter.json_to_toon(data)
                savings = self.toon_converter.get_token_savings(data)
                return f"TOON format:\n{result}\n\nToken savings: {savings.get('savings_percent', 0):.1f}%"
            
            elif target_format == "json":
                result = self.toon_converter.toon_to_json(data)
                return f"JSON format:\n{result}"
            
            else:
                return f"Error: Unsupported format '{target_format}'. Use 'json' or 'toon'"
            
        except Exception as e:
            self.logger.error(f"Format conversion failed: {e}")
            return f"Error converting format: {e}"
    
    def _parse_tool_input(self, input_str: str) -> Dict[str, Any]:
        """Parse tool input string to parameters dictionary."""
        try:
            import json
            return json.loads(input_str)
        except:
            return {"input": input_str}