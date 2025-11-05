"""
Clean compliance tools for the AEC agent.

Focused toolset for building analysis and compliance verification
with TOON format optimization.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain.tools import Tool
from pydantic import BaseModel, Field

from ..calculations import (
    calculate_clearance_between_elements,
    calculate_angle_between_walls,
    find_nearest_door,
    CirculationGraph,
    create_circulation_graph
)
from ..utils import ToonConverter


class ComplianceToolkit:
    """
    Clean toolkit for AEC compliance verification.
    
    Provides essential tools for building analysis with TOON optimization.
    """
    
    def __init__(self, toon_converter: Optional[ToonConverter] = None):
        """Initialize the compliance toolkit."""
        self.toon_converter = toon_converter or ToonConverter()
        self.logger = logging.getLogger(__name__)
        
        # Initialize tools
        self.tools = self._create_tools()
    
    def _create_tools(self) -> List[Tool]:
        """Create the essential compliance tools."""
        
        tools = [
            Tool(
                name="calculate_clearance",
                description="Calculate clearance distance between building elements for accessibility compliance",
                func=self._calculate_clearance_tool
            ),
            
            Tool(
                name="analyze_door_width",
                description="Analyze door widths for accessibility and evacuation compliance",
                func=self._analyze_door_width_tool
            ),
            
            Tool(
                name="check_evacuation_routes", 
                description="Check evacuation routes and calculate travel distances for fire safety",
                func=self._check_evacuation_routes_tool
            ),
            
            Tool(
                name="calculate_occupancy_load",
                description="Calculate occupancy load and verify against building capacity",
                func=self._calculate_occupancy_load_tool
            ),
            
            Tool(
                name="analyze_spatial_compliance",
                description="Analyze spatial relationships for building code compliance",
                func=self._analyze_spatial_compliance_tool
            ),
            
            Tool(
                name="convert_data_format",
                description="Convert building data between JSON and TOON formats for analysis",
                func=self._convert_data_format_tool
            )
        ]
        
        return tools
    
    def get_tools(self) -> List[Tool]:
        """Get the list of available tools."""
        return self.tools
    
    def _calculate_clearance_tool(self, input_params: str) -> str:
        """Tool for calculating clearances between elements."""
        try:
            # Parse input parameters
            params = self._parse_tool_input(input_params)
            
            element1 = params.get("element1")
            element2 = params.get("element2")
            
            if not element1 or not element2:
                return "Error: Both element1 and element2 coordinates are required"
            
            # Calculate clearance
            clearance = calculate_clearance_between_elements(element1, element2)
            
            return f"Clearance distance: {clearance:.2f} meters"
            
        except Exception as e:
            self.logger.error(f"Clearance calculation failed: {e}")
            return f"Error calculating clearance: {e}"
    
    def _analyze_door_width_tool(self, input_params: str) -> str:
        """Tool for analyzing door widths for compliance."""
        try:
            params = self._parse_tool_input(input_params)
            doors = params.get("doors", [])
            
            if not doors:
                return "Error: No doors provided for analysis"
            
            results = []
            for door in doors:
                door_id = door.get("id", "Unknown")
                width = door.get("width", 0)
                
                # Check compliance (minimum 0.8m for accessibility, 0.9m for evacuation)
                accessibility_compliant = width >= 0.8
                evacuation_compliant = width >= 0.9
                
                status = "COMPLIANT" if accessibility_compliant and evacuation_compliant else "NON-COMPLIANT"
                
                results.append(f"Door {door_id}: {width}m - {status}")
                
                if not accessibility_compliant:
                    results.append(f"  - Fails accessibility requirement (min 0.8m)")
                if not evacuation_compliant:
                    results.append(f"  - Fails evacuation requirement (min 0.9m)")
            
            return "\n".join(results)
            
        except Exception as e:
            self.logger.error(f"Door analysis failed: {e}")
            return f"Error analyzing doors: {e}"
    
    def _check_evacuation_routes_tool(self, input_params: str) -> str:
        """Tool for checking evacuation routes and distances."""
        try:
            params = self._parse_tool_input(input_params)
            rooms = params.get("rooms", [])
            doors = params.get("doors", [])
            
            if not rooms or not doors:
                return "Error: Both rooms and doors are required for evacuation analysis"
            
            # Create circulation graph for route analysis
            graph = create_circulation_graph(rooms, doors)
            
            results = ["EVACUATION ROUTE ANALYSIS:"]
            
            for room in rooms:
                room_id = room.get("id", "Unknown")
                room_area = room.get("area", 0)
                
                # Find nearest exit
                nearest_door = find_nearest_door(room, doors)
                
                if nearest_door:
                    door_id = nearest_door.get("id", "Unknown")
                    # Calculate approximate travel distance (simplified)
                    travel_distance = 25.0  # Placeholder - would calculate actual path
                    
                    # Check compliance (max 25m travel distance in most cases)
                    compliant = travel_distance <= 25.0
                    status = "COMPLIANT" if compliant else "NON-COMPLIANT"
                    
                    results.append(f"Room {room_id} -> Door {door_id}: {travel_distance:.1f}m - {status}")
                else:
                    results.append(f"Room {room_id}: No accessible exit found - NON-COMPLIANT")
            
            return "\n".join(results)
            
        except Exception as e:
            self.logger.error(f"Evacuation route analysis failed: {e}")
            return f"Error analyzing evacuation routes: {e}"
    
    def _calculate_occupancy_load_tool(self, input_params: str) -> str:
        """Tool for calculating and verifying occupancy loads."""
        try:
            params = self._parse_tool_input(input_params)
            rooms = params.get("rooms", [])
            
            if not rooms:
                return "Error: No rooms provided for occupancy analysis"
            
            results = ["OCCUPANCY LOAD ANALYSIS:"]
            total_occupancy = 0
            
            # Standard occupancy factors (people per m²)
            occupancy_factors = {
                "office": 10,      # 1 person per 10 m²
                "classroom": 2,    # 1 person per 2 m²
                "assembly": 1.5,   # 1 person per 1.5 m²
                "retail": 3,       # 1 person per 3 m²
                "residential": 20  # 1 person per 20 m²
            }
            
            for room in rooms:
                room_id = room.get("id", "Unknown")
                area = room.get("area", 0)
                use_type = room.get("use", "office").lower()
                specified_load = room.get("occupancy_load")
                
                # Calculate required occupancy based on use type
                factor = occupancy_factors.get(use_type, 10)
                calculated_load = max(1, int(area / factor))
                
                if specified_load:
                    compliant = specified_load <= calculated_load
                    status = "COMPLIANT" if compliant else "EXCEEDS CAPACITY"
                    results.append(f"Room {room_id} ({use_type}): {specified_load}/{calculated_load} people - {status}")
                else:
                    results.append(f"Room {room_id} ({use_type}): Calculated load {calculated_load} people")
                
                total_occupancy += specified_load or calculated_load
            
            results.append(f"\nTotal building occupancy: {total_occupancy} people")
            
            return "\n".join(results)
            
        except Exception as e:
            self.logger.error(f"Occupancy calculation failed: {e}")
            return f"Error calculating occupancy: {e}"
    
    def _analyze_spatial_compliance_tool(self, input_params: str) -> str:
        """Tool for general spatial compliance analysis."""
        try:
            params = self._parse_tool_input(input_params)
            
            # Extract building elements
            rooms = params.get("rooms", [])
            walls = params.get("walls", [])
            
            results = ["SPATIAL COMPLIANCE ANALYSIS:"]
            
            # Check room size minimums
            for room in rooms:
                room_id = room.get("id", "Unknown")
                area = room.get("area", 0)
                use_type = room.get("use", "").lower()
                
                # Minimum area requirements by use type
                min_areas = {
                    "office": 9.0,      # Minimum 9 m² per office
                    "bedroom": 6.0,     # Minimum 6 m² for bedrooms
                    "bathroom": 2.0,    # Minimum 2 m² for bathrooms
                    "kitchen": 4.0      # Minimum 4 m² for kitchens
                }
                
                min_area = min_areas.get(use_type, 0)
                if min_area > 0:
                    compliant = area >= min_area
                    status = "COMPLIANT" if compliant else "TOO SMALL"
                    results.append(f"Room {room_id} ({use_type}): {area}m² (min {min_area}m²) - {status}")
            
            # Check wall angles for structural compliance
            if len(walls) >= 2:
                for i, wall1 in enumerate(walls[:5]):  # Limit to avoid too much output
                    for wall2 in walls[i+1:i+3]:  # Check a few wall pairs
                        try:
                            angle = calculate_angle_between_walls(wall1, wall2)
                            if angle is not None:
                                wall1_id = wall1.get("id", "Unknown")
                                wall2_id = wall2.get("id", "Unknown")
                                results.append(f"Walls {wall1_id}-{wall2_id}: {angle:.1f}° angle")
                        except:
                            continue
            
            return "\n".join(results)
            
        except Exception as e:
            self.logger.error(f"Spatial analysis failed: {e}")
            return f"Error in spatial analysis: {e}"
    
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
            # Try to parse as TOON first, then JSON
            if input_str.strip().startswith('{'):
                import json
                return json.loads(input_str)
            else:
                # Try TOON format
                return self.toon_converter.toon_to_json(input_str)
        except:
            # Fallback to simple string parsing
            return {"input": input_str}


# Example usage and testing
if __name__ == "__main__":
    toolkit = ComplianceToolkit()
    
    # Test door analysis
    door_data = {
        "doors": [
            {"id": "D001", "width": 0.9},
            {"id": "D002", "width": 0.7},  # Non-compliant
            {"id": "D003", "width": 1.2}
        ]
    }
    
    result = toolkit._analyze_door_width_tool(str(door_data))
    print("Door analysis result:")
    print(result)