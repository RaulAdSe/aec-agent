"""
Agent tools for AEC compliance verification.

This module provides the 6 core tools that the ReAct agent can use to
autonomously verify building code compliance.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool

from src.schemas import Project, Room, Door, ComplianceCheck
from src.calculations.geometry import (
    calculate_room_area, 
    calculate_room_centroid,
    calculate_door_clear_width,
    calculate_egress_capacity
)
from src.calculations.graph import (
    create_circulation_graph, 
    calculate_egress_distance,
    find_all_evacuation_routes,
    calculate_room_connectivity_score,
    find_critical_circulation_points,
    validate_evacuation_compliance,
    get_room_adjacency_list
)


# Global variables to hold project data and RAG manager
_project_data: Optional[Project] = None
_rag_manager = None


def load_project_data(json_path: Path) -> Project:
    """
    Load project data from JSON file.
    
    Args:
        json_path: Path to the project JSON file
        
    Returns:
        Project object
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON data is invalid
    """
    global _project_data
    
    if not json_path.exists():
        raise FileNotFoundError(f"Project file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        _project_data = Project(**data)
        return _project_data
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in project file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading project data: {e}")


def set_vectorstore_manager(rag_manager):
    """
    Set the RAG vectorstore manager for normativa queries.
    
    Args:
        rag_manager: VectorstoreManager instance
    """
    global _rag_manager
    _rag_manager = rag_manager


@tool
def get_room_info(room_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a room.
    
    Args:
        room_id: The ID of the room (e.g., 'R001')
    
    Returns:
        Dictionary with room properties including area, use, level, and geometric data
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    room = _project_data.get_room_by_id(room_id)
    if room is None:
        return {"error": f"Room {room_id} not found in project"}
    
    # Calculate additional geometric properties
    area = calculate_room_area(room)
    centroid = calculate_room_centroid(room)
    occupancy_capacity = calculate_egress_capacity(area, room.use)
    
    return {
        "id": room.id,
        "name": room.name,
        "area_sqm": area,
        "use": room.use,
        "level": room.level,
        "occupancy_load": room.occupancy_load,
        "fire_rating": room.fire_rating,
        "centroid": {
            "x": centroid.x if centroid else None,
            "y": centroid.y if centroid else None
        } if centroid else None,
        "calculated_occupancy_capacity": occupancy_capacity,
        "has_boundary": room.boundary is not None and room.boundary.points is not None
    }


@tool
def get_door_info(door_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a door.
    
    Args:
        door_id: The ID of the door (e.g., 'D001')
    
    Returns:
        Dictionary with door properties including dimensions, type, and compliance data
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    door = _project_data.get_door_by_id(door_id)
    if door is None:
        return {"error": f"Door {door_id} not found in project"}
    
    # Calculate additional properties
    clear_width = calculate_door_clear_width(door)
    door_area = (door.width_mm * door.height_mm) / 1000000.0  # Convert to sqm
    
    return {
        "id": door.id,
        "name": door.name,
        "width_mm": door.width_mm,
        "height_mm": door.height_mm,
        "clear_width_mm": clear_width,
        "area_sqm": door_area,
        "door_type": door.door_type,
        "fire_rating": door.fire_rating,
        "position": {
            "x": door.position.x,
            "y": door.position.y,
            "z": door.position.z
        },
        "from_room": door.from_room,
        "to_room": door.to_room,
        "is_emergency_exit": door.is_emergency_exit,
        "is_accessible": door.is_accessible
    }


@tool
def list_all_doors() -> List[Dict[str, Any]]:
    """
    List all doors in the project with basic information.
    
    Returns:
        List of dictionaries with door information
    """
    if _project_data is None:
        return [{"error": "No project data loaded. Call load_project_data() first."}]
    
    doors = _project_data.get_all_doors()
    
    door_list = []
    for door in doors:
        door_list.append({
            "id": door.id,
            "name": door.name,
            "width_mm": door.width_mm,
            "door_type": door.door_type,
            "from_room": door.from_room,
            "to_room": door.to_room,
            "is_emergency_exit": door.is_emergency_exit,
            "fire_rating": door.fire_rating
        })
    
    return door_list


@tool
def check_door_width_compliance(door_id: str) -> Dict[str, Any]:
    """
    Check if a door meets minimum width requirements for compliance.
    
    Args:
        door_id: The ID of the door to check
    
    Returns:
        Dictionary with compliance check results
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    door = _project_data.get_door_by_id(door_id)
    if door is None:
        return {"error": f"Door {door_id} not found in project"}
    
    # Get door information
    door_info = get_door_info.invoke({"door_id": door_id})
    if "error" in door_info:
        return door_info
    
    clear_width = door_info["clear_width_mm"]
    
    # Define minimum width requirements (in mm)
    min_widths = {
        "single": 800,      # Standard door
        "double": 1200,     # Double door (600mm per leaf)
        "sliding": 800,     # Sliding door
        "revolving": 800,   # Revolving door
        "fire_door": 800,   # Fire door
        "emergency_exit": 900  # Emergency exit door
    }
    
    # Special case for emergency exits
    if door.is_emergency_exit:
        min_width = 900  # CTE requirement for emergency exits
    else:
        min_width = min_widths.get(door.door_type, 800)
    
    is_compliant = clear_width >= min_width
    
    return {
        "door_id": door_id,
        "door_type": door.door_type,
        "clear_width_mm": clear_width,
        "required_width_mm": min_width,
        "is_compliant": is_compliant,
        "compliance_status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
        "message": f"Door {door_id} {'meets' if is_compliant else 'does not meet'} minimum width requirement of {min_width}mm",
        "regulation_reference": "CTE DB-SI Section 3.1" if door.is_emergency_exit else "CTE DB-SUA Section 2.1"
    }


@tool
def query_normativa(question: str) -> Dict[str, Any]:
    """
    Query building codes and regulations using RAG system.
    
    Args:
        question: Question about building codes, regulations, or compliance requirements
    
    Returns:
        Dictionary with answer and source information
    """
    if _rag_manager is None:
        return {
            "error": "RAG system not initialized. Call set_vectorstore_manager() first.",
            "answer": "I cannot query building codes without the RAG system being initialized.",
            "sources": []
        }
    
    try:
        # Use the RAG manager to query the normativa
        result = _rag_manager.query(question)
        
        return {
            "question": question,
            "answer": result.get("answer", "No answer found"),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "regulation_references": result.get("regulation_references", [])
        }
    
    except Exception as e:
        return {
            "error": f"Error querying normativa: {str(e)}",
            "question": question,
            "answer": "Unable to retrieve information from building codes.",
            "sources": []
        }


@tool
def calculate_egress_distance(room_id: str) -> Dict[str, Any]:
    """
    Calculate evacuation distance from a room to the nearest exit.
    
    Args:
        room_id: The ID of the room to calculate egress distance for
    
    Returns:
        Dictionary with egress distance information and compliance status
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    room = _project_data.get_room_by_id(room_id)
    if room is None:
        return {"error": f"Room {room_id} not found in project"}
    
    try:
        # Create circulation graph and calculate egress distance
        graph = create_circulation_graph(_project_data)
        egress_info = graph.calculate_egress_distance(room_id)
        
        if egress_info.get("error"):
            return egress_info
        
        # Get room information for compliance checking
        room_info = get_room_info.invoke({"room_id": room_id})
        if "error" in room_info:
            return room_info
        
        # Define maximum egress distances based on room use (in meters)
        max_distances = {
            "residential": 30,     # Residential buildings
            "commercial": 25,      # Commercial buildings
            "retail": 25,          # Retail spaces
            "office": 25,          # Office buildings
            "assembly": 20,        # Assembly spaces
            "storage": 30,         # Storage areas
            "restroom": 25,        # Restrooms
            "meeting": 25,         # Meeting rooms
            "reception": 25,       # Reception areas
        }
        
        room_use = room_info.get("use", "commercial")
        max_distance = max_distances.get(room_use, 25)  # Default 25m
        
        distance = egress_info["distance"]
        is_compliant = distance <= max_distance
        
        return {
            "room_id": room_id,
            "room_use": room_use,
            "egress_distance_m": distance,
            "max_allowed_distance_m": max_distance,
            "is_compliant": is_compliant,
            "compliance_status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
            "exit_room_id": egress_info["exit_room_id"],
            "path": egress_info["path"],
            "is_accessible": egress_info["is_accessible"],
            "message": f"Room {room_id} egress distance is {distance:.1f}m, {'within' if is_compliant else 'exceeds'} the maximum allowed {max_distance}m for {room_use} use",
            "regulation_reference": "CTE DB-SI Section 3.2 - Egress distances"
        }
    
    except Exception as e:
        return {
            "error": f"Error calculating egress distance: {str(e)}",
            "room_id": room_id
        }


@tool
def analyze_building_circulation() -> Dict[str, Any]:
    """
    Analyze overall building circulation and identify potential issues.
    
    Returns:
        Dictionary with comprehensive circulation analysis
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    try:
        # Get critical circulation points
        critical_points = find_critical_circulation_points(_project_data)
        
        # Get evacuation compliance
        compliance = validate_evacuation_compliance(_project_data)
        
        # Get room adjacency
        adjacency = get_room_adjacency_list(_project_data)
        
        return {
            "critical_circulation_points": critical_points,
            "evacuation_compliance": compliance,
            "room_adjacency": adjacency,
            "analysis_summary": {
                "total_rooms": len(adjacency),
                "critical_rooms": critical_points["critical_room_count"],
                "compliance_rate": compliance["compliance_rate"],
                "overall_circulation_health": "good" if compliance["compliance_rate"] > 0.9 and critical_points["critical_room_count"] < 3 else "needs_attention"
            }
        }
    
    except Exception as e:
        return {"error": f"Error analyzing circulation: {str(e)}"}


@tool
def find_all_evacuation_routes_tool() -> Dict[str, Any]:
    """
    Find evacuation routes for all rooms in the building.
    
    Returns:
        Dictionary with evacuation routes for all rooms
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    try:
        routes = find_all_evacuation_routes(_project_data)
        return routes
    
    except Exception as e:
        return {"error": f"Error finding evacuation routes: {str(e)}"}


@tool
def check_room_connectivity(room_id: str) -> Dict[str, Any]:
    """
    Check how well-connected a specific room is to the rest of the building.
    
    Args:
        room_id: The ID of the room to analyze
    
    Returns:
        Dictionary with room connectivity analysis
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    try:
        connectivity = calculate_room_connectivity_score(_project_data, room_id)
        return connectivity
    
    except Exception as e:
        return {"error": f"Error analyzing room connectivity: {str(e)}"}


@tool
def calculate_occupancy_load(room_id: str) -> Dict[str, Any]:
    """
    Calculate the occupancy load and egress capacity for a room.
    
    Args:
        room_id: The ID of the room
    
    Returns:
        Dictionary with occupancy analysis
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    room = _project_data.get_room_by_id(room_id)
    if room is None:
        return {"error": f"Room {room_id} not found in project"}
    
    try:
        # Calculate room area
        area = calculate_room_area(room)
        
        # Calculate occupancy based on room use
        occupancy_factors = {
            "residential": 0.05,  # 1 person per 20 sqm
            "commercial": 0.1,    # 1 person per 10 sqm
            "retail": 0.2,        # 1 person per 5 sqm
            "office": 0.1,        # 1 person per 10 sqm
            "assembly": 0.5,      # 1 person per 2 sqm
            "storage": 0.02,      # 1 person per 50 sqm
            "restroom": 0.1,      # 1 person per 10 sqm
            "meeting": 0.2,       # 1 person per 5 sqm
            "reception": 0.1,     # 1 person per 10 sqm
        }
        
        factor = occupancy_factors.get(room.use, 0.1)
        calculated_occupancy = max(1, int(area * factor))
        
        # Calculate required egress width (5mm per person minimum)
        required_egress_width = calculated_occupancy * 5  # mm
        
        # Find doors connected to this room
        connected_doors = []
        for level in _project_data.levels:
            for door in level.doors:
                if door.from_room == room_id or door.to_room == room_id:
                    connected_doors.append({
                        "door_id": door.id,
                        "width_mm": door.width_mm,
                        "is_emergency_exit": door.is_emergency_exit
                    })
        
        total_egress_width = sum(door["width_mm"] for door in connected_doors 
                                if door["is_emergency_exit"])
        
        return {
            "room_id": room_id,
            "area_sqm": area,
            "room_use": room.use,
            "occupancy_factor": factor,
            "calculated_occupancy": calculated_occupancy,
            "required_egress_width_mm": required_egress_width,
            "available_egress_width_mm": total_egress_width,
            "egress_adequate": total_egress_width >= required_egress_width,
            "connected_doors": connected_doors,
            "egress_deficit_mm": max(0, required_egress_width - total_egress_width)
        }
    
    except Exception as e:
        return {"error": f"Error calculating occupancy load: {str(e)}"}


@tool
def analyze_door_compliance_comprehensive() -> Dict[str, Any]:
    """
    Comprehensive analysis of all doors for various compliance requirements.
    
    Returns:
        Dictionary with comprehensive door compliance analysis
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    try:
        doors = _project_data.get_all_doors()
        
        compliance_results = {
            "total_doors": len(doors),
            "width_compliance": {"compliant": 0, "non_compliant": []},
            "emergency_exit_compliance": {"compliant": 0, "non_compliant": []},
            "accessibility_compliance": {"compliant": 0, "non_compliant": []},
            "overall_compliance_rate": 0
        }
        
        for door in doors:
            door_info = {
                "door_id": door.id,
                "width_mm": door.width_mm,
                "door_type": door.door_type,
                "is_emergency_exit": door.is_emergency_exit
            }
            
            # Width compliance check
            min_width = 900 if door.is_emergency_exit else 800
            if door.width_mm >= min_width:
                compliance_results["width_compliance"]["compliant"] += 1
            else:
                door_info["width_deficit"] = min_width - door.width_mm
                compliance_results["width_compliance"]["non_compliant"].append(door_info.copy())
            
            # Emergency exit compliance (minimum 900mm)
            if door.is_emergency_exit:
                if door.width_mm >= 900:
                    compliance_results["emergency_exit_compliance"]["compliant"] += 1
                else:
                    door_info["emergency_deficit"] = 900 - door.width_mm
                    compliance_results["emergency_exit_compliance"]["non_compliant"].append(door_info.copy())
            
            # Accessibility compliance (minimum 800mm clear width)
            clear_width = calculate_door_clear_width(door)
            if clear_width >= 800:
                compliance_results["accessibility_compliance"]["compliant"] += 1
            else:
                door_info["accessibility_deficit"] = 800 - clear_width
                compliance_results["accessibility_compliance"]["non_compliant"].append(door_info.copy())
        
        # Calculate overall compliance rate
        total_checks = len(doors) * 3  # 3 compliance checks per door
        total_compliant = (compliance_results["width_compliance"]["compliant"] + 
                          compliance_results["emergency_exit_compliance"]["compliant"] + 
                          compliance_results["accessibility_compliance"]["compliant"])
        
        compliance_results["overall_compliance_rate"] = total_compliant / total_checks if total_checks > 0 else 0
        
        return compliance_results
    
    except Exception as e:
        return {"error": f"Error analyzing door compliance: {str(e)}"}


# Additional utility functions for the agent

def get_project_summary() -> Dict[str, Any]:
    """
    Get a summary of the loaded project.
    
    Returns:
        Dictionary with project summary information
    """
    if _project_data is None:
        return {"error": "No project data loaded"}
    
    rooms = _project_data.get_all_rooms()
    doors = _project_data.get_all_doors()
    walls = _project_data.get_all_walls()
    
    # Calculate total area
    total_area = sum(calculate_room_area(room) for room in rooms)
    
    # Count rooms by use
    room_uses = {}
    for room in rooms:
        use = room.use
        room_uses[use] = room_uses.get(use, 0) + 1
    
    # Count doors by type
    door_types = {}
    for door in doors:
        door_type = door.door_type
        door_types[door_type] = door_types.get(door_type, 0) + 1
    
    return {
        "project_name": _project_data.metadata.project_name,
        "building_type": _project_data.metadata.building_type,
        "total_levels": len(_project_data.levels),
        "total_rooms": len(rooms),
        "total_doors": len(doors),
        "total_walls": len(walls),
        "total_area_sqm": total_area,
        "room_uses": room_uses,
        "door_types": door_types,
        "levels": [level.name for level in _project_data.levels]
    }


def get_available_tools() -> List[Dict[str, str]]:
    """
    Get list of available agent tools with descriptions.
    
    Returns:
        List of tool information dictionaries
    """
    return [
        {
            "name": "get_room_info",
            "description": "Get detailed information about a room including area, use, and geometric properties",
            "parameters": ["room_id: str"]
        },
        {
            "name": "get_door_info", 
            "description": "Get detailed information about a door including dimensions and compliance data",
            "parameters": ["door_id: str"]
        },
        {
            "name": "list_all_doors",
            "description": "List all doors in the project with basic information",
            "parameters": []
        },
        {
            "name": "check_door_width_compliance",
            "description": "Check if a door meets minimum width requirements for compliance",
            "parameters": ["door_id: str"]
        },
        {
            "name": "query_normativa",
            "description": "Query building codes and regulations using RAG system",
            "parameters": ["question: str"]
        },
        {
            "name": "calculate_egress_distance",
            "description": "Calculate evacuation distance from a room to the nearest exit",
            "parameters": ["room_id: str"]
        },
        {
            "name": "analyze_building_circulation",
            "description": "Analyze overall building circulation and identify potential issues",
            "parameters": []
        },
        {
            "name": "find_all_evacuation_routes_tool",
            "description": "Find evacuation routes for all rooms in the building",
            "parameters": []
        },
        {
            "name": "check_room_connectivity",
            "description": "Check how well-connected a specific room is to the rest of the building",
            "parameters": ["room_id: str"]
        },
        {
            "name": "calculate_occupancy_load",
            "description": "Calculate the occupancy load and egress capacity for a room",
            "parameters": ["room_id: str"]
        },
        {
            "name": "analyze_door_compliance_comprehensive",
            "description": "Comprehensive analysis of all doors for various compliance requirements",
            "parameters": []
        }
    ]
