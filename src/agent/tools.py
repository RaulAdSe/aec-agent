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
from src.calculations.graph import create_circulation_graph, calculate_egress_distance


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
        }
    ]
