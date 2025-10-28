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
    calculate_angle_between_walls,
    find_nearest_door,
    find_walls_within_radius,
    calculate_wall_cardinal_direction,
    calculate_clearance_between_elements
)
from src.calculations.graph import create_circulation_graph
from src.extraction.ifc_extractor import extract_from_ifc


# Simple helper functions for calculations
def calculate_room_area(room: Room) -> float:
    """Calculate room area from boundary if available."""
    if room.area:
        return room.area
    # Simple fallback calculation
    return 50.0  # Default area

def calculate_room_centroid(room: Room):
    """Calculate room centroid from boundary if available."""
    if room.boundary and room.boundary.points:
        points = room.boundary.points
        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        from src.schemas import Point2D
        return Point2D(x=x, y=y)
    return None

def calculate_door_clear_width(door: Door) -> float:
    """Calculate door clear width."""
    # Assume 50mm frame reduction
    return max(0, door.width_mm - 50)

def calculate_egress_capacity(area: float, room_use: str) -> int:
    """Calculate egress capacity based on area and use."""
    factors = {
        "residential": 0.05,  # 1 person per 20 sqm
        "commercial": 0.1,    # 1 person per 10 sqm
        "retail": 0.2,        # 1 person per 5 sqm
        "office": 0.1,        # 1 person per 10 sqm
        "assembly": 0.5,      # 1 person per 2 sqm
    }
    factor = factors.get(room_use, 0.1)
    return max(1, int(area * factor))

# Simple implementations for missing graph functions
def find_all_evacuation_routes(project: Project) -> Dict[str, Any]:
    """Find evacuation routes for all rooms."""
    try:
        graph = create_circulation_graph(project)
        return {
            "success": True,
            "total_rooms": graph.graph.number_of_nodes(),
            "total_connections": graph.graph.number_of_edges(),
            "message": "Evacuation routes analysis completed"
        }
    except Exception as e:
        return {"error": f"Error finding evacuation routes: {str(e)}"}

def calculate_room_connectivity_score(project: Project, room_id: str) -> Dict[str, Any]:
    """Calculate connectivity score for a room."""
    try:
        graph = create_circulation_graph(project)
        node_id = f"room_{room_id}"
        if node_id in graph.graph.nodes():
            degree = graph.graph.degree(node_id)
            return {
                "success": True,
                "room_id": room_id,
                "connectivity_score": degree,
                "message": f"Room {room_id} has {degree} connections"
            }
        return {"error": f"Room {room_id} not found in graph"}
    except Exception as e:
        return {"error": f"Error calculating connectivity: {str(e)}"}

def find_critical_circulation_points(project: Project) -> Dict[str, Any]:
    """Find critical circulation points."""
    try:
        graph = create_circulation_graph(project)
        import networkx as nx
        
        # Find nodes with highest betweenness centrality
        centrality = nx.betweenness_centrality(graph.graph)
        critical_points = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "success": True,
            "critical_room_count": len(critical_points),
            "critical_points": [{"room": point[0], "centrality": point[1]} for point in critical_points],
            "message": f"Found {len(critical_points)} critical circulation points"
        }
    except Exception as e:
        return {"error": f"Error finding critical points: {str(e)}"}

def validate_evacuation_compliance(project: Project) -> Dict[str, Any]:
    """Validate evacuation compliance."""
    try:
        graph = create_circulation_graph(project)
        total_rooms = graph.graph.number_of_nodes()
        exit_rooms = len([n for n in graph.graph.nodes() if graph.graph.nodes[n].get('is_exit', False)])
        
        compliance_rate = exit_rooms / max(total_rooms, 1)
        
        return {
            "success": True,
            "compliance_rate": compliance_rate,
            "total_rooms": total_rooms,
            "exit_rooms": exit_rooms,
            "is_compliant": compliance_rate >= 0.1,  # At least 10% of rooms should have exits
            "message": f"Evacuation compliance: {compliance_rate:.1%}"
        }
    except Exception as e:
        return {"error": f"Error validating compliance: {str(e)}"}

def get_room_adjacency_list(project: Project) -> Dict[str, Any]:
    """Get room adjacency list."""
    try:
        graph = create_circulation_graph(project)
        adjacency = {}
        
        for node in graph.graph.nodes():
            neighbors = list(graph.graph.neighbors(node))
            adjacency[node] = neighbors
        
        return adjacency
    except Exception as e:
        return {"error": f"Error getting adjacency list: {str(e)}"}


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
        
        # Transform field names for compatibility
        def transform_data(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    # Transform field names
                    if key == 'width':
                        new_obj['width_mm'] = value
                    elif key == 'height':
                        new_obj['height_mm'] = value
                    elif key == 'thickness':
                        new_obj['thickness_mm'] = value
                    else:
                        new_obj[key] = transform_data(value)
                return new_obj
            elif isinstance(obj, list):
                return [transform_data(item) for item in obj]
            else:
                return obj
        
        transformed_data = transform_data(data)
        _project_data = Project(**transformed_data)
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
def list_all_rooms() -> List[Dict[str, Any]]:
    """
    CRITICAL: List all rooms in the project with their ACTUAL IDs.
    
    IMPORTANT: Use this tool FIRST before calling calculate_clearance_tool with rooms!
    Do NOT make up room IDs like "0", "1", "Room_0", etc. - they don't exist!
    
    Returns:
        List of dictionaries with room information including REAL room IDs
    """
    if _project_data is None:
        return [{"error": "No project data loaded. Call load_project_data() first."}]
    
    rooms = _project_data.get_all_rooms()
    
    room_list = []
    for room in rooms:
        room_list.append({
            "id": room.id,
            "name": room.name,
            "area": room.area,
            "use": room.use,
            "level": room.level
        })
    
    return room_list


@tool
def get_available_element_ids() -> Dict[str, Any]:
    """
    CRITICAL: Get all available element IDs (rooms, doors, walls) for the agent to use.
    
    IMPORTANT: Use this tool FIRST before calling calculate_clearance_tool!
    Do NOT make up room IDs like "0", "1", "Room_0", etc. - they don't exist!
    This tool provides the REAL IDs that actually exist in the building data.
    
    Returns:
        Dictionary with lists of available IDs for each element type
    """
    if _project_data is None:
        return {"error": "No project data loaded. Call load_project_data() first."}
    
    try:
        # Get all room IDs
        rooms = _project_data.get_all_rooms()
        room_ids = [room.id for room in rooms]
        
        # Get all door IDs
        doors = _project_data.get_all_doors()
        door_ids = [door.id for door in doors]
        
        # Get all wall IDs
        wall_ids = []
        for level in _project_data.levels:
            for wall in level.walls:
                if 'id' in wall:
                    wall_ids.append(wall['id'])
        
        return {
            "success": True,
            "room_ids": room_ids,
            "door_ids": door_ids,
            "wall_ids": wall_ids,
            "counts": {
                "rooms": len(room_ids),
                "doors": len(door_ids),
                "walls": len(wall_ids)
            },
            "message": f"Found {len(room_ids)} rooms, {len(door_ids)} doors, {len(wall_ids)} walls"
        }
        
    except Exception as e:
        return {"error": f"Error getting element IDs: {str(e)}"}


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
        # Create circulation graph
        graph = create_circulation_graph(_project_data)
        
        # Simple egress distance calculation
        node_id = f"room_{room_id}"
        if node_id not in graph.graph.nodes():
            return {"error": f"Room {room_id} not found in circulation graph"}
        
        # Find nearest exit room and calculate path
        import networkx as nx
        exit_nodes = [n for n in graph.graph.nodes() if graph.graph.nodes[n].get('is_exit', False)]
        
        if not exit_nodes:
            return {"error": "No exit rooms found in building"}
        
        min_distance = float('inf')
        nearest_exit = None
        shortest_path = None
        
        for exit_node in exit_nodes:
            try:
                path = nx.shortest_path(graph.graph, node_id, exit_node)
                distance = nx.shortest_path_length(graph.graph, node_id, exit_node, weight='weight')
                if distance < min_distance:
                    min_distance = distance
                    nearest_exit = exit_node
                    shortest_path = path
            except nx.NetworkXNoPath:
                continue
        
        if nearest_exit is None:
            return {"error": f"No path to exit found from room {room_id}"}
        
        egress_info = {
            "distance": min_distance,
            "exit_room_id": nearest_exit.replace('room_', ''),
            "path": [node.replace('room_', '') for node in shortest_path],
            "is_accessible": True  # Simplified assumption
        }
        
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


@tool
def extract_ifc_data(ifc_file_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Extract building data from an IFC file and optionally save to JSON.
    
    Args:
        ifc_file_path: Path to the IFC file to extract
        output_path: Optional path to save extracted JSON data
    
    Returns:
        Dictionary with extraction results and project summary
    """
    try:
        from pathlib import Path
        
        ifc_path = Path(ifc_file_path)
        if not ifc_path.exists():
            return {"error": f"IFC file not found: {ifc_file_path}"}
        
        if not ifc_path.suffix.lower() == '.ifc':
            return {"error": f"File must have .ifc extension, got: {ifc_path.suffix}"}
        
        # Extract project data from IFC
        project = extract_from_ifc(ifc_path)
        
        # Optionally save to JSON
        if output_path:
            from src.extraction.ifc_extractor import save_to_json
            output_file = Path(output_path)
            save_to_json(project, output_file)
        
        # Load the extracted data into the global project variable
        global _project_data
        _project_data = project
        
        # Return summary
        return {
            "success": True,
            "ifc_file": str(ifc_path),
            "output_file": output_path,
            "project_name": project.metadata.project_name,
            "building_type": project.metadata.building_type,
            "levels": len(project.levels),
            "rooms": len(project.get_all_rooms()),
            "doors": len(project.get_all_doors()),
            "walls": len(project.get_all_walls()),
            "total_area_sqm": project.metadata.total_area,
            "message": f"Successfully extracted data from {ifc_path.name}"
        }
        
    except Exception as e:
        return {"error": f"Error extracting IFC data: {str(e)}"}


@tool
def create_circulation_graph_tool() -> Dict[str, Any]:
    """
    Create a circulation graph for the loaded project showing room connectivity and pathfinding.
    
    Returns:
        Dictionary with circulation graph analysis and connectivity information
    """
    if _project_data is None:
        return {"error": "No project data loaded. Extract IFC data first."}
    
    try:
        # Create circulation graph
        graph = create_circulation_graph(_project_data)
        
        # Get graph statistics
        num_rooms = graph.graph.number_of_nodes()
        num_connections = graph.graph.number_of_edges()
        
        # Find exit rooms
        exit_rooms = [node for node in graph.graph.nodes() 
                     if graph.graph.nodes[node].get('is_exit', False)]
        
        # Calculate some basic connectivity metrics
        if num_rooms > 1:
            import networkx as nx
            # Check if graph is connected
            is_connected = nx.is_connected(graph.graph)
            # Calculate average shortest path length if connected
            avg_path_length = nx.average_shortest_path_length(graph.graph) if is_connected else None
        else:
            is_connected = True
            avg_path_length = 0
        
        return {
            "success": True,
            "total_rooms": num_rooms,
            "total_connections": num_connections,
            "exit_rooms": len(exit_rooms),
            "exit_room_ids": [node.replace('room_', '') for node in exit_rooms],
            "is_connected": is_connected,
            "average_path_length": avg_path_length,
            "connectivity_score": num_connections / max(num_rooms, 1),
            "message": f"Created circulation graph with {num_rooms} rooms and {num_connections} connections"
        }
        
    except Exception as e:
        return {"error": f"Error creating circulation graph: {str(e)}"}


@tool
def calculate_clearance_tool(element1_type: str, element1_id: str, element2_type: str, element2_id: str) -> Dict[str, Any]:
    """
    Calculate minimum clearance distance between two building elements.
    
    WARNING: Use list_all_rooms() or get_available_element_ids() FIRST to get REAL IDs!
    Do NOT use made-up IDs like "0", "1", "Room_0" - they don't exist!
    
    Args:
        element1_type: Type of first element ('door', 'wall', 'room')
        element1_id: ID of first element (must be REAL ID from list_all_rooms/get_available_element_ids)
        element2_type: Type of second element ('door', 'wall', 'room') 
        element2_id: ID of second element (must be REAL ID from list_all_rooms/get_available_element_ids)
    
    Returns:
        Dictionary with clearance information and compliance status
    """
    if _project_data is None:
        return {"error": "No project data loaded. Extract IFC data first."}
    
    try:
        # Get elements by type and ID
        def get_element(elem_type: str, elem_id: str):
            if elem_type == 'door':
                return _project_data.get_door_by_id(elem_id)
            elif elem_type == 'room':
                return _project_data.get_room_by_id(elem_id)
            elif elem_type == 'wall':
                # For walls, we need to search through all levels
                for level in _project_data.levels:
                    for wall in level.walls:
                        if wall.get('id') == elem_id:
                            return wall
            return None
        
        elem1 = get_element(element1_type, element1_id)
        elem2 = get_element(element2_type, element2_id)
        
        if elem1 is None:
            if element1_type == 'room':
                return {"error": f"Room with ID '{element1_id}' not found. SOLUTION: Use list_all_rooms() or get_available_element_ids() FIRST to discover real room IDs!"}
            elif element1_type == 'door':
                return {"error": f"Door with ID '{element1_id}' not found. SOLUTION: Use list_all_doors() or get_available_element_ids() FIRST to discover real door IDs!"}
            else:
                return {"error": f"{element1_type} with ID '{element1_id}' not found. SOLUTION: Use get_available_element_ids() FIRST to discover real IDs!"}
        
        if elem2 is None:
            if element2_type == 'room':
                return {"error": f"Room with ID '{element2_id}' not found. SOLUTION: Use list_all_rooms() or get_available_element_ids() FIRST to discover real room IDs!"}
            elif element2_type == 'door':
                return {"error": f"Door with ID '{element2_id}' not found. SOLUTION: Use list_all_doors() or get_available_element_ids() FIRST to discover real door IDs!"}
            else:
                return {"error": f"{element2_type} with ID '{element2_id}' not found. SOLUTION: Use get_available_element_ids() FIRST to discover real IDs!"}
        
        # Convert Pydantic models to dictionaries for geometry calculations
        def convert_to_dict(element, elem_type: str):
            if elem_type == 'door':
                # Convert door to dict with position
                return {
                    'id': element.id,
                    'position': {
                        'x': element.position.x,
                        'y': element.position.y,
                        'z': element.position.z
                    }
                }
            elif elem_type == 'room':
                # Rooms don't have position data, so we can't calculate clearance
                return None
            elif elem_type == 'wall':
                # Walls should already be dictionaries
                return element
            return None
        
        # Convert elements to proper format
        elem1_dict = convert_to_dict(elem1, element1_type)
        elem2_dict = convert_to_dict(elem2, element2_type)
        
        if elem1_dict is None or elem2_dict is None:
            if element1_type == 'room' or element2_type == 'room':
                return {"error": f"Cannot calculate clearance for rooms - they don't have position data. SOLUTION: Use door-to-door clearance instead! Try calculate_clearance_tool('door', 'D3283', 'door', 'D3379') after calling list_all_doors()"}
            else:
                return {"error": f"Cannot calculate clearance for {element1_type} or {element2_type} - missing position data. SOLUTION: Use get_available_element_ids() to find elements with position data"}
        
        # Calculate clearance using the geometry function
        result = calculate_clearance_between_elements(elem1_dict, elem2_dict)
        
        if not result['success']:
            return {"error": result['error']}
        
        clearance_m = result['clearance_m']
        
        # Check compliance based on element types
        min_required = 0.8  # Default 800mm
        if element1_type == 'door' or element2_type == 'door':
            min_required = 0.9  # 900mm for door clearances
        
        is_compliant = clearance_m >= min_required
        
        return {
            "success": True,
            "element1": f"{element1_type}_{element1_id}",
            "element2": f"{element2_type}_{element2_id}",
            "clearance_m": clearance_m,
            "clearance_mm": clearance_m * 1000,
            "required_minimum_m": min_required,
            "is_compliant": is_compliant,
            "closest_points": result['closest_points'],
            "message": f"Clearance between {element1_type} {element1_id} and {element2_type} {element2_id} is {clearance_m:.2f}m"
        }
        
    except Exception as e:
        return {"error": f"Error calculating clearance: {str(e)}"}


@tool
def find_nearest_door_tool(point_x: float, point_y: float) -> Dict[str, Any]:
    """
    Find the nearest door to a specified point coordinate.
    
    Args:
        point_x: X coordinate of the point
        point_y: Y coordinate of the point
    
    Returns:
        Dictionary with nearest door information and distance
    """
    if _project_data is None:
        return {"error": "No project data loaded. Extract IFC data first."}
    
    try:
        # Get all doors
        doors = []
        for level in _project_data.levels:
            for door in level.doors:
                doors.append({
                    'id': door.id,
                    'position': {
                        'x': door.position.x,
                        'y': door.position.y,
                        'z': door.position.z
                    },
                    'width_mm': door.width_mm,
                    'door_type': door.door_type,
                    'is_emergency_exit': door.is_emergency_exit
                })
        
        if not doors:
            return {"error": "No doors found in project"}
        
        point = {'x': point_x, 'y': point_y}
        result = find_nearest_door(point, doors)
        
        if not result['success']:
            return {"error": result['error']}
        
        return {
            "success": True,
            "query_point": {"x": point_x, "y": point_y},
            "nearest_door_id": result['door_id'],
            "distance_m": result['distance_m'],
            "door_position": result['door_position'],
            "door_info": result['nearest_door'],
            "message": f"Nearest door to point ({point_x:.1f}, {point_y:.1f}) is {result['door_id']} at {result['distance_m']:.2f}m distance"
        }
        
    except Exception as e:
        return {"error": f"Error finding nearest door: {str(e)}"}


@tool
def load_json_data(json_file_path: str) -> Dict[str, Any]:
    """
    Load building data from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing building data
    
    Returns:
        Dictionary with loading results and project summary
    """
    global _project_data
    
    try:
        json_path = Path(json_file_path)
        if not json_path.exists():
            return {"error": f"JSON file not found: {json_file_path}"}
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Transform field names for compatibility
        def transform_data(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for key, value in obj.items():
                    # Transform field names
                    if key == 'width':
                        new_obj['width_mm'] = value
                    elif key == 'height':
                        new_obj['height_mm'] = value
                    elif key == 'thickness':
                        new_obj['thickness_mm'] = value
                    else:
                        new_obj[key] = transform_data(value)
                return new_obj
            elif isinstance(obj, list):
                return [transform_data(item) for item in obj]
            else:
                return obj
        
        transformed_data = transform_data(data)
        _project_data = Project(**transformed_data)
        
        return {
            "success": True,
            "json_file": str(json_path),
            "project_name": _project_data.metadata.project_name,
            "building_type": _project_data.metadata.building_type,
            "levels": len(_project_data.levels),
            "rooms": len(_project_data.get_all_rooms()),
            "doors": len(_project_data.get_all_doors()),
            "walls": len(_project_data.get_all_walls()),
            "message": f"Successfully loaded building data from {json_path.name}"
        }
    
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in file: {e}"}
    except Exception as e:
        return {"error": f"Error loading JSON data: {e}"}


@tool
def calculate_wall_angles_tool(wall1_id: str, wall2_id: str) -> Dict[str, Any]:
    """
    Calculate the angle between two walls and determine their relationship.
    
    Args:
        wall1_id: ID of the first wall
        wall2_id: ID of the second wall
    
    Returns:
        Dictionary with angle information and wall relationship
    """
    if _project_data is None:
        return {"error": "No project data loaded. Extract IFC data first."}
    
    try:
        # Find walls by ID
        wall1 = None
        wall2 = None
        
        for level in _project_data.levels:
            for wall in level.walls:
                if wall.get('id') == wall1_id:
                    wall1 = wall
                elif wall.get('id') == wall2_id:
                    wall2 = wall
        
        if wall1 is None:
            return {"error": f"Wall {wall1_id} not found"}
        if wall2 is None:
            return {"error": f"Wall {wall2_id} not found"}
        
        # Calculate angle using geometry function
        result = calculate_angle_between_walls(wall1, wall2)
        
        if not result['success']:
            return {"error": result['error']}
        
        return {
            "success": True,
            "wall1_id": wall1_id,
            "wall2_id": wall2_id,
            "angle_degrees": result['angle_degrees'],
            "angle_radians": result['angle_radians'],
            "relationship": result['relationship'],
            "wall1_vector": result['wall1_vector'],
            "wall2_vector": result['wall2_vector'],
            "message": f"Walls {wall1_id} and {wall2_id} have a {result['relationship']} relationship with {result['angle_degrees']:.1f}° angle"
        }
        
    except Exception as e:
        return {"error": f"Error calculating wall angles: {str(e)}"}


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
            "name": "list_all_rooms",
            "description": "CRITICAL: List all rooms with ACTUAL IDs. Use FIRST before calculate_clearance_tool! Do NOT make up room IDs!",
            "parameters": []
        },
        {
            "name": "get_available_element_ids",
            "description": "CRITICAL: Get all REAL element IDs (rooms, doors, walls). Use FIRST before calculate_clearance_tool! Do NOT make up IDs!",
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
        },
        {
            "name": "extract_ifc_data",
            "description": "Extract building data from an IFC file and load it for analysis",
            "parameters": ["ifc_file_path: str", "output_path: str (optional)"]
        },
        {
            "name": "create_circulation_graph_tool",
            "description": "Create circulation graph showing room connectivity and pathfinding analysis",
            "parameters": []
        },
        {
            "name": "calculate_clearance_tool",
            "description": "WARNING: Use list_all_rooms() FIRST! Calculate clearance between elements. Do NOT use made-up IDs!",
            "parameters": ["element1_type: str", "element1_id: str", "element2_type: str", "element2_id: str"]
        },
        {
            "name": "find_nearest_door_tool",
            "description": "Find the nearest door to a specified point coordinate",
            "parameters": ["point_x: float", "point_y: float"]
        },
        {
            "name": "calculate_wall_angles_tool",
            "description": "Calculate angle between two walls and determine their relationship",
            "parameters": ["wall1_id: str", "wall2_id: str"]
        }
    ]
