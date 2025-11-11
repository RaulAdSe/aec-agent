"""Building Data Toolkit - Generic functions for AEC building data operations.

This toolkit provides a minimal set of powerful, generic functions that can handle
all building data operations through flexible parameters and configurations.

The design philosophy is: "5 generic functions instead of 50 specific ones"
Each function is designed to be reusable across multiple compliance scenarios.

These functions are designed to be called directly by AI agents.

AGENT USAGE FLOW:
1. load_building_data() - Load IFC data first
2. get_all_elements() - Get element IDs by type  
3. get_all_properties() - Discover available property names
4. Use advanced tools with discovered IDs/properties:
   - query_elements() - Advanced filtering
   - calculate() - Mathematical operations  
   - find_related() - Spatial relationships
   - validate_rule() - Compliance checking
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

# Global variable to store loaded building data
_building_data: Optional[Dict] = None
_logger = logging.getLogger(__name__)


def load_building_data(data_path: str) -> Dict:
    """
    Load building data from JSON file into global variable.
    
    Args:
        data_path: Path to the building data JSON file
        
    Returns:
        Dict with status, data (loaded building data), and logs
        
    Examples:
        result = load_building_data("building.json")
        if result["status"] == "success":
            print("Data loaded with", result["data"]["file_info"]["total_elements"], "elements")
    """
    global _building_data
    logs = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            _building_data = json.load(f)
        
        total_elements = _building_data.get('file_info', {}).get('total_elements', 'unknown')
        logs.append(f"Loaded building data from {data_path}")
        logs.append(f"Total elements: {total_elements}")
        
        return {
            "status": "success",
            "data": _building_data,
            "logs": logs
        }
        
    except FileNotFoundError:
        error_msg = f"Building data file not found: {data_path}"
        logs.append(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": logs
        }
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in building data file: {e}"
        logs.append(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": logs
        }
    except Exception as e:
        error_msg = f"Error loading building data: {e}"
        logs.append(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": logs
        }


# ===== CONVENIENCE TOOL: GET ALL PROPERTIES =====
def get_all_properties(element_id: str) -> Dict:
    """
    Get all properties and quantities for an element.
    
    PRE-CONDITION: Use get_all_elements() or query_elements() first to obtain element_id
    
    Returns the complete property structure so agents can see
    all available data and choose appropriate properties based
    on their language knowledge.
    
    Args:
        element_id: Unique identifier of the element (obtained from get_all_elements/query_elements)
        
    Returns:
        Dict with status, data (all element properties), and logs
        
    Examples:
        # Step 1: Get element IDs first
        doors = get_all_elements("doors")
        door_id = doors["data"][0]["id"]
        
        # Step 2: Get all properties for analysis
        result = get_all_properties(door_id)
        if result["status"] == "success":
            properties = result["data"]["properties"]  
            quantities = result["data"]["quantities"]
            # Agent can now see all available fields and choose appropriate property paths
    """
    logs = []
    
    if _building_data is None:
        return {
            "status": "error",
            "data": None,
            "logs": ["Building data not loaded. Call load_building_data() first."]
        }
    
    # Find the element across all types
    element = None
    for element_type in ["spaces", "doors", "walls", "slabs", "stairs"]:
        if element_type in _building_data:
            for el in _building_data[element_type]:
                if el.get("id") == element_id:
                    element = el
                    break
        if element:
            break
    
    if not element:
        return {
            "status": "not_found",
            "data": None,
            "logs": [f"Element with id '{element_id}' not found"]
        }
    
    logs.append(f"Retrieved all properties for element {element_id}")
    return {
        "status": "success",
        "data": element,
        "logs": logs
    }


# ===== CONVENIENCE TOOL: GET ALL ELEMENTS =====
def get_all_elements(element_type: str) -> Dict:
    """
    Get all elements of a specific type from the building data.
    
    Simple, direct access to all elements without any filtering.
    This is the most common operation - just get me all doors, all rooms, etc.
    
    Args:
        element_type: Type of element to retrieve
            - "spaces" - All rooms and spaces
            - "doors" - All doors
            - "walls" - All wall elements  
            - "slabs" - All floor/ceiling slabs
            - "stairs" - All staircase elements
            
    Returns:
        List of all elements of the specified type
        
    Examples:
        # Get all doors in the building
        all_doors = get_all_elements("doors")
        
        # Get all spaces/rooms
        all_rooms = get_all_elements("spaces")
        
        # Get all walls
        all_walls = get_all_elements("walls")
        
        # Get all stairs
        all_stairs = get_all_elements("stairs")
        
        # Get all slabs (floors/ceilings)
        all_slabs = get_all_elements("slabs")
    """
    logs = []
    
    if _building_data is None:
        error_msg = "Building data not loaded. Call load_building_data() first."
        _logger.error(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": [error_msg]
        }
    
    if element_type not in _building_data:
        warning_msg = f"Element type '{element_type}' not found in building data"
        _logger.warning(warning_msg)
        logs.append(warning_msg)
        return {
            "status": "success", 
            "data": [],
            "logs": logs
        }
    
    elements = _building_data[element_type]
    success_msg = f"Retrieved {len(elements)} {element_type} elements"
    _logger.debug(success_msg)
    logs.append(success_msg)
    
    return {
        "status": "success",
        "data": elements,
        "logs": logs
    }


# ===== TOOL 2: GENERIC QUERY =====
def query_elements(element_type: str, filters: Optional[Dict] = None) -> Dict:
    """
    Query and filter building elements of any type.
    
    PRE-CONDITION: Use get_all_properties() on a sample element first to discover available property names
    
    This is the primary data access tool. It can find any elements
    with any combination of filters.
    
    Args:
        element_type: Type of element to query
            - "spaces" - Rooms and spaces
            - "doors" - All doors  
            - "walls" - Wall elements
            - "slabs" - Floor/ceiling slabs
            - "stairs" - Staircase elements
            
        filters: Optional filters to apply
            - Direct property filters: {"name": "Office"}
            - Comparison filters: {"area__gt": 50, "width__lt": 2.0}
            - List filters: {"id__in": ["id1", "id2"]}
            - Nested property filters: {"properties.IsExternal": True}
            
    Returns:
        List of matching elements with full data
        
    Examples:
        # Step 1: Discover available properties first
        sample_elements = get_all_elements("doors")  
        sample_id = sample_elements["data"][0]["id"]
        properties_info = get_all_properties(sample_id)
        # Now you know available property names
        
        # Step 2: Use discovered property names in filters
        query_elements("spaces", {"properties.Ebene": "E01_OKRD"})
        query_elements("doors", {"quantities.Width__gt": 1.0})
        query_elements("spaces", {"quantities.NetFloorArea__gt": 50})
        query_elements("doors", {"properties.IsExternal": True})
        query_elements("spaces", {"id__in": ["room1", "room2"]})
    """
    logs = []
    
    # Get all elements of the specified type
    result = get_all_elements(element_type)
    if result["status"] == "error":
        return result
    
    elements = result["data"]
    logs.extend(result["logs"])
    
    if not filters:
        logs.append("No filters applied")
        return {
            "status": "success",
            "data": elements,
            "logs": logs
        }
    
    def _matches_filter(element: Dict, filter_key: str, filter_value: Any) -> bool:
        """Check if element matches a single filter."""
        # Handle comparison operators
        if "__" in filter_key:
            property_path, operator = filter_key.rsplit("__", 1)
            prop_result = get_property(element["id"], property_path)
            actual_value = prop_result["data"] if prop_result["status"] == "success" else None
            
            if actual_value is None:
                return False
                
            try:
                if operator == "gt":
                    return float(actual_value) > float(filter_value)
                elif operator == "lt":
                    return float(actual_value) < float(filter_value)
                elif operator == "gte":
                    return float(actual_value) >= float(filter_value)
                elif operator == "lte":
                    return float(actual_value) <= float(filter_value)
                elif operator == "in":
                    return actual_value in filter_value
                elif operator == "contains":
                    return str(filter_value).lower() in str(actual_value).lower()
                else:
                    _logger.warning(f"Unknown operator: {operator}")
                    return False
            except (ValueError, TypeError):
                return False
        else:
            # Direct property comparison
            prop_result = get_property(element["id"], filter_key)
            actual_value = prop_result["data"] if prop_result["status"] == "success" else None
            return actual_value == filter_value
    
    # Apply all filters
    filtered_elements = []
    for element in elements:
        matches_all = True
        for filter_key, filter_value in filters.items():
            if not _matches_filter(element, filter_key, filter_value):
                matches_all = False
                break
        if matches_all:
            filtered_elements.append(element)
    
    success_msg = f"Query returned {len(filtered_elements)} of {len(elements)} {element_type} elements"
    _logger.debug(success_msg)
    logs.append(success_msg)
    
    return {
        "status": "success",
        "data": filtered_elements,
        "logs": logs
    }


# ===== INTERNAL HELPER: PROPERTY GETTER =====
def get_property(element_id: str, property_path: str) -> Dict:
    """
    INTERNAL HELPER: Extract any property from any building element.
    
    NOTE FOR AGENTS: Use get_all_properties() instead for agent workflows.
    This function is used internally by other toolkit functions.
    
    Uses dot notation to access nested properties safely.
    Handles missing properties gracefully.
    """
    logs = []
    
    if _building_data is None:
        error_msg = "Building data not loaded. Call load_building_data() first."
        _logger.error(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": [error_msg]
        }
    
    # Find the element across all types
    element = None
    for element_type in ["spaces", "doors", "walls", "slabs", "stairs"]:
        if element_type in _building_data:
            for el in _building_data[element_type]:
                if el.get("id") == element_id:
                    element = el
                    break
        if element:
            break
    
    if not element:
        warning_msg = f"Element with id '{element_id}' not found"
        _logger.warning(warning_msg)
        return {
            "status": "not_found",
            "data": None,
            "logs": [warning_msg]
        }
    
    # Navigate the property path using dot notation
    try:
        current = element
        path_parts = property_path.split(".")
        
        for part in path_parts:
            # Handle array indices (e.g., "0", "1", etc.)
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    error_msg = f"Array index '{idx}' out of range for property path '{property_path}' in element '{element_id}'"
                    return {
                        "status": "not_found",
                        "data": None,
                        "logs": [error_msg]
                    }
            else:
                # Handle dictionary keys
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    error_msg = f"Property '{part}' not found in path '{property_path}' for element '{element_id}'"
                    return {
                        "status": "not_found",
                        "data": None,
                        "logs": [error_msg]
                    }
        
        success_msg = f"Retrieved property '{property_path}' for element '{element_id}'"
        _logger.debug(success_msg)
        logs.append(success_msg)
        
        return {
            "status": "success",
            "data": current,
            "logs": logs
        }
        
    except (KeyError, IndexError, TypeError) as e:
        error_msg = f"Property path '{property_path}' not found for element '{element_id}': {e}"
        _logger.debug(error_msg)
        return {
            "status": "not_found",
            "data": None,
            "logs": [error_msg]
        }


# ===== TOOL 3: GENERIC CALCULATOR =====
def calculate(operation: str, **kwargs) -> Dict:
    """
    Perform any calculation operation on building data.
    
    PRE-CONDITION: Use get_all_elements() to obtain element_ids, get_all_properties() to discover property paths
    
    Single calculation tool that handles all mathematical operations
    through operation type and flexible parameters.
    
    Args:
        operation: Type of calculation to perform
        **kwargs: Operation-specific parameters
        
    Supported Operations:
    
    DISTANCE CALCULATIONS:
        operation="distance_2d"
            point1: [x, y] - First point coordinates
            point2: [x, y] - Second point coordinates
            
        operation="distance_3d" 
            point1: [x, y, z] - First point coordinates
            point2: [x, y, z] - Second point coordinates
            
        operation="distance_between_elements"
            element1_id: str - First element ID
            element2_id: str - Second element ID
            
    AREA CALCULATIONS:
        operation="area_sum"
            element_ids: List[str] - Elements to sum areas
            
        operation="total_floor_area"
            level: str - Building level identifier
            
    VOLUME CALCULATIONS:
        operation="volume_sum"
            element_ids: List[str] - Elements to sum volumes
            
    OCCUPANCY CALCULATIONS:
        operation="total_occupancy"
            element_ids: List[str] - Spaces to sum occupancy
            
        operation="occupancy_density"
            space_id: str - Space to calculate density
            
    PATH CALCULATIONS:
        operation="path_length"
            waypoints: List[List[float]] - Path coordinates
            
        operation="shortest_path_length"
            from_element: str - Starting element ID
            to_element: str - Target element ID
            
    STATISTICS:
        operation="statistics"
            values: List[float] - Values to analyze
            returns: {"min": float, "max": float, "avg": float, "sum": float}
            
    Returns:
        Calculation result (type depends on operation)
        
    Examples:
        # Calculate distance between two points
        calculate("distance_2d", point1=[0, 0], point2=[3, 4])  # Returns: 5.0
        
        # Calculate distance between elements
        calculate("distance_between_elements", 
                 element1_id="room_123", element2_id="door_456")
        
        # Sum areas of multiple rooms
        calculate("area_sum", element_ids=["room1", "room2", "room3"])
        
        # Calculate total occupancy for a floor
        spaces = query_elements("spaces", {"level": 1})
        space_ids = [s["id"] for s in spaces]
        calculate("total_occupancy", element_ids=space_ids)
        
        # Get statistics for door widths
        doors = query_elements("doors")
        widths = [get_property(d["id"], "quantities.Width") for d in doors]
        calculate("statistics", values=widths)
    """
    import math
    logs = []
    
    try:
        if operation == "distance_2d":
            point1 = kwargs.get("point1")
            point2 = kwargs.get("point2")
            if not point1 or not point2:
                return {"status": "error", "data": None, "logs": ["distance_2d requires point1 and point2"]}
            result = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
            return {"status": "success", "data": result, "logs": [f"Calculated 2D distance: {result}"]}
        
        elif operation == "distance_3d":
            point1 = kwargs.get("point1")
            point2 = kwargs.get("point2")
            if not point1 or not point2:
                return {"status": "error", "data": None, "logs": ["distance_3d requires point1 and point2"]}
            result = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
            return {"status": "success", "data": result, "logs": [f"Calculated 3D distance: {result}"]}
        
        elif operation == "distance_between_elements":
            element1_id = kwargs.get("element1_id")
            element2_id = kwargs.get("element2_id")
            if not element1_id or not element2_id:
                return {"status": "error", "data": None, "logs": ["distance_between_elements requires element1_id and element2_id"]}
            
            # Get center points from bounding boxes
            bbox1_result = get_property(element1_id, "geometry.bbox")
            bbox2_result = get_property(element2_id, "geometry.bbox")
            
            if bbox1_result["status"] != "success" or bbox2_result["status"] != "success":
                return {"status": "error", "data": None, "logs": ["Elements must have geometry.bbox data"]}
            
            bbox1, bbox2 = bbox1_result["data"], bbox2_result["data"]
            
            # Calculate center points
            center1 = [
                (bbox1["min"][0] + bbox1["max"][0]) / 2,
                (bbox1["min"][1] + bbox1["max"][1]) / 2,
                (bbox1["min"][2] + bbox1["max"][2]) / 2
            ]
            center2 = [
                (bbox2["min"][0] + bbox2["max"][0]) / 2,
                (bbox2["min"][1] + bbox2["max"][1]) / 2,
                (bbox2["min"][2] + bbox2["max"][2]) / 2
            ]
            
            distance_result = calculate("distance_3d", point1=center1, point2=center2)
            if distance_result["status"] == "success":
                logs.extend(distance_result["logs"])
                logs.append(f"Distance between elements {element1_id} and {element2_id}: {distance_result['data']}")
                return {"status": "success", "data": distance_result["data"], "logs": logs}
            else:
                return distance_result
        
        elif operation == "area_sum":
            element_ids = kwargs.get("element_ids", [])
            area_property = kwargs.get("area_property", "quantities.NetFloorArea")
            total_area = 0.0
            processed_count = 0
            
            for element_id in element_ids:
                area_result = get_property(element_id, area_property)
                if area_result["status"] == "success" and area_result["data"]:
                    total_area += float(area_result["data"])
                    processed_count += 1
            
            logs.append(f"Summed areas from {processed_count}/{len(element_ids)} elements")
            return {"status": "success", "data": total_area, "logs": logs}
        
        elif operation == "volume_sum":
            element_ids = kwargs.get("element_ids", [])
            volume_property = kwargs.get("volume_property", "quantities.GrossVolume")
            total_volume = 0.0
            processed_count = 0
            
            for element_id in element_ids:
                volume_result = get_property(element_id, volume_property)
                if volume_result["status"] == "success" and volume_result["data"]:
                    total_volume += float(volume_result["data"])
                    processed_count += 1
            
            logs.append(f"Summed volumes from {processed_count}/{len(element_ids)} elements")
            return {"status": "success", "data": total_volume, "logs": logs}
        
        elif operation == "total_floor_area":
            level = kwargs.get("level")
            area_property = kwargs.get("area_property", "quantities.NetFloorArea")
            if not level:
                return {"status": "error", "data": None, "logs": ["total_floor_area requires level"]}
            
            spaces_result = query_elements("spaces", {"properties.Ebene": level})
            if spaces_result["status"] != "success":
                return spaces_result
            
            space_ids = [s["id"] for s in spaces_result["data"]]
            return calculate("area_sum", element_ids=space_ids, area_property=area_property)
        
        elif operation == "total_occupancy":
            element_ids = kwargs.get("element_ids", [])
            occupancy_property = kwargs.get("occupancy_property", "properties.Personenzahl")
            total_occupancy = 0.0
            processed_count = 0
            
            for element_id in element_ids:
                occupancy_result = get_property(element_id, occupancy_property)
                if occupancy_result["status"] == "success" and occupancy_result["data"]:
                    total_occupancy += float(occupancy_result["data"])
                    processed_count += 1
            
            logs.append(f"Summed occupancy from {processed_count}/{len(element_ids)} elements")
            return {"status": "success", "data": total_occupancy, "logs": logs}
        
        elif operation == "occupancy_density":
            space_id = kwargs.get("space_id")
            occupancy_property = kwargs.get("occupancy_property", "properties.Personenzahl")
            area_property = kwargs.get("area_property", "quantities.NetFloorArea")
            
            if not space_id:
                return {"status": "error", "data": None, "logs": ["occupancy_density requires space_id"]}
            
            occupancy_result = get_property(space_id, occupancy_property)
            area_result = get_property(space_id, area_property)
            
            if occupancy_result["status"] != "success" or area_result["status"] != "success":
                return {"status": "error", "data": None, "logs": ["Could not get occupancy or area data"]}
            
            occupancy = float(occupancy_result["data"]) if occupancy_result["data"] else 0.0
            area = float(area_result["data"]) if area_result["data"] else 0.0
            
            if area == 0:
                return {"status": "error", "data": None, "logs": ["Cannot calculate density with zero area"]}
            
            density = occupancy / area
            logs.append(f"Calculated occupancy density: {density} persons/m²")
            return {"status": "success", "data": density, "logs": logs}
        
        elif operation == "path_length":
            waypoints = kwargs.get("waypoints", [])
            if len(waypoints) < 2:
                return {"status": "success", "data": 0.0, "logs": ["Path needs at least 2 waypoints"]}
            
            total_length = 0.0
            for i in range(1, len(waypoints)):
                if len(waypoints[i]) == 2 and len(waypoints[i-1]) == 2:
                    distance_result = calculate("distance_2d", point1=waypoints[i-1], point2=waypoints[i])
                else:
                    distance_result = calculate("distance_3d", point1=waypoints[i-1], point2=waypoints[i])
                
                if distance_result["status"] == "success":
                    total_length += distance_result["data"]
            
            logs.append(f"Calculated path length: {total_length}m for {len(waypoints)} waypoints")
            return {"status": "success", "data": total_length, "logs": logs}
        
        elif operation == "statistics":
            values = kwargs.get("values", [])
            if not values:
                return {"status": "success", "data": {"min": 0, "max": 0, "avg": 0, "sum": 0}, "logs": ["No values provided"]}
            
            # Extract actual values from get_property results if needed
            numeric_values = []
            for val in values:
                if isinstance(val, dict) and val.get("status") == "success":
                    val = val.get("data")
                if val is not None:
                    try:
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        continue
            
            if not numeric_values:
                return {"status": "success", "data": {"min": 0, "max": 0, "avg": 0, "sum": 0}, "logs": ["No valid numeric values"]}
            
            result = {
                "min": min(numeric_values),
                "max": max(numeric_values),
                "avg": sum(numeric_values) / len(numeric_values),
                "sum": sum(numeric_values)
            }
            return {"status": "success", "data": result, "logs": [f"Calculated statistics for {len(numeric_values)} values"]}
        
        else:
            return {"status": "error", "data": None, "logs": [f"Operation '{operation}' not implemented yet"]}
    
    except Exception as e:
        return {"status": "error", "data": None, "logs": [f"Error in calculation: {str(e)}"]}


# ===== TOOL 4: GENERIC RELATIONSHIP FINDER =====
def find_related(element_id: str, relationship_type: str, **kwargs) -> Dict:
    """
    Find elements related to a given element through spatial or logical relationships.
    
    PRE-CONDITION: Use get_all_elements() to obtain element_id
    
    Discovers connections and relationships between building elements
    using geometric analysis and property relationships.
    
    Args:
        element_id: ID of the source element
        relationship_type: Type of relationship to find
        **kwargs: Relationship-specific parameters
        
    Supported Relationships:
    
    SPATIAL RELATIONSHIPS:
        relationship_type="connected_doors"
            Returns doors that connect to the given space
            
        relationship_type="adjacent_spaces"
            Returns spaces that share boundaries with given space
            tolerance: float - Distance tolerance for adjacency (default: 0.1m)
            
        relationship_type="same_level" 
            Returns all elements on the same building level
            
        relationship_type="within_distance"
            max_distance: float - Maximum distance in meters
            element_types: List[str] - Types to search (default: all)
            
    CONNECTIVITY RELATIONSHIPS:
        relationship_type="connected_spaces"
            For doors: returns spaces the door connects
            
        relationship_type="access_path"
            target_element: str - Target element ID
            Returns path of connected elements from source to target
            
    HIERARCHY RELATIONSHIPS:
        relationship_type="parent_elements"
            Returns elements that contain this element
            
        relationship_type="child_elements" 
            Returns elements contained within this element
            
    PROPERTY RELATIONSHIPS:
        relationship_type="same_property"
            property_path: str - Property to match
            Returns elements with same property value
            
    Returns:
        List of related elements with full data
        
    Examples:
        # Find doors connected to a room
        find_related("room_123", "connected_doors")
        
        # Find adjacent rooms
        find_related("room_123", "adjacent_spaces", tolerance=0.2)
        
        # Find all elements on same level
        find_related("room_123", "same_level")
        
        # Find nearby doors within 5m
        find_related("room_123", "within_distance", 
                    max_distance=5.0, element_types=["doors"])
        
        # Find spaces connected by a door
        find_related("door_456", "connected_spaces")
        
        # Find rooms with same use type
        find_related("room_123", "same_property", 
                    property_path="properties.use_type")
    """
    logs = []
    
    try:
        # Check if element exists
        element_result = get_property(element_id, "id")
        if element_result["status"] != "success":
            return {
                "status": "error",
                "data": None,
                "logs": [f"Element with id '{element_id}' not found"]
            }
        
        if relationship_type == "connected_spaces":
            # For doors: find spaces in connected_spaces array
            doors_result = get_all_elements("doors")
            if doors_result["status"] != "success":
                return doors_result
                
            for door in doors_result["data"]:
                if door.get("id") == element_id:
                    connected = door.get("connected_spaces", [])
                    logs.append(f"Found {len(connected)} connected spaces for door {element_id}")
                    return {
                        "status": "success",
                        "data": connected,
                        "logs": logs
                    }
            
            return {
                "status": "not_found",
                "data": [],
                "logs": [f"Door {element_id} not found or has no connected spaces"]
            }
        
        elif relationship_type == "same_level":
            # Find elements on same building level
            element_level_result = get_property(element_id, "properties.Ebene")
            if element_level_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Could not determine level for element {element_id}"]
                }
                
            level = element_level_result["data"]
            related_elements = []
            
            for element_type in ["spaces", "doors", "walls", "slabs", "stairs"]:
                elements_result = query_elements(element_type, {"properties.Ebene": level})
                if elements_result["status"] == "success":
                    # Exclude the original element
                    for elem in elements_result["data"]:
                        if elem.get("id") != element_id:
                            related_elements.append(elem)
            
            logs.append(f"Found {len(related_elements)} elements on level {level}")
            return {
                "status": "success",
                "data": related_elements,
                "logs": logs
            }
        
        elif relationship_type == "same_property":
            property_path = kwargs.get("property_path")
            if not property_path:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["same_property requires property_path parameter"]
                }
            
            # Get the property value for the source element
            prop_result = get_property(element_id, property_path)
            if prop_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Could not get property {property_path} for element {element_id}"]
                }
            
            target_value = prop_result["data"]
            related_elements = []
            
            for element_type in ["spaces", "doors", "walls", "slabs", "stairs"]:
                elements_result = query_elements(element_type, {property_path: target_value})
                if elements_result["status"] == "success":
                    # Exclude the original element
                    for elem in elements_result["data"]:
                        if elem.get("id") != element_id:
                            related_elements.append(elem)
            
            logs.append(f"Found {len(related_elements)} elements with {property_path}={target_value}")
            return {
                "status": "success",
                "data": related_elements,
                "logs": logs
            }
        
        elif relationship_type == "within_distance":
            max_distance = kwargs.get("max_distance")
            element_types = kwargs.get("element_types", ["spaces", "doors", "walls", "slabs", "stairs"])
            
            if max_distance is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["within_distance requires max_distance parameter"]
                }
            
            related_elements = []
            
            for element_type in element_types:
                elements_result = get_all_elements(element_type)
                if elements_result["status"] == "success":
                    for elem in elements_result["data"]:
                        if elem.get("id") != element_id:
                            # Calculate distance between elements
                            distance_result = calculate("distance_between_elements", 
                                                      element1_id=element_id, 
                                                      element2_id=elem["id"])
                            if distance_result["status"] == "success":
                                if distance_result["data"] <= max_distance:
                                    related_elements.append(elem)
            
            logs.append(f"Found {len(related_elements)} elements within {max_distance}m")
            return {
                "status": "success",
                "data": related_elements,
                "logs": logs
            }
        
        elif relationship_type == "connected_doors":
            # Find doors that connect to a given space
            space_result = get_property(element_id, "id")
            if space_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Space {element_id} not found"]
                }
            
            doors_result = get_all_elements("doors")
            if doors_result["status"] != "success":
                return doors_result
            
            connected_doors = []
            for door in doors_result["data"]:
                connected_spaces = door.get("connected_spaces", [])
                # Check if this space is in the door's connected_spaces
                for connected_space in connected_spaces:
                    if connected_space.get("id") == element_id:
                        connected_doors.append(door)
                        break
            
            logs.append(f"Found {len(connected_doors)} doors connected to space {element_id}")
            return {
                "status": "success",
                "data": connected_doors,
                "logs": logs
            }
        
        elif relationship_type == "adjacent_spaces":
            tolerance = kwargs.get("tolerance", 0.1)  # Default 10cm tolerance
            
            # Get the source space's bounding box
            source_bbox_result = get_property(element_id, "geometry.bbox")
            if source_bbox_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Could not get geometry for element {element_id}"]
                }
            
            source_bbox = source_bbox_result["data"]
            
            # Get all other spaces
            spaces_result = get_all_elements("spaces")
            if spaces_result["status"] != "success":
                return spaces_result
            
            adjacent_spaces = []
            for space in spaces_result["data"]:
                if space.get("id") != element_id:
                    space_bbox_result = get_property(space["id"], "geometry.bbox")
                    if space_bbox_result["status"] == "success":
                        space_bbox = space_bbox_result["data"]
                        
                        # Check if bounding boxes are adjacent (within tolerance)
                        # Simplified adjacency check - spaces are adjacent if their bounding boxes are very close
                        min_distance = float('inf')
                        
                        # Check distance between all corners of the bounding boxes
                        for i in range(2):  # min/max
                            for j in range(2):  # min/max
                                for k in range(2):  # min/max
                                    corner1 = [
                                        source_bbox["min"][0] if i == 0 else source_bbox["max"][0],
                                        source_bbox["min"][1] if j == 0 else source_bbox["max"][1],
                                        source_bbox["min"][2] if k == 0 else source_bbox["max"][2]
                                    ]
                                    for ii in range(2):
                                        for jj in range(2):
                                            for kk in range(2):
                                                corner2 = [
                                                    space_bbox["min"][0] if ii == 0 else space_bbox["max"][0],
                                                    space_bbox["min"][1] if jj == 0 else space_bbox["max"][1],
                                                    space_bbox["min"][2] if kk == 0 else space_bbox["max"][2]
                                                ]
                                                distance_result = calculate("distance_3d", point1=corner1, point2=corner2)
                                                if distance_result["status"] == "success":
                                                    min_distance = min(min_distance, distance_result["data"])
                        
                        if min_distance <= tolerance:
                            adjacent_spaces.append(space)
            
            logs.append(f"Found {len(adjacent_spaces)} spaces adjacent to {element_id} (tolerance: {tolerance}m)")
            return {
                "status": "success",
                "data": adjacent_spaces,
                "logs": logs
            }
        
        else:
            return {
                "status": "error",
                "data": None,
                "logs": [f"Relationship type '{relationship_type}' not implemented yet"]
            }
    
    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "logs": [f"Error finding related elements: {str(e)}"]
        }


# ===== TOOL 5: GENERIC VALIDATOR =====
def validate_rule(rule_type: str, element_id: str, criteria: Dict) -> Dict:
    """
    Validate compliance rules against building elements.
    
    PRE-CONDITION: Use get_all_elements() to obtain element_id, get_all_properties() to discover property paths for criteria
    
    Generic validation tool that can check any compliance requirement
    against any element type through configurable rule types.
    
    Args:
        rule_type: Type of validation rule to apply
        element_id: ID of element to validate
        criteria: Rule-specific validation criteria
        
    Supported Rule Types:
    
    DIMENSIONAL RULES:
        rule_type="min_width"
            min_value: float - Minimum required width
            
        rule_type="max_width"
            max_value: float - Maximum allowed width
            
        rule_type="min_area"
            min_value: float - Minimum required area
            
        rule_type="min_height"
            min_value: float - Minimum required height
            
    ACCESSIBILITY RULES:
        rule_type="accessibility_width"
            min_width: float - Minimum width for accessibility
            door_type: str - Type of door (default: "general")
            
        rule_type="accessibility_path"
            max_slope: float - Maximum allowed slope
            min_width: float - Minimum path width
            
    FIRE SAFETY RULES:
        rule_type="fire_rating"
            required_rating: str - Required fire rating
            
        rule_type="exit_capacity"
            max_occupancy: int - Maximum occupancy load
            min_exit_width: float - Minimum total exit width
            
    OCCUPANCY RULES:
        rule_type="max_occupancy"
            max_persons: int - Maximum allowed persons
            
        rule_type="occupancy_density"
            max_density: float - Maximum persons per sqm
            
    PROPERTY RULES:
        rule_type="required_property"
            property_path: str - Required property path
            required_value: Any - Expected property value
            
        rule_type="property_range"
            property_path: str - Property to check
            min_value: float - Minimum value
            max_value: float - Maximum value
            
    Returns:
        Validation result dictionary:
        {
            "is_valid": bool,
            "rule_type": str,
            "element_id": str,
            "criteria": Dict,
            "actual_value": Any,
            "message": str,
            "details": Dict
        }
        
    Examples:
        # Check minimum door width
        validate_rule("min_width", "door_123", {"min_value": 0.8})
        
        # Check accessibility compliance
        validate_rule("accessibility_width", "door_456", 
                     {"min_width": 0.85, "door_type": "emergency_exit"})
        
        # Check fire rating
        validate_rule("fire_rating", "door_789", 
                     {"required_rating": "EI30"})
        
        # Check room area requirements
        validate_rule("min_area", "room_123", {"min_value": 10.0})
        
        # Check occupancy limits  
        validate_rule("max_occupancy", "room_456", {"max_persons": 50})
        
        # Check property exists with specific value
        validate_rule("required_property", "door_123",
                     {"property_path": "properties.IsExternal", 
                      "required_value": True})
    """
    logs = []
    
    try:
        # Check if element exists
        element_check = get_property(element_id, "id")
        if element_check["status"] != "success":
            return {
                "status": "error",
                "data": {
                    "is_valid": False,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": None,
                    "message": f"Element {element_id} not found",
                    "details": {}
                },
                "logs": [f"Element {element_id} not found"]
            }
        
        # Dimensional rules
        if rule_type == "min_width":
            min_value = criteria.get("min_value")
            if min_value is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["min_width rule requires min_value in criteria"]
                }
            
            # Get width from criteria (agent should specify the exact property path)
            width_property = criteria.get("width_property", "quantities.Width")
            width_result = get_property(element_id, width_property)
            
            if width_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Width property '{width_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_width = float(width_result["data"])
            is_valid = actual_width >= min_value
            
            message = f"Width {actual_width}m {'meets' if is_valid else 'fails'} minimum requirement of {min_value}m"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_width,
                    "message": message,
                    "details": {"width_property_used": width_property}
                },
                "logs": logs
            }
        
        elif rule_type == "min_area":
            min_value = criteria.get("min_value")
            if min_value is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["min_area rule requires min_value in criteria"]
                }
            
            # Get area from criteria (agent should specify the exact property path)  
            area_property = criteria.get("area_property", "quantities.NetFloorArea")
            area_result = get_property(element_id, area_property)
            
            if area_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Area property '{area_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_area = float(area_result["data"])
            is_valid = actual_area >= min_value
            
            message = f"Area {actual_area}m² {'meets' if is_valid else 'fails'} minimum requirement of {min_value}m²"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_area,
                    "message": message,
                    "details": {}
                },
                "logs": logs
            }
        
        elif rule_type == "required_property":
            property_path = criteria.get("property_path")
            required_value = criteria.get("required_value")
            
            if not property_path:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["required_property rule requires property_path in criteria"]
                }
            
            prop_result = get_property(element_id, property_path)
            
            if prop_result["status"] != "success":
                return {
                    "status": "success",
                    "data": {
                        "is_valid": False,
                        "rule_type": rule_type,
                        "element_id": element_id,
                        "criteria": criteria,
                        "actual_value": None,
                        "message": f"Required property {property_path} not found",
                        "details": {}
                    },
                    "logs": [f"Property {property_path} not found for element {element_id}"]
                }
            
            actual_value = prop_result["data"]
            is_valid = actual_value == required_value
            
            message = f"Property {property_path} = {actual_value} {'matches' if is_valid else 'does not match'} required value {required_value}"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_value,
                    "message": message,
                    "details": {}
                },
                "logs": logs
            }
        
        elif rule_type == "max_width":
            max_value = criteria.get("max_value")
            if max_value is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["max_width rule requires max_value in criteria"]
                }
            
            # Get width from criteria (agent should specify the exact property path)
            width_property = criteria.get("width_property", "quantities.Width")
            width_result = get_property(element_id, width_property)
            
            if width_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Width property '{width_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_width = float(width_result["data"])
            is_valid = actual_width <= max_value
            
            message = f"Width {actual_width}m {'meets' if is_valid else 'exceeds'} maximum requirement of {max_value}m"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_width,
                    "message": message,
                    "details": {"width_property_used": width_property}
                },
                "logs": logs
            }
        
        elif rule_type == "min_height":
            min_value = criteria.get("min_value")
            if min_value is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["min_height rule requires min_value in criteria"]
                }
            
            # Get height from criteria (agent should specify the exact property path)
            height_property = criteria.get("height_property", "quantities.Height")
            height_result = get_property(element_id, height_property)
            
            if height_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Height property '{height_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_height = float(height_result["data"])
            is_valid = actual_height >= min_value
            
            message = f"Height {actual_height}m {'meets' if is_valid else 'fails'} minimum requirement of {min_value}m"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_height,
                    "message": message,
                    "details": {"height_property_used": height_property}
                },
                "logs": logs
            }
        
        elif rule_type == "accessibility_width":
            min_width = criteria.get("min_width")
            door_type = criteria.get("door_type", "general")
            
            if min_width is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["accessibility_width rule requires min_width in criteria"]
                }
            
            # Get width from criteria (agent should specify the exact property path)
            width_property = criteria.get("width_property", "quantities.Width")
            width_result = get_property(element_id, width_property)
            
            if width_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Width property '{width_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_width = float(width_result["data"])
            is_valid = actual_width >= min_width
            
            message = f"Width {actual_width}m {'meets' if is_valid else 'fails'} accessibility requirement of {min_width}m for {door_type} door"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_width,
                    "message": message,
                    "details": {"door_type": door_type, "width_property_used": width_property}
                },
                "logs": logs
            }
        
        elif rule_type == "fire_rating":
            required_rating = criteria.get("required_rating")
            if not required_rating:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["fire_rating rule requires required_rating in criteria"]
                }
            
            # Get fire rating from criteria (agent should specify the exact property path)
            rating_property = criteria.get("rating_property", "properties.FireRating")
            rating_result = get_property(element_id, rating_property)
            
            if rating_result["status"] != "success":
                # Check alternative fire rating properties
                alternative_properties = ["properties.fire_rating", "properties.FireResistance", "properties.IsFireRated"]
                actual_rating = None
                
                for prop in alternative_properties:
                    alt_result = get_property(element_id, prop)
                    if alt_result["status"] == "success":
                        actual_rating = alt_result["data"]
                        rating_property = prop
                        break
                
                if actual_rating is None:
                    return {
                        "status": "error",
                        "data": None,
                        "logs": [f"Fire rating property '{rating_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                    }
            else:
                actual_rating = rating_result["data"]
            
            # Handle boolean fire rating properties
            if isinstance(actual_rating, bool):
                if actual_rating:
                    # If it's just a boolean True, we assume it has some fire rating
                    is_valid = required_rating.lower() in ["true", "any", "yes"]
                    message = f"Element has fire rating: {actual_rating}, required: {required_rating}"
                else:
                    is_valid = required_rating.lower() in ["false", "none", "no"]
                    message = f"Element has no fire rating, required: {required_rating}"
            else:
                # String comparison for specific ratings like "EI30", "EI60", etc.
                is_valid = str(actual_rating).strip().upper() == str(required_rating).strip().upper()
                message = f"Fire rating '{actual_rating}' {'matches' if is_valid else 'does not match'} required rating '{required_rating}'"
            
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_rating,
                    "message": message,
                    "details": {"rating_property_used": rating_property}
                },
                "logs": logs
            }
        
        elif rule_type == "exit_capacity":
            max_occupancy = criteria.get("max_occupancy")
            min_exit_width = criteria.get("min_exit_width")
            
            if max_occupancy is None or min_exit_width is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["exit_capacity rule requires max_occupancy and min_exit_width in criteria"]
                }
            
            # Get current occupancy
            occupancy_property = criteria.get("occupancy_property", "properties.OccupancyLoad")
            occupancy_result = get_property(element_id, occupancy_property)
            
            if occupancy_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Occupancy property '{occupancy_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_occupancy = float(occupancy_result["data"])
            
            # Find connected doors to calculate total exit width
            doors_result = find_related(element_id, "connected_doors")
            if doors_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Could not find connected doors for element {element_id}"]
                }
            
            total_exit_width = 0.0
            for door in doors_result["data"]:
                door_width_result = get_property(door["id"], "quantities.Width")
                if door_width_result["status"] == "success":
                    total_exit_width += float(door_width_result["data"])
            
            occupancy_valid = actual_occupancy <= max_occupancy
            exit_width_valid = total_exit_width >= min_exit_width
            is_valid = occupancy_valid and exit_width_valid
            
            message = f"Occupancy {actual_occupancy} ({'OK' if occupancy_valid else 'EXCEEDS'} max {max_occupancy}), Exit width {total_exit_width}m ({'OK' if exit_width_valid else 'INSUFFICIENT'}, min {min_exit_width}m)"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": {"occupancy": actual_occupancy, "exit_width": total_exit_width},
                    "message": message,
                    "details": {"connected_doors": len(doors_result["data"])}
                },
                "logs": logs
            }
        
        elif rule_type == "max_occupancy":
            max_persons = criteria.get("max_persons")
            if max_persons is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["max_occupancy rule requires max_persons in criteria"]
                }
            
            # Get current occupancy
            occupancy_property = criteria.get("occupancy_property", "properties.OccupancyLoad")
            occupancy_result = get_property(element_id, occupancy_property)
            
            if occupancy_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Occupancy property '{occupancy_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_occupancy = float(occupancy_result["data"])
            is_valid = actual_occupancy <= max_persons
            
            message = f"Occupancy {actual_occupancy} persons {'meets' if is_valid else 'exceeds'} maximum of {max_persons} persons"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_occupancy,
                    "message": message,
                    "details": {}
                },
                "logs": logs
            }
        
        elif rule_type == "occupancy_density":
            max_density = criteria.get("max_density")
            if max_density is None:
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["occupancy_density rule requires max_density in criteria"]
                }
            
            # Get current occupancy and area
            occupancy_property = criteria.get("occupancy_property", "properties.OccupancyLoad")
            area_property = criteria.get("area_property", "quantities.NetFloorArea")
            
            occupancy_result = get_property(element_id, occupancy_property)
            area_result = get_property(element_id, area_property)
            
            if occupancy_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Occupancy property '{occupancy_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            if area_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Area property '{area_property}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            actual_occupancy = float(occupancy_result["data"])
            actual_area = float(area_result["data"])
            
            if actual_area == 0:
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Area is zero for element {element_id}, cannot calculate density"]
                }
            
            actual_density = actual_occupancy / actual_area
            is_valid = actual_density <= max_density
            
            message = f"Occupancy density {actual_density:.2f} persons/m² {'meets' if is_valid else 'exceeds'} maximum of {max_density} persons/m²"
            logs.append(message)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_density,
                    "message": message,
                    "details": {"occupancy": actual_occupancy, "area": actual_area}
                },
                "logs": logs
            }
        
        elif rule_type == "property_range":
            property_path = criteria.get("property_path")
            min_value = criteria.get("min_value")
            max_value = criteria.get("max_value")
            
            if not property_path or (min_value is None and max_value is None):
                return {
                    "status": "error",
                    "data": None,
                    "logs": ["property_range rule requires property_path and at least one of min_value or max_value in criteria"]
                }
            
            prop_result = get_property(element_id, property_path)
            
            if prop_result["status"] != "success":
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Property '{property_path}' not found for element {element_id}. Use get_all_properties() to see available properties."]
                }
            
            try:
                actual_value = float(prop_result["data"])
            except (ValueError, TypeError):
                return {
                    "status": "error",
                    "data": None,
                    "logs": [f"Property '{property_path}' value '{prop_result['data']}' is not numeric"]
                }
            
            is_valid = True
            validation_details = []
            
            if min_value is not None:
                min_valid = actual_value >= min_value
                is_valid = is_valid and min_valid
                validation_details.append(f"min: {actual_value} >= {min_value} = {min_valid}")
            
            if max_value is not None:
                max_valid = actual_value <= max_value
                is_valid = is_valid and max_valid
                validation_details.append(f"max: {actual_value} <= {max_value} = {max_valid}")
            
            message = f"Property {property_path} = {actual_value} {'within' if is_valid else 'outside'} range [{min_value}, {max_value}]"
            logs.append(message)
            logs.extend(validation_details)
            
            return {
                "status": "success",
                "data": {
                    "is_valid": is_valid,
                    "rule_type": rule_type,
                    "element_id": element_id,
                    "criteria": criteria,
                    "actual_value": actual_value,
                    "message": message,
                    "details": {"validation_details": validation_details}
                },
                "logs": logs
            }
        
        else:
            return {
                "status": "error",
                "data": None,
                "logs": [f"Rule type '{rule_type}' not implemented yet"]
            }
    
    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "logs": [f"Error validating rule: {str(e)}"]
        }