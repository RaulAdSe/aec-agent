"""Building Data Toolkit - Generic functions for AEC building data operations.

This toolkit provides a minimal set of powerful, generic functions that can handle
all building data operations through flexible parameters and configurations.

The design philosophy is: "5 generic functions instead of 50 specific ones"
Each function is designed to be reusable across multiple compliance scenarios.

These functions are designed to be called directly by AI agents.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

# Global variable to store loaded building data
_building_data: Optional[Dict] = None
_logger = logging.getLogger(__name__)


def load_building_data(data_path: str) -> None:
    """Load building data from JSON file into global variable."""
    global _building_data
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            _building_data = json.load(f)
        _logger.info(f"Loaded building data from {data_path}")
        _logger.info(f"Total elements: {_building_data.get('file_info', {}).get('total_elements', 'unknown')}")
    except FileNotFoundError:
        _logger.error(f"Building data file not found: {data_path}")
        raise
    except json.JSONDecodeError as e:
        _logger.error(f"Invalid JSON in building data file: {e}")
        raise
    except Exception as e:
        _logger.error(f"Error loading building data: {e}")
        raise


# ===== CONVENIENCE TOOL: GET ALL PROPERTIES =====
def get_all_properties(element_id: str) -> Dict:
    """
    Get all properties and quantities for an element.
    
    Returns the complete property structure so agents can see
    all available data and choose appropriate properties based
    on their language knowledge.
    
    Args:
        element_id: Unique identifier of the element
        
    Returns:
        Dict with status, data (all element properties), and logs
        
    Examples:
        # Get all properties for analysis
        result = get_all_properties("door_123")
        if result["status"] == "success":
            properties = result["data"]["properties"]  
            quantities = result["data"]["quantities"]
            # Agent can now see all available fields
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


# ===== TOOL 1: GENERIC QUERY =====
def query_elements(element_type: str, filters: Optional[Dict] = None) -> Dict:
    """
    Query and filter building elements of any type.
    
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
        # Get all spaces
        query_elements("spaces")
        
        # Get spaces on level 1
        query_elements("spaces", {"properties.Ebene": "E01_OKRD"})
        
        # Get fire-rated doors
        query_elements("doors", {"properties.fire_rated": True})
        
        # Get large rooms (area > 50 sqm)
        query_elements("spaces", {"quantities.NetFloorArea__gt": 50})
        
        # Get external doors
        query_elements("doors", {"properties.IsExternal": True})
        
        # Get specific elements by ID list
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


# ===== TOOL 2: GENERIC PROPERTY GETTER =====
def get_property(element_id: str, property_path: str) -> Dict:
    """
    Extract any property from any building element.
    
    Uses dot notation to access nested properties safely.
    Handles missing properties gracefully.
    
    Args:
        element_id: Unique identifier of the element
        property_path: Dot-separated path to the property
            - "name" - Direct property
            - "properties.Fläche" - Nested in properties
            - "quantities.NetFloorArea" - Nested in quantities  
            - "geometry.bbox.min.0" - Deep nested with array index
            
    Returns:
        Property value, or None if not found
        
    Examples:
        # Get room name
        get_property("space_123", "name")
        
        # Get door width
        get_property("door_456", "quantities.Width")
        
        # Get room area
        get_property("space_123", "quantities.NetFloorArea")
        
        # Get geometry bounding box
        get_property("wall_789", "geometry.bbox.min")
        
        # Get custom property
        get_property("door_456", "properties.IsExternal")
        
        # Get specific coordinate
        get_property("space_123", "geometry.bbox.min.0")
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
                    return None
            else:
                # Handle dictionary keys
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
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