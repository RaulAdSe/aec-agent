"""
Geometric calculations for AEC compliance verification.

This module provides functions for calculating areas, distances, centroids,
and other geometric properties needed for building code compliance checks.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import numpy as np

from src.schemas import Point2D, Point3D, Boundary, Room, Door


def calculate_polygon_area(points: List[Point2D]) -> float:
    """
    Calculate the area of a polygon using the shoelace formula.
    
    Args:
        points: List of 2D points forming the polygon boundary
        
    Returns:
        Area in square meters
        
    Raises:
        ValueError: If polygon has less than 3 points
    """
    if len(points) < 3:
        raise ValueError("Polygon must have at least 3 points")
    
    # Use Shapely for robust area calculation
    polygon_points = [(p.x, p.y) for p in points]
    polygon = Polygon(polygon_points)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon: self-intersecting or degenerate")
    
    return abs(polygon.area)


def calculate_polygon_perimeter(points: List[Point2D]) -> float:
    """
    Calculate the perimeter of a polygon.
    
    Args:
        points: List of 2D points forming the polygon boundary
        
    Returns:
        Perimeter in meters
    """
    if len(points) < 3:
        raise ValueError("Polygon must have at least 3 points")
    
    polygon_points = [(p.x, p.y) for p in points]
    polygon = Polygon(polygon_points)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon: self-intersecting or degenerate")
    
    return polygon.length


def calculate_polygon_centroid(points: List[Point2D]) -> Point2D:
    """
    Calculate the centroid (center of mass) of a polygon.
    
    Args:
        points: List of 2D points forming the polygon boundary
        
    Returns:
        Centroid as Point2D
        
    Raises:
        ValueError: If polygon has less than 3 points
    """
    if len(points) < 3:
        raise ValueError("Polygon must have at least 3 points")
    
    polygon_points = [(p.x, p.y) for p in points]
    polygon = Polygon(polygon_points)
    
    if not polygon.is_valid:
        raise ValueError("Invalid polygon: self-intersecting or degenerate")
    
    centroid = polygon.centroid
    return Point2D(x=centroid.x, y=centroid.y)


def calculate_distance_2d(point1: Point2D, point2: Point2D) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        point1: First point
        point2: Second point
        
    Returns:
        Distance in meters
    """
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    return math.sqrt(dx * dx + dy * dy)


def calculate_distance_3d(point1: Point3D, point2: Point3D) -> float:
    """
    Calculate Euclidean distance between two 3D points.
    
    Args:
        point1: First point
        point2: Second point
        
    Returns:
        Distance in meters
    """
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    dz = point2.z - point1.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_room_area(room: Room) -> float:
    """
    Calculate the area of a room.
    
    Args:
        room: Room object with boundary information
        
    Returns:
        Area in square meters
        
    Note:
        If room has a boundary, calculates from boundary.
        Otherwise, returns the stored area value.
    """
    if room.boundary and room.boundary.points:
        return calculate_polygon_area(room.boundary.points)
    else:
        return room.area


def calculate_room_centroid(room: Room) -> Optional[Point2D]:
    """
    Calculate the centroid of a room.
    
    Args:
        room: Room object with boundary information
        
    Returns:
        Centroid as Point2D, or None if no boundary available
    """
    if room.boundary and room.boundary.points:
        return calculate_polygon_centroid(room.boundary.points)
    return None


def calculate_room_perimeter(room: Room) -> Optional[float]:
    """
    Calculate the perimeter of a room.
    
    Args:
        room: Room object with boundary information
        
    Returns:
        Perimeter in meters, or None if no boundary available
    """
    if room.boundary and room.boundary.points:
        return calculate_polygon_perimeter(room.boundary.points)
    return None


def point_in_polygon(point: Point2D, polygon_points: List[Point2D]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: Point to test
        polygon_points: List of points forming the polygon boundary
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    if len(polygon_points) < 3:
        return False
    
    # Use Shapely for robust point-in-polygon test
    polygon_coords = [(p.x, p.y) for p in polygon_points]
    polygon = Polygon(polygon_coords)
    test_point = Point(point.x, point.y)
    
    return polygon.contains(test_point)


def calculate_door_clear_width(door: Door) -> float:
    """
    Calculate the clear width of a door (actual usable width).
    
    Args:
        door: Door object
        
    Returns:
        Clear width in millimeters
        
    Note:
        For single doors, clear width is the door width.
        For double doors, clear width is typically 2/3 of total width.
    """
    if door.door_type == "double":
        # Double doors: clear width is typically 2/3 of total width
        return door.width_mm * 0.67
    else:
        # Single doors: clear width is the door width
        return door.width_mm


def calculate_door_area(door: Door) -> float:
    """
    Calculate the area of a door opening.
    
    Args:
        door: Door object
        
    Returns:
        Door area in square meters
    """
    width_m = door.width_mm / 1000.0
    height_m = door.height_mm / 1000.0
    return width_m * height_m


def calculate_corridor_width(corridor_points: List[Point2D]) -> float:
    """
    Calculate the effective width of a corridor.
    
    Args:
        corridor_points: Points defining the corridor centerline
        
    Returns:
        Effective corridor width in meters
        
    Note:
        This is a simplified calculation. In practice, corridor width
        would need to account for wall thickness and obstructions.
    """
    if len(corridor_points) < 2:
        return 0.0
    
    # For now, assume a standard corridor width
    # In a real implementation, this would analyze the corridor geometry
    return 1.2  # 1.2 meters standard corridor width


def calculate_egress_capacity(area_sqm: float, occupancy_type: str) -> int:
    """
    Calculate the maximum occupancy capacity for egress calculations.
    
    Args:
        area_sqm: Area in square meters
        occupancy_type: Type of occupancy (residential, commercial, etc.)
        
    Returns:
        Maximum occupancy capacity
        
    Note:
        This uses simplified occupancy factors. Real calculations would
        reference specific building codes and occupancy classifications.
    """
    # Simplified occupancy factors (persons per square meter)
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
    
    factor = occupancy_factors.get(occupancy_type, 0.1)  # Default factor
    return max(1, int(area_sqm * factor))


def calculate_fire_rating_equivalent(rating: str) -> int:
    """
    Convert fire rating string to minutes.
    
    Args:
        rating: Fire rating string (e.g., "RF_60")
        
    Returns:
        Fire rating in minutes
    """
    rating_map = {
        "no_rating": 0,
        "RF_30": 30,
        "RF_60": 60,
        "RF_90": 90,
        "RF_120": 120,
    }
    
    return rating_map.get(rating, 0)


def calculate_wall_length(wall_start: Point3D, wall_end: Point3D) -> float:
    """
    Calculate the length of a wall.
    
    Args:
        wall_start: Wall start point
        wall_end: Wall end point
        
    Returns:
        Wall length in meters
    """
    return calculate_distance_3d(wall_start, wall_end)


def calculate_wall_area(wall_start: Point3D, wall_end: Point3D, height_mm: float) -> float:
    """
    Calculate the area of a wall.
    
    Args:
        wall_start: Wall start point
        wall_end: Wall end point
        height_mm: Wall height in millimeters
        
    Returns:
        Wall area in square meters
    """
    length_m = calculate_wall_length(wall_start, wall_end)
    height_m = height_mm / 1000.0
    return length_m * height_m


def calculate_intersection_area(polygon1_points: List[Point2D], polygon2_points: List[Point2D]) -> float:
    """
    Calculate the intersection area between two polygons.
    
    Args:
        polygon1_points: First polygon points
        polygon2_points: Second polygon points
        
    Returns:
        Intersection area in square meters
    """
    if len(polygon1_points) < 3 or len(polygon2_points) < 3:
        return 0.0
    
    try:
        poly1 = Polygon([(p.x, p.y) for p in polygon1_points])
        poly2 = Polygon([(p.x, p.y) for p in polygon2_points])
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2)
        return abs(intersection.area)
    
    except Exception:
        return 0.0


def calculate_union_area(polygon1_points: List[Point2D], polygon2_points: List[Point2D]) -> float:
    """
    Calculate the union area of two polygons.
    
    Args:
        polygon1_points: First polygon points
        polygon2_points: Second polygon points
        
    Returns:
        Union area in square meters
    """
    if len(polygon1_points) < 3 or len(polygon2_points) < 3:
        return 0.0
    
    try:
        poly1 = Polygon([(p.x, p.y) for p in polygon1_points])
        poly2 = Polygon([(p.x, p.y) for p in polygon2_points])
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        union = unary_union([poly1, poly2])
        return abs(union.area)
    
    except Exception:
        return 0.0


# ============================================================================
# ADVANCED SPATIAL ANALYSIS FUNCTIONS FOR AGENT USE
# ============================================================================

def calculate_room_adjacency_matrix(rooms: List[Room]) -> Dict[str, List[str]]:
    """
    Calculate which rooms are adjacent to each other based on shared walls.
    
    Args:
        rooms: List of Room objects
        
    Returns:
        Dictionary mapping room_id to list of adjacent room_ids
    """
    adjacency = {room.id: [] for room in rooms}
    
    for i, room1 in enumerate(rooms):
        if not room1.boundary or not room1.boundary.points:
            continue
            
        poly1 = Polygon([(p.x, p.y) for p in room1.boundary.points])
        if not poly1.is_valid:
            continue
            
        for j, room2 in enumerate(rooms[i+1:], i+1):
            if not room2.boundary or not room2.boundary.points:
                continue
                
            poly2 = Polygon([(p.x, p.y) for p in room2.boundary.points])
            if not poly2.is_valid:
                continue
            
            # Check if rooms share a boundary (are adjacent)
            intersection = poly1.intersection(poly2)
            
            # If intersection is a line (has length but no area), rooms are adjacent
            if hasattr(intersection, 'length') and intersection.length > 0.1:  # 10cm tolerance
                adjacency[room1.id].append(room2.id)
                adjacency[room2.id].append(room1.id)
    
    return adjacency


def calculate_sight_line_analysis(start_point: Point2D, target_point: Point2D, 
                                obstacles: List[List[Point2D]]) -> Dict[str, any]:
    """
    Check if there's a clear sight line between two points, considering obstacles.
    
    Args:
        start_point: Starting point
        target_point: Target point
        obstacles: List of polygon obstacles (walls, furniture, etc.)
        
    Returns:
        Dictionary with sight line analysis results
    """
    try:
        # Create sight line
        sight_line = LineString([(start_point.x, start_point.y), 
                                (target_point.x, target_point.y)])
        
        blocked_by = []
        total_obstruction = 0.0
        
        for i, obstacle_points in enumerate(obstacles):
            if len(obstacle_points) < 3:
                continue
                
            obstacle = Polygon([(p.x, p.y) for p in obstacle_points])
            if not obstacle.is_valid:
                continue
            
            # Check if sight line intersects with obstacle
            intersection = sight_line.intersection(obstacle)
            if intersection and not intersection.is_empty:
                blocked_by.append(f"obstacle_{i}")
                
                # Calculate length of obstruction
                if hasattr(intersection, 'length'):
                    total_obstruction += intersection.length
        
        is_clear = len(blocked_by) == 0
        total_distance = sight_line.length
        obstruction_ratio = total_obstruction / total_distance if total_distance > 0 else 0
        
        return {
            "is_clear": is_clear,
            "total_distance": total_distance,
            "blocked_by": blocked_by,
            "obstruction_length": total_obstruction,
            "obstruction_ratio": obstruction_ratio,
            "visibility_score": max(0, 1 - obstruction_ratio)
        }
        
    except Exception as e:
        return {
            "is_clear": False,
            "error": str(e),
            "total_distance": 0,
            "blocked_by": [],
            "obstruction_length": 0,
            "obstruction_ratio": 1.0,
            "visibility_score": 0.0
        }


def calculate_compartmentation_analysis(rooms: List[Room], 
                                      fire_walls: List[Dict]) -> Dict[str, any]:
    """
    Analyze fire compartmentation and separation.
    
    Args:
        rooms: List of Room objects
        fire_walls: List of wall dictionaries with fire ratings
        
    Returns:
        Dictionary with compartmentation analysis
    """
    compartments = []
    fire_rated_boundaries = []
    
    # Group rooms by fire compartment
    # This is a simplified analysis - in reality would need more complex logic
    for room in rooms:
        if room.fire_rating and room.fire_rating != "no_rating":
            compartments.append({
                "room_id": room.id,
                "fire_rating": room.fire_rating,
                "area": room.area,
                "use": room.use
            })
    
    # Analyze fire-rated walls
    for wall in fire_walls:
        fire_rating = wall.get('fire_rating', 'no_rating')
        if fire_rating != 'no_rating':
            fire_rated_boundaries.append({
                "wall_id": wall.get('id', 'unknown'),
                "fire_rating": fire_rating,
                "rating_minutes": calculate_fire_rating_equivalent(fire_rating)
            })
    
    return {
        "compartment_count": len(compartments),
        "compartments": compartments,
        "fire_rated_walls": len(fire_rated_boundaries),
        "fire_boundaries": fire_rated_boundaries,
        "compliance_notes": [
            "Fire compartmentation limits building size",
            "Rated walls must maintain continuity",
            "Openings require fire-rated assemblies"
        ]
    }


def calculate_corridor_analysis(corridor_points: List[Point2D], 
                               min_width: float = 1200.0) -> Dict[str, any]:
    """
    Analyze corridor dimensions and compliance.
    
    Args:
        corridor_points: Points defining corridor centerline or boundary
        min_width: Minimum required width in millimeters
        
    Returns:
        Dictionary with corridor analysis results
    """
    if len(corridor_points) < 2:
        return {
            "error": "Insufficient points for corridor analysis",
            "is_compliant": False
        }
    
    try:
        # Calculate corridor length
        total_length = 0.0
        for i in range(len(corridor_points) - 1):
            segment_length = calculate_distance_2d(corridor_points[i], corridor_points[i+1])
            total_length += segment_length
        
        # For simplified analysis, assume standard corridor width
        # In practice, this would analyze the actual corridor polygon
        estimated_width = calculate_corridor_width(corridor_points)
        width_mm = estimated_width * 1000  # Convert to mm
        
        is_compliant = width_mm >= min_width
        
        return {
            "length_m": total_length,
            "width_mm": width_mm,
            "min_required_width_mm": min_width,
            "is_compliant": is_compliant,
            "area_sqm": total_length * estimated_width,
            "compliance_status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
            "width_deficit_mm": max(0, min_width - width_mm) if not is_compliant else 0
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "is_compliant": False
        }


def calculate_spatial_relationships(rooms: List[Room]) -> Dict[str, any]:
    """
    Calculate comprehensive spatial relationships between rooms.
    
    Args:
        rooms: List of Room objects
        
    Returns:
        Dictionary with spatial relationship analysis
    """
    relationships = {
        "adjacency_matrix": calculate_room_adjacency_matrix(rooms),
        "room_distances": {},
        "centroids": {},
        "size_relationships": {},
        "spatial_clusters": []
    }
    
    # Calculate centroids and distances
    for room in rooms:
        if room.boundary and room.boundary.points:
            centroid = calculate_room_centroid(room)
            if centroid:
                relationships["centroids"][room.id] = {
                    "x": centroid.x,
                    "y": centroid.y
                }
    
    # Calculate distances between all rooms
    room_ids = list(relationships["centroids"].keys())
    for i, room1_id in enumerate(room_ids):
        relationships["room_distances"][room1_id] = {}
        centroid1 = relationships["centroids"][room1_id]
        
        for room2_id in room_ids[i+1:]:
            centroid2 = relationships["centroids"][room2_id]
            distance = math.sqrt(
                (centroid2["x"] - centroid1["x"])**2 + 
                (centroid2["y"] - centroid1["y"])**2
            )
            relationships["room_distances"][room1_id][room2_id] = distance
            
            # Add reverse mapping
            if room2_id not in relationships["room_distances"]:
                relationships["room_distances"][room2_id] = {}
            relationships["room_distances"][room2_id][room1_id] = distance
    
    # Analyze size relationships
    areas = [(room.id, calculate_room_area(room)) for room in rooms]
    areas.sort(key=lambda x: x[1], reverse=True)
    
    total_area = sum(area for _, area in areas)
    for room_id, area in areas:
        relationships["size_relationships"][room_id] = {
            "area_sqm": area,
            "percentage_of_total": (area / total_area * 100) if total_area > 0 else 0,
            "size_category": "large" if area > total_area * 0.2 else "medium" if area > total_area * 0.1 else "small"
        }
    
    return relationships


def calculate_bottleneck_analysis(circulation_points: List[Point2D], 
                                 door_widths: List[float]) -> Dict[str, any]:
    """
    Identify potential bottlenecks in circulation paths.
    
    Args:
        circulation_points: Points along circulation path
        door_widths: Widths of doors along the path (in mm)
        
    Returns:
        Dictionary with bottleneck analysis
    """
    if not door_widths:
        return {
            "bottlenecks": [],
            "min_width_mm": 0,
            "bottleneck_count": 0
        }
    
    min_width = min(door_widths)
    avg_width = sum(door_widths) / len(door_widths)
    
    # Identify bottlenecks (doors significantly narrower than average)
    bottlenecks = []
    threshold = avg_width * 0.8  # 20% below average
    
    for i, width in enumerate(door_widths):
        if width < threshold:
            bottlenecks.append({
                "position_index": i,
                "width_mm": width,
                "deficit_from_average": avg_width - width,
                "severity": "high" if width < min_width * 1.1 else "medium"
            })
    
    return {
        "bottlenecks": bottlenecks,
        "bottleneck_count": len(bottlenecks),
        "min_width_mm": min_width,
        "avg_width_mm": avg_width,
        "width_variance": np.var(door_widths) if len(door_widths) > 1 else 0,
        "flow_efficiency": max(0, 1 - len(bottlenecks) / len(door_widths))
    }


# ============================================================================
# SPATIAL REASONING PRIMITIVES FOR AGENT USE
# ============================================================================

def calculate_angle_between_walls(wall1: Dict, wall2: Dict) -> Dict[str, Any]:
    """
    Calculate the angle between two walls.
    
    Args:
        wall1: Wall dictionary with start_point and end_point
        wall2: Wall dictionary with start_point and end_point
        
    Returns:
        Dictionary with angle information and success status
    """
    try:
        if not wall1 or not wall2:
            return {'success': False, 'error': 'Both walls must be provided'}
        
        if 'start_point' not in wall1 or 'end_point' not in wall1:
            return {'success': False, 'error': 'wall1 must have start_point and end_point'}
        
        if 'start_point' not in wall2 or 'end_point' not in wall2:
            return {'success': False, 'error': 'wall2 must have start_point and end_point'}
        
        # Calculate wall vectors
        w1_start = wall1['start_point']
        w1_end = wall1['end_point']
        w2_start = wall2['start_point']
        w2_end = wall2['end_point']
        
        # Vector for wall1
        v1 = (w1_end['x'] - w1_start['x'], w1_end['y'] - w1_start['y'])
        # Vector for wall2
        v2 = (w2_end['x'] - w2_start['x'], w2_end['y'] - w2_start['y'])
        
        # Calculate angle using dot product
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return {'success': False, 'error': 'One or both walls have zero length'}
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to avoid numerical errors
        
        angle_radians = math.acos(cos_angle)
        angle_degrees = math.degrees(angle_radians)
        
        # Determine relationship
        if abs(angle_degrees - 0) < 5 or abs(angle_degrees - 180) < 5:
            relationship = "parallel"
        elif abs(angle_degrees - 90) < 5:
            relationship = "perpendicular"
        else:
            relationship = "angled"
        
        return {
            'success': True,
            'angle_degrees': angle_degrees,
            'angle_radians': angle_radians,
            'relationship': relationship,
            'wall1_vector': v1,
            'wall2_vector': v2,
            'error': None
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Angle calculation failed: {str(e)}'}


def find_nearest_door(point: Dict, doors: List[Dict]) -> Dict[str, Any]:
    """
    Find the nearest door to a given point.
    
    Args:
        point: Point dictionary with x, y coordinates
        doors: List of door dictionaries with position
        
    Returns:
        Dictionary with nearest door information
    """
    try:
        if not point or 'x' not in point or 'y' not in point:
            return {'success': False, 'error': 'Point must have x and y coordinates'}
        
        if not doors:
            return {'success': False, 'error': 'No doors provided'}
        
        nearest_door = None
        min_distance = float('inf')
        
        for door in doors:
            if 'position' not in door:
                continue
            
            door_pos = door['position']
            if 'x' not in door_pos or 'y' not in door_pos:
                continue
            
            # Calculate distance
            dx = point['x'] - door_pos['x']
            dy = point['y'] - door_pos['y']
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_door = door
        
        if nearest_door is None:
            return {'success': False, 'error': 'No valid doors found'}
        
        return {
            'success': True,
            'nearest_door': nearest_door,
            'distance_m': min_distance,
            'door_id': nearest_door.get('id', 'unknown'),
            'door_position': nearest_door['position'],
            'error': None
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Nearest door search failed: {str(e)}'}


def find_walls_within_radius(point: Dict, walls: List[Dict], radius_m: float) -> Dict[str, Any]:
    """
    Find all walls within a specified radius of a point.
    
    Args:
        point: Point dictionary with x, y coordinates
        walls: List of wall dictionaries with start_point and end_point
        radius_m: Search radius in meters
        
    Returns:
        Dictionary with walls within radius
    """
    try:
        if not point or 'x' not in point or 'y' not in point:
            return {'success': False, 'error': 'Point must have x and y coordinates'}
        
        if not walls:
            return {'success': False, 'error': 'No walls provided'}
        
        if radius_m <= 0:
            return {'success': False, 'error': 'Radius must be positive'}
        
        walls_within_radius = []
        
        for wall in walls:
            if 'start_point' not in wall or 'end_point' not in wall:
                continue
            
            start = wall['start_point']
            end = wall['end_point']
            
            # Calculate distance from point to wall (simplified: distance to wall midpoint)
            wall_mid_x = (start['x'] + end['x']) / 2
            wall_mid_y = (start['y'] + end['y']) / 2
            
            dx = point['x'] - wall_mid_x
            dy = point['y'] - wall_mid_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance <= radius_m:
                walls_within_radius.append({
                    'wall': wall,
                    'distance_m': distance,
                    'wall_id': wall.get('id', 'unknown')
                })
        
        return {
            'success': True,
            'walls_within_radius': walls_within_radius,
            'count': len(walls_within_radius),
            'radius_m': radius_m,
            'query_point': point,
            'error': None
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Radius search failed: {str(e)}'}


def calculate_wall_cardinal_direction(wall: Dict) -> Dict[str, Any]:
    """
    Calculate the cardinal direction of a wall.
    
    Args:
        wall: Wall dictionary with start_point and end_point
        
    Returns:
        Dictionary with cardinal direction information
    """
    try:
        if not wall:
            return {'success': False, 'error': 'Wall must be provided'}
        
        if 'start_point' not in wall or 'end_point' not in wall:
            return {'success': False, 'error': 'Wall must have start_point and end_point'}
        
        start = wall['start_point']
        end = wall['end_point']
        
        # Calculate wall vector
        dx = end['x'] - start['x']
        dy = end['y'] - start['y']
        
        if dx == 0 and dy == 0:
            return {'success': False, 'error': 'Wall has zero length'}
        
        # Calculate angle from positive X-axis (East)
        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        
        # Normalize to 0-360 degrees
        if angle_degrees < 0:
            angle_degrees += 360
        
        # Determine cardinal direction
        if 337.5 <= angle_degrees or angle_degrees < 22.5:
            direction = "E"  # East
        elif 22.5 <= angle_degrees < 67.5:
            direction = "NE"  # Northeast
        elif 67.5 <= angle_degrees < 112.5:
            direction = "N"  # North
        elif 112.5 <= angle_degrees < 157.5:
            direction = "NW"  # Northwest
        elif 157.5 <= angle_degrees < 202.5:
            direction = "W"  # West
        elif 202.5 <= angle_degrees < 247.5:
            direction = "SW"  # Southwest
        elif 247.5 <= angle_degrees < 292.5:
            direction = "S"  # South
        elif 292.5 <= angle_degrees < 337.5:
            direction = "SE"  # Southeast
        
        return {
            'success': True,
            'cardinal_direction': direction,
            'angle_degrees': angle_degrees,
            'wall_vector': (dx, dy),
            'wall_id': wall.get('id', 'unknown'),
            'error': None
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Cardinal direction calculation failed: {str(e)}'}


def calculate_clearance_between_elements(elem1: Dict, elem2: Dict) -> Dict[str, Any]:
    """
    Calculate the minimum clearance (shortest distance) between two elements.
    
    Args:
        elem1: First element (door or wall) with position/start_point/end_point
        elem2: Second element (door or wall) with position/start_point/end_point
        
    Returns:
        Dictionary with clearance information
    """
    try:
        if not elem1 or not elem2:
            return {'success': False, 'error': 'Both elements must be provided'}
        
        # Extract points from elements
        points1 = []
        points2 = []
        
        # Handle doors (have position)
        if 'position' in elem1:
            pos = elem1['position']
            points1 = [{'x': pos['x'], 'y': pos['y']}]
        elif 'start_point' in elem1 and 'end_point' in elem1:
            points1 = [elem1['start_point'], elem1['end_point']]
        
        if 'position' in elem2:
            pos = elem2['position']
            points2 = [{'x': pos['x'], 'y': pos['y']}]
        elif 'start_point' in elem2 and 'end_point' in elem2:
            points2 = [elem2['start_point'], elem2['end_point']]
        
        if not points1 or not points2:
            return {'success': False, 'error': 'Elements must have position or start/end points'}
        
        # Calculate minimum distance between all point pairs
        min_distance = float('inf')
        closest_points = None
        
        for p1 in points1:
            for p2 in points2:
                dx = p1['x'] - p2['x']
                dy = p1['y'] - p2['y']
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_points = (p1, p2)
        
        return {
            'success': True,
            'clearance_m': min_distance,
            'closest_points': closest_points,
            'elem1_id': elem1.get('id', 'unknown'),
            'elem2_id': elem2.get('id', 'unknown'),
            'error': None
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Clearance calculation failed: {str(e)}'}


# Public export surface (minimal toolkit for notebooks)
__all__ = [
    'calculate_angle_between_walls',
    'find_nearest_door',
    'find_walls_within_radius',
    'calculate_wall_cardinal_direction',
    'calculate_clearance_between_elements',
]
