"""
Geometry calculations for AEC Compliance Agent.

This module provides geometric operations for analyzing building floor plans,
including room boundaries, distances, adjacencies, and spatial relationships.
"""

import math
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import numpy as np

from ..schemas import Room, Door


def get_room_polygon(room: Room) -> Polygon:
    """
    Convert room boundary to Shapely polygon.
    
    Args:
        room: Room object with boundary points
        
    Returns:
        Shapely Polygon object representing the room
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        if len(room.boundary) < 3:
            raise ValueError(f"Room {room.id} boundary must have at least 3 points")
        
        # Ensure the polygon is closed (first point == last point)
        boundary = room.boundary.copy()
        if boundary[0] != boundary[-1]:
            boundary.append(boundary[0])
            
        polygon = Polygon(boundary)
        
        if not polygon.is_valid:
            # Try to fix invalid polygons
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                raise ValueError(f"Room {room.id} has invalid boundary geometry")
                
        return polygon
        
    except Exception as e:
        raise ValueError(f"Failed to create polygon for room {room.id}: {str(e)}")


def calculate_room_area(room: Room) -> float:
    """
    Calculate room area in square meters.
    
    Args:
        room: Room object
        
    Returns:
        Area in square meters
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        polygon = get_room_polygon(room)
        return float(polygon.area)
    except Exception as e:
        raise ValueError(f"Failed to calculate area for room {room.id}: {str(e)}")


def get_room_centroid(room: Room) -> Tuple[float, float]:
    """
    Get room center point (centroid).
    
    Args:
        room: Room object
        
    Returns:
        Tuple of (x, y) coordinates for room center
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        polygon = get_room_polygon(room)
        centroid = polygon.centroid
        return (float(centroid.x), float(centroid.y))
    except Exception as e:
        raise ValueError(f"Failed to calculate centroid for room {room.id}: {str(e)}")


def calculate_perimeter(room: Room) -> float:
    """
    Calculate room perimeter in meters.
    
    Args:
        room: Room object
        
    Returns:
        Perimeter length in meters
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        polygon = get_room_polygon(room)
        return float(polygon.length)
    except Exception as e:
        raise ValueError(f"Failed to calculate perimeter for room {room.id}: {str(e)}")


def point_in_room(point: Tuple[float, float], room: Room) -> bool:
    """
    Check if a point is inside a room.
    
    Args:
        point: Tuple of (x, y) coordinates
        room: Room object
        
    Returns:
        True if point is inside room, False otherwise
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        polygon = get_room_polygon(room)
        shapely_point = Point(point[0], point[1])
        return polygon.contains(shapely_point) or polygon.boundary.contains(shapely_point)
    except Exception as e:
        raise ValueError(f"Failed to check point in room {room.id}: {str(e)}")


def distance_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Distance in meters
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def rooms_are_adjacent(room1: Room, room2: Room, threshold: float = 0.5) -> bool:
    """
    Check if two rooms share a wall (are adjacent).
    
    This function checks if the rooms' boundaries are close enough to be considered
    adjacent, within the specified threshold distance.
    
    Args:
        room1: First room
        room2: Second room
        threshold: Maximum distance to consider rooms adjacent (meters)
        
    Returns:
        True if rooms are adjacent, False otherwise
        
    Raises:
        ValueError: If room boundaries are invalid
    """
    try:
        poly1 = get_room_polygon(room1)
        poly2 = get_room_polygon(room2)
        
        # Check if polygons touch or overlap
        if poly1.touches(poly2) or poly1.intersects(poly2):
            return True
            
        # Check if the closest distance is within threshold
        distance = poly1.distance(poly2)
        return distance <= threshold
        
    except Exception as e:
        raise ValueError(f"Failed to check adjacency between rooms {room1.id} and {room2.id}: {str(e)}")


def find_nearest_exit(position: Tuple[float, float], doors: List[Door]) -> Optional[Door]:
    """
    Find the nearest egress door from a given position.
    
    Args:
        position: Starting position (x, y)
        doors: List of available doors
        
    Returns:
        Nearest egress door, or None if no egress doors available
    """
    egress_doors = [door for door in doors if door.is_egress]
    
    if not egress_doors:
        return None
        
    nearest_door = None
    min_distance = float('inf')
    
    for door in egress_doors:
        door_position = (door.position[0], door.position[1])
        distance = distance_between_points(position, door_position)
        
        if distance < min_distance:
            min_distance = distance
            nearest_door = door
            
    return nearest_door


def get_room_boundary_segments(room: Room) -> List[LineString]:
    """
    Get room boundary as list of line segments.
    
    Args:
        room: Room object
        
    Returns:
        List of LineString objects representing boundary segments
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        polygon = get_room_polygon(room)
        coords = list(polygon.exterior.coords)
        
        segments = []
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            segments.append(segment)
            
        return segments
        
    except Exception as e:
        raise ValueError(f"Failed to get boundary segments for room {room.id}: {str(e)}")


def find_shared_boundary(room1: Room, room2: Room, tolerance: float = 0.1) -> Optional[LineString]:
    """
    Find shared boundary between two adjacent rooms.
    
    Args:
        room1: First room
        room2: Second room
        tolerance: Distance tolerance for considering boundaries shared
        
    Returns:
        LineString representing shared boundary, or None if not adjacent
        
    Raises:
        ValueError: If room boundaries are invalid
    """
    try:
        poly1 = get_room_polygon(room1)
        poly2 = get_room_polygon(room2)
        
        # Get the intersection of the boundaries
        intersection = poly1.boundary.intersection(poly2.boundary)
        
        if intersection.is_empty:
            return None
            
        # If intersection is a LineString, return it
        if hasattr(intersection, 'geom_type') and intersection.geom_type == 'LineString':
            return intersection
            
        # If intersection is a collection, find the longest LineString
        if hasattr(intersection, 'geoms'):
            longest_line = None
            max_length = 0
            
            for geom in intersection.geoms:
                if geom.geom_type == 'LineString' and geom.length > max_length:
                    max_length = geom.length
                    longest_line = geom
                    
            return longest_line
            
        return None
        
    except Exception as e:
        raise ValueError(f"Failed to find shared boundary between rooms {room1.id} and {room2.id}: {str(e)}")


def calculate_room_orientation(room: Room) -> float:
    """
    Calculate the primary orientation of a room in degrees.
    
    This finds the longest wall and returns its orientation relative to the x-axis.
    
    Args:
        room: Room object
        
    Returns:
        Orientation in degrees (0-180)
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        segments = get_room_boundary_segments(room)
        
        if not segments:
            return 0.0
            
        # Find the longest segment
        longest_segment = max(segments, key=lambda seg: seg.length)
        
        # Calculate orientation
        coords = list(longest_segment.coords)
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        
        # Calculate angle in degrees, normalized to 0-180
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 180
            
        return angle
        
    except Exception as e:
        raise ValueError(f"Failed to calculate orientation for room {room.id}: {str(e)}")


def get_room_bounding_box(room: Room) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of a room.
    
    Args:
        room: Room object
        
    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
        
    Raises:
        ValueError: If room boundary is invalid
    """
    try:
        polygon = get_room_polygon(room)
        bounds = polygon.bounds
        return bounds  # (min_x, min_y, max_x, max_y)
        
    except Exception as e:
        raise ValueError(f"Failed to get bounding box for room {room.id}: {str(e)}")


def calculate_door_clearance_area(door: Door, clearance_radius: float = 1.5) -> Polygon:
    """
    Calculate the clearance area around a door for accessibility analysis.
    
    Args:
        door: Door object
        clearance_radius: Required clearance radius in meters
        
    Returns:
        Polygon representing the clearance area
    """
    door_point = Point(door.position[0], door.position[1])
    clearance_area = door_point.buffer(clearance_radius)
    return clearance_area