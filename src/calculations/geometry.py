"""
Geometric calculations for AEC compliance verification.

This module provides functions for calculating areas, distances, centroids,
and other geometric properties needed for building code compliance checks.
"""

import math
from typing import List, Tuple, Optional
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
