"""
Geometric calculations for AEC compliance verification.

This module provides minimal spatial reasoning primitives needed for building code compliance checks.
"""

import math
from typing import Dict, Any, List

from src.schemas import Point2D, Point3D, Boundary, Room, Door


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