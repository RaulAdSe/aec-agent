"""
Graph-based calculations for circulation analysis.

This module provides minimal circulation graph functionality needed for building visualization.
"""

import networkx as nx
import math
from typing import List, Dict, Tuple, Optional, Set

from src.schemas import Point2D, Point3D, Room, Door, Project


class CirculationGraph:
    """
    Minimal graph representation of building circulation for visualization.
    
    This class creates a network graph where:
    - Nodes represent rooms
    - Edges represent doors and circulation paths
    - Weights represent distances
    """
    
    def __init__(self, project: Project):
        """
        Initialize circulation graph from project data.
        
        Args:
            project: Project object containing rooms, doors, and levels
        """
        self.project = project
        self.graph = nx.Graph()
        self.room_positions = {}
        self.door_positions = {}
        self.exit_nodes = set()
        
        self._build_graph()
    
    def _build_graph(self):
        """Build the circulation graph from project data."""
        # Add all rooms as nodes
        for level in self.project.levels:
            for room in level.rooms:
                node_id = f"room_{room.id}"
                self.graph.add_node(node_id, 
                                  type="room",
                                  room_id=room.id,
                                  room_name=room.name,
                                  area=room.area,
                                  use=room.use,
                                  level=level.name)
                
                # Store room position (use centroid if available)
                if room.boundary and room.boundary.points:
                    centroid = self._calculate_centroid(room.boundary.points)
                    self.room_positions[room.id] = centroid
                else:
                    # Use a default position if no boundary available
                    self.room_positions[room.id] = Point2D(x=0, y=0)
        
        # Add doors as edges between rooms
        for level in self.project.levels:
            for door in level.doors:
                if door.from_room and door.to_room:
                    # Add edge between rooms
                    from_node = f"room_{door.from_room}"
                    to_node = f"room_{door.to_room}"
                    
                    # Calculate distance via the actual door position (centroid -> door -> centroid)
                    distance = self._calculate_door_constrained_distance(
                        door.from_room, door.to_room, door
                    )
                    
                    self.graph.add_edge(from_node, to_node,
                                      door_id=door.id,
                                      door_width=door.width_mm,
                                      door_type=door.door_type,
                                      is_emergency_exit=door.is_emergency_exit,
                                      weight=distance)
                    
                    # Store door position
                    self.door_positions[door.id] = door.position
        
        # Identify exit nodes (rooms with emergency exit doors)
        for level in self.project.levels:
            for door in level.doors:
                if door.is_emergency_exit:
                    if door.to_room:
                        exit_node = f"room_{door.to_room}"
                        self.exit_nodes.add(exit_node)
                        self.graph.nodes[exit_node]['is_exit'] = True
                    if door.from_room:
                        exit_node = f"room_{door.from_room}"
                        self.exit_nodes.add(exit_node)
                        self.graph.nodes[exit_node]['is_exit'] = True
    
    def _calculate_centroid(self, points: List[Point2D]) -> Point2D:
        """Calculate centroid of a polygon."""
        if not points:
            return Point2D(x=0, y=0)
        
        x_sum = sum(p.x for p in points)
        y_sum = sum(p.y for p in points)
        n = len(points)
        
        return Point2D(x=x_sum / n, y=y_sum / n)
    
    def _calculate_room_distance(self, room1_id: str, room2_id: str) -> float:
        """Calculate distance between two room centroids."""
        pos1 = self.room_positions.get(room1_id, Point2D(x=0, y=0))
        pos2 = self.room_positions.get(room2_id, Point2D(x=0, y=0))
        
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        return math.sqrt(dx * dx + dy * dy)

    def _calculate_door_constrained_distance(self, room1_id: str, room2_id: str, door: Door) -> float:
        """Calculate distance from room1 centroid to door to room2 centroid.

        Falls back to centroid-to-centroid if door position is missing.
        """
        pos1 = self.room_positions.get(room1_id, Point2D(x=0, y=0))
        pos2 = self.room_positions.get(room2_id, Point2D(x=0, y=0))

        if door and getattr(door, 'position', None):
            dp = door.position
            # distance room1 centroid -> door
            dx1 = dp.x - pos1.x
            dy1 = dp.y - pos1.y
            d1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            # distance door -> room2 centroid
            dx2 = pos2.x - dp.x
            dy2 = pos2.y - dp.y
            d2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            return d1 + d2
        # Fallback
        return self._calculate_room_distance(room1_id, room2_id)


def create_circulation_graph(project: Project) -> CirculationGraph:
    """
    Create a circulation graph from project data.
    
    Args:
        project: Project object containing building data
        
    Returns:
        CirculationGraph instance
    """
    return CirculationGraph(project)


# Public export surface (minimal toolkit for notebooks)
__all__ = [
    'CirculationGraph',
    'create_circulation_graph',
]