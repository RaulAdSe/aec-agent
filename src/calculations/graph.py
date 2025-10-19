"""
Graph-based calculations for circulation and egress analysis.

This module provides functions for creating circulation graphs and calculating
egress distances, evacuation routes, and accessibility compliance.
"""

import networkx as nx
import math
from typing import List, Dict, Tuple, Optional, Set
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
import numpy as np

from src.schemas import Point2D, Point3D, Room, Door, Project


class CirculationGraph:
    """
    Graph representation of building circulation for egress analysis.
    
    This class creates a network graph where:
    - Nodes represent rooms, corridors, and exits
    - Edges represent doors and circulation paths
    - Weights represent distances or travel times
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
                    
                    # Calculate distance between room centroids
                    distance = self._calculate_room_distance(door.from_room, door.to_room)
                    
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
    
    def calculate_egress_distance(self, room_id: str) -> Dict[str, any]:
        """
        Calculate the shortest egress distance from a room to the nearest exit.
        
        Args:
            room_id: ID of the room to calculate egress distance for
            
        Returns:
            Dictionary containing:
            - distance: Shortest distance to exit in meters
            - exit_room_id: ID of the nearest exit room
            - path: List of room IDs in the egress path
            - is_accessible: Whether the path is accessible
        """
        room_node = f"room_{room_id}"
        
        if room_node not in self.graph:
            return {
                "distance": float('inf'),
                "exit_room_id": None,
                "path": [],
                "is_accessible": False,
                "error": f"Room {room_id} not found in graph"
            }
        
        if not self.exit_nodes:
            return {
                "distance": float('inf'),
                "exit_room_id": None,
                "path": [],
                "is_accessible": False,
                "error": "No exit nodes found in building"
            }
        
        # Find shortest path to any exit
        shortest_distance = float('inf')
        shortest_path = []
        nearest_exit = None
        
        for exit_node in self.exit_nodes:
            try:
                if nx.has_path(self.graph, room_node, exit_node):
                    path = nx.shortest_path(self.graph, room_node, exit_node, weight='weight')
                    distance = nx.shortest_path_length(self.graph, room_node, exit_node, weight='weight')
                    
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_path = path
                        nearest_exit = exit_node
            except nx.NetworkXNoPath:
                continue
        
        # Check if path is accessible (all doors meet minimum width requirements)
        is_accessible = self._check_path_accessibility(shortest_path)
        
        return {
            "distance": shortest_distance,
            "exit_room_id": nearest_exit.replace("room_", "") if nearest_exit else None,
            "path": [node.replace("room_", "") for node in shortest_path],
            "is_accessible": is_accessible,
            "error": None
        }
    
    def _check_path_accessibility(self, path: List[str]) -> bool:
        """
        Check if a path is accessible (all doors meet minimum width requirements).
        
        Args:
            path: List of node IDs in the path
            
        Returns:
            True if path is accessible, False otherwise
        """
        if len(path) < 2:
            return True
        
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            
            if self.graph.has_edge(node1, node2):
                edge_data = self.graph[node1][node2]
                door_width = edge_data.get('door_width', 0)
                
                # Minimum accessible door width is 800mm (ADA standard)
                if door_width < 800:
                    return False
        
        return True
    
    def calculate_egress_capacity(self, room_id: str) -> int:
        """
        Calculate the egress capacity for a room based on its area and use.
        
        Args:
            room_id: ID of the room
            
        Returns:
            Maximum occupancy capacity for egress calculations
        """
        room_node = f"room_{room_id}"
        
        if room_node not in self.graph.nodes:
            return 0
        
        node_data = self.graph.nodes[room_node]
        area = node_data.get('area', 0)
        use = node_data.get('use', 'commercial')
        
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
        
        factor = occupancy_factors.get(use, 0.1)  # Default factor
        return max(1, int(area * factor))
    
    def get_room_connections(self, room_id: str) -> List[Dict[str, any]]:
        """
        Get all rooms connected to a given room.
        
        Args:
            room_id: ID of the room
            
        Returns:
            List of dictionaries containing connection information
        """
        room_node = f"room_{room_id}"
        
        if room_node not in self.graph:
            return []
        
        connections = []
        for neighbor in self.graph.neighbors(room_node):
            edge_data = self.graph[room_node][neighbor]
            neighbor_room_id = neighbor.replace("room_", "")
            
            connections.append({
                "room_id": neighbor_room_id,
                "door_id": edge_data.get('door_id'),
                "door_width": edge_data.get('door_width'),
                "door_type": edge_data.get('door_type'),
                "distance": edge_data.get('weight', 0),
                "is_emergency_exit": edge_data.get('is_emergency_exit', False)
            })
        
        return connections
    
    def find_egress_routes(self, room_id: str, max_routes: int = 3) -> List[Dict[str, any]]:
        """
        Find multiple egress routes from a room to exits.
        
        Args:
            room_id: ID of the room
            max_routes: Maximum number of routes to find
            
        Returns:
            List of route dictionaries
        """
        room_node = f"room_{room_id}"
        
        if room_node not in self.graph or not self.exit_nodes:
            return []
        
        routes = []
        
        for exit_node in self.exit_nodes:
            try:
                if nx.has_path(self.graph, room_node, exit_node):
                    # Find all simple paths (no repeated nodes)
                    paths = list(nx.all_simple_paths(
                        self.graph, room_node, exit_node, cutoff=10
                    ))
                    
                    # Sort by length and take the shortest ones
                    paths.sort(key=len)
                    
                    for path in paths[:max_routes]:
                        distance = sum(
                            self.graph[path[i]][path[i+1]]['weight'] 
                            for i in range(len(path) - 1)
                        )
                        
                        is_accessible = self._check_path_accessibility(path)
                        
                        routes.append({
                            "path": [node.replace("room_", "") for node in path],
                            "distance": distance,
                            "exit_room_id": exit_node.replace("room_", ""),
                            "is_accessible": is_accessible,
                            "path_length": len(path)
                        })
                        
                        if len(routes) >= max_routes:
                            break
                            
            except nx.NetworkXNoPath:
                continue
        
        # Sort routes by distance
        routes.sort(key=lambda x: x['distance'])
        return routes[:max_routes]
    
    def calculate_travel_time(self, room_id: str, walking_speed: float = 1.2) -> Dict[str, any]:
        """
        Calculate travel time to nearest exit.
        
        Args:
            room_id: ID of the room
            walking_speed: Walking speed in meters per second (default 1.2 m/s)
            
        Returns:
            Dictionary with travel time information
        """
        egress_info = self.calculate_egress_distance(room_id)
        
        if egress_info['distance'] == float('inf'):
            return {
                "travel_time_seconds": float('inf'),
                "travel_time_minutes": float('inf'),
                "distance": float('inf'),
                "error": egress_info.get('error', 'No path to exit found')
            }
        
        travel_time_seconds = egress_info['distance'] / walking_speed
        travel_time_minutes = travel_time_seconds / 60.0
        
        return {
            "travel_time_seconds": travel_time_seconds,
            "travel_time_minutes": travel_time_minutes,
            "distance": egress_info['distance'],
            "exit_room_id": egress_info['exit_room_id'],
            "path": egress_info['path'],
            "is_accessible": egress_info['is_accessible']
        }
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the circulation graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "exit_nodes": len(self.exit_nodes),
            "connected_components": nx.number_connected_components(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }


def create_circulation_graph(project: Project) -> CirculationGraph:
    """
    Create a circulation graph from project data.
    
    Args:
        project: Project object containing building data
        
    Returns:
        CirculationGraph instance
    """
    return CirculationGraph(project)


def calculate_egress_distance(project: Project, room_id: str) -> Dict[str, any]:
    """
    Calculate egress distance for a room using the circulation graph.
    
    Args:
        project: Project object
        room_id: ID of the room
        
    Returns:
        Egress distance information
    """
    graph = create_circulation_graph(project)
    return graph.calculate_egress_distance(room_id)


def calculate_travel_time(project: Project, room_id: str, walking_speed: float = 1.2) -> Dict[str, any]:
    """
    Calculate travel time to nearest exit.
    
    Args:
        project: Project object
        room_id: ID of the room
        walking_speed: Walking speed in meters per second
        
    Returns:
        Travel time information
    """
    graph = create_circulation_graph(project)
    return graph.calculate_travel_time(room_id, walking_speed)
