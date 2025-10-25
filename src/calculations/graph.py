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


def get_room_adjacency_list(project: Project) -> Dict[str, List[str]]:
    """
    Get a simple adjacency list of which rooms connect to which.
    
    Args:
        project: Project object
        
    Returns:
        Dictionary mapping room_id to list of connected room_ids
    """
    graph = create_circulation_graph(project)
    adjacency = {}
    
    for level in project.levels:
        for room in level.rooms:
            room_id = room.id
            connections = graph.get_room_connections(room_id)
            adjacency[room_id] = [conn["room_id"] for conn in connections]
    
    return adjacency


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


# ============================================================================
# PRACTICAL TOOLS FOR AGENT USE
# ============================================================================

def find_all_evacuation_routes(project: Project) -> Dict[str, any]:
    """
    Find evacuation routes for all rooms in the building.
    
    Args:
        project: Project object
        
    Returns:
        Dictionary with evacuation analysis for all rooms
    """
    graph = create_circulation_graph(project)
    all_rooms = project.get_all_rooms()
    
    results = {
        "total_rooms": len(all_rooms),
        "room_routes": {},
        "longest_route": None,
        "shortest_route": None,
        "average_distance": 0,
        "compliance_summary": {}
    }
    
    distances = []
    
    for room in all_rooms:
        route_info = graph.calculate_egress_distance(room.id)
        results["room_routes"][room.id] = route_info
        
        if route_info["distance"] < float('inf'):
            distances.append(route_info["distance"])
    
    if distances:
        results["average_distance"] = sum(distances) / len(distances)
        
        # Find longest and shortest routes
        for room_id, route in results["room_routes"].items():
            if route["distance"] < float('inf'):
                if results["longest_route"] is None or route["distance"] > results["longest_route"]["distance"]:
                    results["longest_route"] = {"room_id": room_id, **route}
                if results["shortest_route"] is None or route["distance"] < results["shortest_route"]["distance"]:
                    results["shortest_route"] = {"room_id": room_id, **route}
    
    # Compliance analysis
    compliant_rooms = sum(1 for route in results["room_routes"].values() 
                         if route["distance"] <= 25)  # 25m limit
    results["compliance_summary"] = {
        "compliant_rooms": compliant_rooms,
        "total_rooms": len(all_rooms),
        "compliance_rate": compliant_rooms / len(all_rooms) if all_rooms else 0
    }
    
    return results


def calculate_room_connectivity_score(project: Project, room_id: str) -> Dict[str, any]:
    """
    Calculate how well-connected a room is to the rest of the building.
    
    Args:
        project: Project object
        room_id: ID of the room to analyze
        
    Returns:
        Dictionary with connectivity analysis
    """
    graph = create_circulation_graph(project)
    room_node = f"room_{room_id}"
    
    if room_node not in graph.graph:
        return {"error": f"Room {room_id} not found"}
    
    # Basic connectivity metrics
    direct_connections = len(list(graph.graph.neighbors(room_node)))
    total_rooms = graph.graph.number_of_nodes()
    
    # Calculate average distance to all other rooms
    distances_to_others = []
    for other_node in graph.graph.nodes():
        if other_node != room_node:
            try:
                distance = nx.shortest_path_length(graph.graph, room_node, other_node, weight='weight')
                distances_to_others.append(distance)
            except nx.NetworkXNoPath:
                distances_to_others.append(float('inf'))
    
    reachable_rooms = sum(1 for d in distances_to_others if d < float('inf'))
    avg_distance = sum(d for d in distances_to_others if d < float('inf')) / max(1, reachable_rooms)
    
    # Calculate centrality (how central this room is to circulation)
    try:
        betweenness = nx.betweenness_centrality(graph.graph, weight='weight')[room_node]
        closeness = nx.closeness_centrality(graph.graph, distance='weight')[room_node]
    except:
        betweenness = 0
        closeness = 0
    
    return {
        "room_id": room_id,
        "direct_connections": direct_connections,
        "reachable_rooms": reachable_rooms,
        "total_rooms": total_rooms - 1,  # Exclude self
        "reachability_score": reachable_rooms / max(1, total_rooms - 1),
        "average_distance": avg_distance,
        "betweenness_centrality": betweenness,
        "closeness_centrality": closeness,
        "connectivity_grade": "high" if direct_connections >= 3 else "medium" if direct_connections >= 2 else "low"
    }


def find_critical_circulation_points(project: Project) -> Dict[str, any]:
    """
    Identify rooms that are critical for building circulation.
    
    Args:
        project: Project object
        
    Returns:
        Dictionary with critical point analysis
    """
    graph = create_circulation_graph(project)
    
    # Find articulation points (rooms whose removal would disconnect the building)
    articulation_points = list(nx.articulation_points(graph.graph))
    
    # Find rooms with high betweenness centrality (many paths go through them)
    betweenness = nx.betweenness_centrality(graph.graph, weight='weight')
    high_betweenness = [(node, score) for node, score in betweenness.items() 
                       if score > 0.1]  # Threshold for "high" centrality
    
    # Analyze each critical point
    critical_rooms = []
    for room_node in articulation_points:
        room_id = room_node.replace("room_", "")
        room = project.get_room_by_id(room_id)
        if room:
            connections = graph.get_room_connections(room_id)
            critical_rooms.append({
                "room_id": room_id,
                "room_name": room.name,
                "criticality_type": "articulation_point",
                "connection_count": len(connections),
                "betweenness_score": betweenness.get(room_node, 0)
            })
    
    return {
        "critical_room_count": len(critical_rooms),
        "critical_rooms": critical_rooms,
        "high_betweenness_rooms": high_betweenness,
        "connectivity_risk": "high" if len(critical_rooms) > 2 else "medium" if len(critical_rooms) > 0 else "low",
        "recommendations": [
            "Consider adding alternative circulation paths",
            "Ensure critical rooms have adequate door widths",
            "Plan for emergency access to critical areas"
        ] if critical_rooms else ["Building has good circulation redundancy"]
    }


def calculate_door_usage_analysis(project: Project) -> Dict[str, any]:
    """
    Analyze how much each door is used in circulation paths.
    
    Args:
        project: Project object
        
    Returns:
        Dictionary with door usage analysis
    """
    graph = create_circulation_graph(project)
    all_rooms = project.get_all_rooms()
    all_doors = project.get_all_doors()
    
    door_usage = {door.id: 0 for door in all_doors}
    
    # Count how many evacuation routes use each door
    for room in all_rooms:
        route_info = graph.calculate_egress_distance(room.id)
        path = route_info.get("path", [])
        
        # Walk through the path and count door usage
        for i in range(len(path) - 1):
            room_node_1 = f"room_{path[i]}"
            room_node_2 = f"room_{path[i+1]}"
            
            if graph.graph.has_edge(room_node_1, room_node_2):
                edge_data = graph.graph[room_node_1][room_node_2]
                door_id = edge_data.get('door_id')
                if door_id and door_id in door_usage:
                    door_usage[door_id] += 1
    
    # Analyze the results
    total_usage = sum(door_usage.values())
    high_usage_doors = [(door_id, count) for door_id, count in door_usage.items() 
                       if count > total_usage * 0.2]  # Top 20% usage
    
    unused_doors = [door_id for door_id, count in door_usage.items() if count == 0]
    
    return {
        "door_usage_counts": door_usage,
        "total_usage": total_usage,
        "high_usage_doors": high_usage_doors,
        "unused_doors": unused_doors,
        "most_used_door": max(door_usage.items(), key=lambda x: x[1]) if door_usage else None,
        "usage_distribution": {
            "high_usage": len(high_usage_doors),
            "medium_usage": len([c for c in door_usage.values() if 0 < c <= total_usage * 0.2]),
            "unused": len(unused_doors)
        }
    }


def validate_evacuation_compliance(project: Project, max_distance: float = 25.0) -> Dict[str, any]:
    """
    Validate building evacuation compliance against regulations.
    
    Args:
        project: Project object
        max_distance: Maximum allowed evacuation distance in meters
        
    Returns:
        Dictionary with compliance validation results
    """
    graph = create_circulation_graph(project)
    all_rooms = project.get_all_rooms()
    
    compliant_rooms = []
    non_compliant_rooms = []
    inaccessible_rooms = []
    
    for room in all_rooms:
        route_info = graph.calculate_egress_distance(room.id)
        distance = route_info.get("distance", float('inf'))
        
        if distance == float('inf'):
            inaccessible_rooms.append({
                "room_id": room.id,
                "room_name": room.name,
                "issue": "No evacuation path available"
            })
        elif distance > max_distance:
            non_compliant_rooms.append({
                "room_id": room.id,
                "room_name": room.name,
                "distance": distance,
                "excess_distance": distance - max_distance
            })
        else:
            compliant_rooms.append({
                "room_id": room.id,
                "room_name": room.name,
                "distance": distance
            })
    
    total_rooms = len(all_rooms)
    compliance_rate = len(compliant_rooms) / total_rooms if total_rooms > 0 else 0
    
    return {
        "compliance_rate": compliance_rate,
        "compliant_rooms": compliant_rooms,
        "non_compliant_rooms": non_compliant_rooms,
        "inaccessible_rooms": inaccessible_rooms,
        "total_rooms": total_rooms,
        "max_distance_limit": max_distance,
        "overall_status": "COMPLIANT" if compliance_rate >= 1.0 else "PARTIAL" if compliance_rate >= 0.8 else "NON_COMPLIANT",
        "recommendations": [
            f"Add evacuation routes for {len(inaccessible_rooms)} inaccessible rooms",
            f"Reduce evacuation distances for {len(non_compliant_rooms)} rooms",
            "Consider adding additional exits or improving circulation paths"
        ] if (non_compliant_rooms or inaccessible_rooms) else ["Building meets evacuation requirements"]
    }
