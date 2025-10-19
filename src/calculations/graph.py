"""
Graph-based circulation analysis for AEC Compliance Agent.

This module provides graph algorithms for analyzing building circulation,
evacuation routes, and connectivity using NetworkX.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass

from ..schemas import Project, Room, Door
from .geometry import get_room_centroid, distance_between_points, rooms_are_adjacent


@dataclass
class PathResult:
    """Result of a path calculation."""
    path: List[str]
    distance: float
    doors_used: List[str]


class CirculationGraph:
    """
    Graph-based analysis of building circulation and evacuation routes.
    
    This class creates a NetworkX graph where rooms are nodes and doors are edges,
    enabling analysis of circulation patterns and evacuation routes.
    """
    
    def __init__(self, project: Project):
        """
        Initialize circulation graph from project data.
        
        Args:
            project: Project object containing building data
        """
        self.project = project
        self.graph = nx.Graph()
        self._room_positions = {}
        self._door_connections = {}
        self.build_graph()
    
    def build_graph(self) -> None:
        """
        Create NetworkX graph from rooms and doors.
        
        Rooms become nodes, doors become edges connecting rooms.
        Edge weights represent travel distances.
        """
        try:
            # Clear existing graph
            self.graph.clear()
            self._room_positions.clear()
            self._door_connections.clear()
            
            # Add room nodes
            for room in self.project.rooms:
                self.graph.add_node(room.id, 
                                  name=room.name,
                                  use_type=room.use_type,
                                  area=room.area or 0,
                                  level=room.level)
                
                # Store room centroid for positioning
                try:
                    centroid = get_room_centroid(room)
                    self._room_positions[room.id] = centroid
                except Exception as e:
                    # Fallback: use average of boundary points
                    x_coords = [point[0] for point in room.boundary]
                    y_coords = [point[1] for point in room.boundary]
                    fallback_pos = (sum(x_coords) / len(x_coords), 
                                  sum(y_coords) / len(y_coords))
                    self._room_positions[room.id] = fallback_pos
            
            # Add door edges
            for door in self.project.doors:
                if door.connected_rooms and len(door.connected_rooms) >= 2:
                    # Door explicitly connects specific rooms
                    room1_id = door.connected_rooms[0]
                    room2_id = door.connected_rooms[1]
                    
                    if room1_id in self._room_positions and room2_id in self._room_positions:
                        distance = self._calculate_door_distance(door, room1_id, room2_id)
                        self._add_door_edge(room1_id, room2_id, door, distance)
                        
                else:
                    # Find rooms connected by door position
                    connected_rooms = self._find_rooms_connected_by_door(door)
                    
                    if len(connected_rooms) >= 2:
                        for i in range(len(connected_rooms)):
                            for j in range(i + 1, len(connected_rooms)):
                                room1_id = connected_rooms[i]
                                room2_id = connected_rooms[j]
                                distance = self._calculate_door_distance(door, room1_id, room2_id)
                                self._add_door_edge(room1_id, room2_id, door, distance)
            
            # Add adjacency-based connections for rooms without explicit door connections
            self._add_adjacency_connections()
            
        except Exception as e:
            raise RuntimeError(f"Failed to build circulation graph: {str(e)}")
    
    def _calculate_door_distance(self, door: Door, room1_id: str, room2_id: str) -> float:
        """Calculate travel distance through a door between room centroids."""
        room1_pos = self._room_positions[room1_id]
        room2_pos = self._room_positions[room2_id]
        door_pos = (door.position[0], door.position[1])
        
        # Distance = room1_centroid -> door -> room2_centroid
        dist1 = distance_between_points(room1_pos, door_pos)
        dist2 = distance_between_points(door_pos, room2_pos)
        
        return dist1 + dist2
    
    def _add_door_edge(self, room1_id: str, room2_id: str, door: Door, distance: float) -> None:
        """Add an edge representing a door connection."""
        edge_data = {
            'door_id': door.id,
            'distance': distance,
            'door_width': door.width,
            'door_type': door.door_type,
            'is_egress': door.is_egress,
            'fire_rating': door.fire_rating,
            'weight': distance  # NetworkX uses 'weight' for shortest path
        }
        
        self.graph.add_edge(room1_id, room2_id, **edge_data)
        self._door_connections[door.id] = (room1_id, room2_id)
    
    def _find_rooms_connected_by_door(self, door: Door, threshold: float = 2.0) -> List[str]:
        """Find rooms that are likely connected by a door based on proximity."""
        door_pos = (door.position[0], door.position[1])
        connected_rooms = []
        
        for room_id, room_pos in self._room_positions.items():
            distance = distance_between_points(door_pos, room_pos)
            if distance <= threshold:
                connected_rooms.append(room_id)
        
        return connected_rooms
    
    def _add_adjacency_connections(self, max_distance: float = 15.0) -> None:
        """Add connections between adjacent rooms that don't have explicit doors."""
        rooms_by_id = {room.id: room for room in self.project.rooms}
        
        for room1 in self.project.rooms:
            for room2 in self.project.rooms:
                if room1.id >= room2.id:  # Avoid duplicates
                    continue
                    
                # Skip if already connected
                if self.graph.has_edge(room1.id, room2.id):
                    continue
                
                try:
                    if rooms_are_adjacent(room1, room2, threshold=1.0):
                        # Calculate direct distance between centroids
                        pos1 = self._room_positions[room1.id]
                        pos2 = self._room_positions[room2.id]
                        distance = distance_between_points(pos1, pos2)
                        
                        if distance <= max_distance:
                            self.graph.add_edge(room1.id, room2.id,
                                              distance=distance,
                                              is_adjacency=True,
                                              weight=distance)
                except Exception:
                    # Skip problematic room pairs
                    continue
    
    def shortest_path(self, from_room: str, to_room: str) -> List[str]:
        """
        Find shortest path between two rooms.
        
        Args:
            from_room: Starting room ID
            to_room: Destination room ID
            
        Returns:
            List of room IDs representing the shortest path
            
        Raises:
            ValueError: If rooms don't exist or no path exists
        """
        try:
            if from_room not in self.graph:
                raise ValueError(f"Room {from_room} not found in graph")
            if to_room not in self.graph:
                raise ValueError(f"Room {to_room} not found in graph")
            
            path = nx.shortest_path(self.graph, from_room, to_room, weight='weight')
            return path
            
        except nx.NetworkXNoPath:
            raise ValueError(f"No path exists between {from_room} and {to_room}")
        except Exception as e:
            raise ValueError(f"Failed to find path: {str(e)}")
    
    def shortest_egress_path(self, room_id: str) -> Tuple[List[str], float]:
        """
        Find shortest path from room to any exit.
        
        Args:
            room_id: Starting room ID
            
        Returns:
            Tuple of (path as list of room IDs, total distance)
            
        Raises:
            ValueError: If room doesn't exist or no egress path exists
        """
        try:
            if room_id not in self.graph:
                raise ValueError(f"Room {room_id} not found in graph")
            
            # Find all rooms connected to egress doors
            egress_rooms = set()
            for door in self.project.doors:
                if door.is_egress and door.id in self._door_connections:
                    room1, room2 = self._door_connections[door.id]
                    egress_rooms.add(room1)
                    egress_rooms.add(room2)
            
            if not egress_rooms:
                raise ValueError("No egress doors found in project")
            
            # Find shortest path to any egress room
            best_path = None
            best_distance = float('inf')
            
            for egress_room in egress_rooms:
                try:
                    path = nx.shortest_path(self.graph, room_id, egress_room, weight='weight')
                    distance = nx.shortest_path_length(self.graph, room_id, egress_room, weight='weight')
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
                        
                except nx.NetworkXNoPath:
                    continue
            
            if best_path is None:
                raise ValueError(f"No egress path found from room {room_id}")
            
            return best_path, best_distance
            
        except Exception as e:
            raise ValueError(f"Failed to find egress path: {str(e)}")
    
    def all_evacuation_routes(self) -> Dict[str, PathResult]:
        """
        Calculate evacuation routes for all rooms.
        
        Returns:
            Dictionary mapping room IDs to their evacuation route information
        """
        evacuation_routes = {}
        
        for room_id in self.graph.nodes():
            try:
                path, distance = self.shortest_egress_path(room_id)
                
                # Find doors used in path
                doors_used = []
                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                    if edge_data and 'door_id' in edge_data:
                        doors_used.append(edge_data['door_id'])
                
                evacuation_routes[room_id] = PathResult(
                    path=path,
                    distance=distance,
                    doors_used=doors_used
                )
                
            except Exception as e:
                # Store failed calculation
                evacuation_routes[room_id] = PathResult(
                    path=[],
                    distance=float('inf'),
                    doors_used=[]
                )
        
        return evacuation_routes
    
    def get_connected_components(self) -> List[Set[str]]:
        """
        Get connected components of the graph.
        
        Returns:
            List of sets, each containing room IDs in a connected component
        """
        return list(nx.connected_components(self.graph))
    
    def calculate_betweenness_centrality(self) -> Dict[str, float]:
        """
        Calculate betweenness centrality for all rooms.
        
        This identifies rooms that are important for circulation flow.
        
        Returns:
            Dictionary mapping room IDs to centrality values
        """
        return nx.betweenness_centrality(self.graph, weight='weight')
    
    def find_articulation_points(self) -> Set[str]:
        """
        Find articulation points (critical rooms for connectivity).
        
        Returns:
            Set of room IDs that are articulation points
        """
        return set(nx.articulation_points(self.graph))
    
    def visualize_graph(self, figsize: Tuple[int, int] = (12, 8), 
                       highlight_path: Optional[List[str]] = None) -> plt.Figure:
        """
        Create graph visualization using matplotlib.
        
        Args:
            figsize: Figure size tuple
            highlight_path: Optional path to highlight in red
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Use room positions if available, otherwise use spring layout
            if self._room_positions:
                pos = self._room_positions
            else:
                pos = nx.spring_layout(self.graph, k=3, iterations=50)
            
            # Draw all edges
            edge_colors = []
            edge_widths = []
            
            for edge in self.graph.edges():
                edge_data = self.graph.get_edge_data(edge[0], edge[1])
                
                if highlight_path and self._edge_in_path(edge, highlight_path):
                    edge_colors.append('red')
                    edge_widths.append(3)
                elif edge_data and edge_data.get('is_egress', False):
                    edge_colors.append('green')
                    edge_widths.append(2)
                else:
                    edge_colors.append('gray')
                    edge_widths.append(1)
            
            nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, 
                                 width=edge_widths, alpha=0.7, ax=ax)
            
            # Draw nodes
            node_colors = []
            node_sizes = []
            
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                use_type = node_data.get('use_type', 'unknown')
                
                if highlight_path and node in highlight_path:
                    if node == highlight_path[0]:
                        node_colors.append('red')  # Start
                    elif node == highlight_path[-1]:
                        node_colors.append('green')  # End
                    else:
                        node_colors.append('orange')  # Path
                    node_sizes.append(800)
                elif use_type == 'corridor':
                    node_colors.append('lightblue')
                    node_sizes.append(600)
                elif use_type == 'stairs':
                    node_colors.append('yellow')
                    node_sizes.append(700)
                elif use_type == 'office':
                    node_colors.append('lightgreen')
                    node_sizes.append(500)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(400)
            
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                                 node_size=node_sizes, alpha=0.8, ax=ax)
            
            # Add labels
            labels = {}
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                name = node_data.get('name', node)
                labels[node] = name[:10]  # Truncate long names
            
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, ax=ax)
            
            # Add egress door indicators
            self._add_egress_indicators(ax, pos)
            
            ax.set_title('Building Circulation Graph', fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                          markersize=10, label='Office'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                          markersize=10, label='Corridor'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                          markersize=10, label='Stairs'),
                plt.Line2D([0], [0], color='green', linewidth=2, label='Egress Door'),
                plt.Line2D([0], [0], color='gray', linewidth=1, label='Internal Door')
            ]
            
            if highlight_path:
                legend_elements.extend([
                    plt.Line2D([0], [0], color='red', linewidth=3, label='Highlighted Path'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                              markersize=10, label='Start'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                              markersize=10, label='Exit')
                ])
            
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            raise RuntimeError(f"Failed to create graph visualization: {str(e)}")
    
    def _edge_in_path(self, edge: Tuple[str, str], path: List[str]) -> bool:
        """Check if an edge is part of the given path."""
        for i in range(len(path) - 1):
            if (edge[0] == path[i] and edge[1] == path[i + 1]) or \
               (edge[1] == path[i] and edge[0] == path[i + 1]):
                return True
        return False
    
    def _add_egress_indicators(self, ax: plt.Axes, pos: Dict[str, Tuple[float, float]]) -> None:
        """Add visual indicators for egress doors."""
        for door in self.project.doors:
            if door.is_egress and door.id in self._door_connections:
                room1, room2 = self._door_connections[door.id]
                if room1 in pos and room2 in pos:
                    # Draw egress indicator between connected rooms
                    x1, y1 = pos[room1]
                    x2, y2 = pos[room2]
                    
                    # Draw a thick green line for egress doors
                    ax.plot([x1, x2], [y1, y2], 'g-', linewidth=4, alpha=0.8)
    
    def get_graph_statistics(self) -> Dict[str, any]:
        """
        Get basic statistics about the circulation graph.
        
        Returns:
            Dictionary with graph metrics
        """
        try:
            stats = {
                'num_rooms': self.graph.number_of_nodes(),
                'num_connections': self.graph.number_of_edges(),
                'is_connected': nx.is_connected(self.graph),
                'num_components': nx.number_connected_components(self.graph),
                'average_clustering': nx.average_clustering(self.graph),
                'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None,
                'average_path_length': nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else None
            }
            
            # Add egress connectivity
            egress_rooms = set()
            for door in self.project.doors:
                if door.is_egress and door.id in self._door_connections:
                    room1, room2 = self._door_connections[door.id]
                    egress_rooms.add(room1)
                    egress_rooms.add(room2)
            
            stats['num_egress_connections'] = len(egress_rooms)
            stats['egress_connectivity_ratio'] = len(egress_rooms) / max(1, self.graph.number_of_nodes())
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}