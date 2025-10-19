"""
Visualization utilities for AEC Compliance Agent.

This module provides matplotlib-based visualization functions for floor plans,
fire equipment coverage, evacuation routes, and building analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MPLPolygon, Circle
from matplotlib.collections import PatchCollection
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
from pathlib import Path

from ..schemas import Project, Room, Door, FireEquipment
from ..calculations.geometry import get_room_polygon, get_room_centroid
from ..calculations.graph import CirculationGraph, PathResult


def plot_floorplan(project: Project, 
                   highlight_path: Optional[List[str]] = None,
                   figsize: Tuple[int, int] = (15, 10),
                   show_room_labels: bool = True,
                   show_doors: bool = True,
                   show_fire_equipment: bool = True) -> plt.Figure:
    """
    Plot building floor plan with matplotlib.
    
    Args:
        project: Project containing building data
        highlight_path: Optional list of room IDs to highlight
        figsize: Figure size tuple
        show_room_labels: Whether to show room names
        show_doors: Whether to show door locations
        show_fire_equipment: Whether to show fire equipment
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ValueError: If project data is invalid
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate overall bounds
        all_x_coords = []
        all_y_coords = []
        
        for room in project.rooms:
            for point in room.boundary:
                all_x_coords.append(point[0])
                all_y_coords.append(point[1])
        
        if not all_x_coords:
            raise ValueError("No room data found in project")
        
        # Plot rooms
        room_patches = []
        room_colors = []
        
        for room in project.rooms:
            try:
                polygon = get_room_polygon(room)
                coords = list(polygon.exterior.coords)
                
                # Create matplotlib polygon
                room_patch = MPLPolygon(coords, closed=True)
                room_patches.append(room_patch)
                
                # Color based on room type and highlight status
                if highlight_path and room.id in highlight_path:
                    if room.id == highlight_path[0]:
                        room_colors.append('red')  # Start room
                    elif room.id == highlight_path[-1]:
                        room_colors.append('green')  # End room
                    else:
                        room_colors.append('orange')  # Path room
                else:
                    room_colors.append(_get_room_color(room.use_type))
                
            except Exception as e:
                print(f"Warning: Could not plot room {room.id}: {str(e)}")
                continue
        
        # Add room patches to plot
        if room_patches:
            room_collection = PatchCollection(room_patches, facecolors=room_colors, 
                                            edgecolors='black', linewidths=1, alpha=0.7)
            ax.add_collection(room_collection)
        
        # Add room labels
        if show_room_labels:
            _add_room_labels(ax, project.rooms)
        
        # Add doors
        if show_doors:
            _add_doors(ax, project.doors, highlight_path)
        
        # Add fire equipment
        if show_fire_equipment:
            _add_fire_equipment(ax, project.fire_equipment)
        
        # Set axis properties
        margin = 2.0
        ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
        ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Title
        title = f"Floor Plan - {project.metadata.project_name}"
        if project.metadata.level_name:
            title += f" - {project.metadata.level_name}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        _add_floorplan_legend(ax, highlight_path is not None)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        raise ValueError(f"Failed to create floor plan visualization: {str(e)}")


def plot_fire_equipment_coverage(project: Project, 
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Show fire equipment coverage areas on floor plan.
    
    Args:
        project: Project containing building data
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Start with basic floor plan
        fig = plot_floorplan(project, show_fire_equipment=False, figsize=figsize)
        ax = fig.gca()
        
        # Add coverage circles for equipment with coverage radius
        coverage_patches = []
        coverage_colors = []
        
        for equipment in project.fire_equipment:
            if equipment.coverage_radius:
                center = (equipment.position[0], equipment.position[1])
                coverage_circle = Circle(center, equipment.coverage_radius, 
                                       fill=True, alpha=0.2)
                coverage_patches.append(coverage_circle)
                coverage_colors.append(_get_equipment_coverage_color(equipment.equipment_type))
        
        # Add coverage areas
        if coverage_patches:
            coverage_collection = PatchCollection(coverage_patches, 
                                                facecolors=coverage_colors,
                                                edgecolors='none',
                                                alpha=0.3)
            ax.add_collection(coverage_collection)
        
        # Add equipment markers
        _add_fire_equipment(ax, project.fire_equipment, show_coverage=True)
        
        # Update title
        ax.set_title(f"Fire Equipment Coverage - {project.metadata.project_name}", 
                    fontsize=14, fontweight='bold')
        
        # Add coverage legend
        _add_coverage_legend(ax)
        
        return fig
        
    except Exception as e:
        raise ValueError(f"Failed to create fire equipment coverage visualization: {str(e)}")


def plot_evacuation_routes(project: Project, 
                          routes: Dict[str, PathResult],
                          figsize: Tuple[int, int] = (15, 10),
                          max_routes_shown: int = 10) -> plt.Figure:
    """
    Show evacuation paths on floor plan.
    
    Args:
        project: Project containing building data
        routes: Dictionary of evacuation routes by room ID
        figsize: Figure size tuple
        max_routes_shown: Maximum number of routes to display
        
    Returns:
        Matplotlib figure object
    """
    try:
        # Start with basic floor plan
        fig = plot_floorplan(project, show_fire_equipment=False, figsize=figsize)
        ax = fig.gca()
        
        # Get room positions for path drawing
        room_positions = {}
        for room in project.rooms:
            try:
                centroid = get_room_centroid(room)
                room_positions[room.id] = centroid
            except Exception:
                # Use boundary center as fallback
                x_coords = [p[0] for p in room.boundary]
                y_coords = [p[1] for p in room.boundary]
                room_positions[room.id] = (sum(x_coords)/len(x_coords), 
                                         sum(y_coords)/len(y_coords))
        
        # Sort routes by distance and show only the longest/most interesting ones
        sorted_routes = sorted(routes.items(), 
                             key=lambda x: x[1].distance, 
                             reverse=True)[:max_routes_shown]
        
        # Draw evacuation paths
        colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_routes)))
        
        for i, (room_id, route) in enumerate(sorted_routes):
            if not route.path or len(route.path) < 2:
                continue
                
            # Get path coordinates
            path_coords = []
            for path_room_id in route.path:
                if path_room_id in room_positions:
                    path_coords.append(room_positions[path_room_id])
            
            if len(path_coords) >= 2:
                # Draw path
                x_coords = [coord[0] for coord in path_coords]
                y_coords = [coord[1] for coord in path_coords]
                
                ax.plot(x_coords, y_coords, 'o-', 
                       color=colors[i], linewidth=2, markersize=6,
                       alpha=0.8, label=f"Route from {room_id} ({route.distance:.1f}m)")
                
                # Mark start and end
                ax.plot(x_coords[0], y_coords[0], 'o', 
                       color='red', markersize=10, markeredgecolor='black')
                ax.plot(x_coords[-1], y_coords[-1], 's', 
                       color='green', markersize=10, markeredgecolor='black')
        
        # Add egress door indicators
        for door in project.doors:
            if door.is_egress:
                ax.plot(door.position[0], door.position[1], '^', 
                       color='green', markersize=12, markeredgecolor='black',
                       label='Exit' if 'Exit' not in [t.get_text() for t in ax.get_legend().get_texts()] else "")
        
        # Update title
        ax.set_title(f"Evacuation Routes - {project.metadata.project_name}", 
                    fontsize=14, fontweight='bold')
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        
        # Ensure we don't duplicate the Exit label
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        ax.legend(unique_handles, unique_labels, 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        raise ValueError(f"Failed to create evacuation routes visualization: {str(e)}")


def save_plot(fig: plt.Figure, 
              filename: str, 
              output_dir: str = "outputs/visualizations",
              dpi: int = 300,
              format: str = "png") -> str:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure to save
        filename: Base filename (without extension)
        output_dir: Output directory path
        dpi: Resolution for raster formats
        format: File format (png, pdf, svg, etc.)
        
    Returns:
        Full path to saved file
        
    Raises:
        OSError: If directory creation or file saving fails
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Construct full filename
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        full_path = output_path / filename
        
        # Save figure
        fig.savefig(full_path, dpi=dpi, format=format, bbox_inches='tight')
        
        return str(full_path)
        
    except Exception as e:
        raise OSError(f"Failed to save plot to {filename}: {str(e)}")


# Helper functions

def _get_room_color(use_type: Optional[str]) -> str:
    """Get color for room based on use type."""
    color_map = {
        'office': '#E8F5E8',      # Light green
        'corridor': '#E8F0FF',     # Light blue
        'stairs': '#FFF2CC',       # Light yellow
        'bathroom': '#FFE8E8',     # Light pink
        'storage': '#F0F0F0',      # Light gray
        'meeting': '#E8E8FF',      # Light purple
        'kitchen': '#FFE8CC',      # Light orange
        'lobby': '#F0F8FF',        # Alice blue
        'emergency': '#FFB6C1',    # Light pink
        None: '#F5F5F5'            # Very light gray
    }
    return color_map.get(use_type, '#F5F5F5')


def _get_equipment_coverage_color(equipment_type: str) -> str:
    """Get color for fire equipment coverage area."""
    color_map = {
        'extinguisher': 'red',
        'hydrant': 'blue',
        'sprinkler': 'cyan',
        'alarm': 'orange',
        'emergency_light': 'yellow',
        'exit_sign': 'green'
    }
    return color_map.get(equipment_type, 'gray')


def _add_room_labels(ax: plt.Axes, rooms: List[Room]) -> None:
    """Add room labels to the plot."""
    for room in rooms:
        try:
            centroid = get_room_centroid(room)
            
            # Use room name if available, otherwise use ID
            label = room.name if room.name and room.name != room.id else room.id
            
            # Truncate long labels
            if len(label) > 15:
                label = label[:12] + "..."
            
            ax.text(centroid[0], centroid[1], label, 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        except Exception:
            # Skip problematic rooms
            continue


def _add_doors(ax: plt.Axes, doors: List[Door], highlight_path: Optional[List[str]] = None) -> None:
    """Add door markers to the plot."""
    for door in doors:
        x, y = door.position[0], door.position[1]
        
        if door.is_egress:
            # Green triangle for egress doors
            ax.plot(x, y, '^', color='green', markersize=10, 
                   markeredgecolor='black', markeredgewidth=1)
        else:
            # Blue circle for regular doors
            ax.plot(x, y, 'o', color='blue', markersize=6, 
                   markeredgecolor='black', markeredgewidth=1)
        
        # Add door ID label
        ax.text(x + 0.5, y + 0.5, door.id, fontsize=6, 
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


def _add_fire_equipment(ax: plt.Axes, equipment: List[FireEquipment], 
                       show_coverage: bool = False) -> None:
    """Add fire equipment markers to the plot."""
    equipment_symbols = {
        'extinguisher': ('s', 'red'),      # Square, red
        'hydrant': ('D', 'blue'),          # Diamond, blue
        'sprinkler': ('*', 'cyan'),        # Star, cyan
        'alarm': ('o', 'orange'),          # Circle, orange
        'emergency_light': ('^', 'yellow'), # Triangle, yellow
        'exit_sign': ('>', 'green')         # Right triangle, green
    }
    
    for eq in equipment:
        x, y = eq.position[0], eq.position[1]
        symbol, color = equipment_symbols.get(eq.equipment_type, ('o', 'gray'))
        
        ax.plot(x, y, symbol, color=color, markersize=8, 
               markeredgecolor='black', markeredgewidth=1)
        
        # Add equipment ID label
        ax.text(x + 0.3, y + 0.3, eq.id, fontsize=6,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))


def _add_floorplan_legend(ax: plt.Axes, has_highlight_path: bool = False) -> None:
    """Add legend for floor plan elements."""
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=8, label='Door'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
                  markersize=8, label='Exit Door'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                  markersize=8, label='Fire Extinguisher'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', 
                  markersize=8, label='Hydrant')
    ]
    
    if has_highlight_path:
        legend_elements.extend([
            patches.Patch(color='red', alpha=0.7, label='Start Room'),
            patches.Patch(color='green', alpha=0.7, label='Exit Room'),
            patches.Patch(color='orange', alpha=0.7, label='Path Room')
        ])
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))


def _add_coverage_legend(ax: plt.Axes) -> None:
    """Add legend for equipment coverage visualization."""
    legend_elements = [
        patches.Patch(color='red', alpha=0.3, label='Extinguisher Coverage'),
        patches.Patch(color='blue', alpha=0.3, label='Hydrant Coverage'),
        patches.Patch(color='cyan', alpha=0.3, label='Sprinkler Coverage'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                  markersize=8, label='Fire Extinguisher'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', 
                  markersize=8, label='Hydrant'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan', 
                  markersize=8, label='Sprinkler')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))


def create_multi_level_visualization(project: Project, 
                                   figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
    """
    Create visualization showing multiple building levels.
    
    Args:
        project: Project with multiple levels
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure with subplots for each level
    """
    try:
        levels = project.levels if project.levels else [None]
        
        if len(levels) == 1:
            # Single level - use regular floor plan
            return plot_floorplan(project, figsize=figsize)
        
        # Multiple levels - create subplots
        cols = min(3, len(levels))
        rows = (len(levels) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, level in enumerate(levels):
            ax = axes[i] if len(levels) > 1 else axes
            
            # Create project subset for this level
            level_rooms = level.rooms if level else []
            level_doors = level.doors if level else []
            level_equipment = level.fire_equipment if level else []
            
            # Plot simplified floor plan for this level
            # (Implementation would need to be adapted for subplot context)
            ax.set_title(f"Level: {level.name if level else 'Unknown'}")
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(levels), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        raise ValueError(f"Failed to create multi-level visualization: {str(e)}")


def generate_compliance_report_plots(project: Project, 
                                   circulation_graph: CirculationGraph,
                                   output_dir: str = "outputs/visualizations") -> List[str]:
    """
    Generate all visualization plots for compliance reporting.
    
    Args:
        project: Project data
        circulation_graph: Built circulation graph
        output_dir: Directory to save plots
        
    Returns:
        List of file paths for generated plots
    """
    try:
        saved_files = []
        
        # 1. Floor plan
        fig1 = plot_floorplan(project)
        file1 = save_plot(fig1, "floorplan", output_dir)
        saved_files.append(file1)
        plt.close(fig1)
        
        # 2. Fire equipment coverage
        fig2 = plot_fire_equipment_coverage(project)
        file2 = save_plot(fig2, "fire_equipment_coverage", output_dir)
        saved_files.append(file2)
        plt.close(fig2)
        
        # 3. Circulation graph
        fig3 = circulation_graph.visualize_graph()
        file3 = save_plot(fig3, "circulation_graph", output_dir)
        saved_files.append(file3)
        plt.close(fig3)
        
        # 4. Evacuation routes
        try:
            routes = circulation_graph.all_evacuation_routes()
            fig4 = plot_evacuation_routes(project, routes)
            file4 = save_plot(fig4, "evacuation_routes", output_dir)
            saved_files.append(file4)
            plt.close(fig4)
        except Exception as e:
            print(f"Warning: Could not generate evacuation routes plot: {str(e)}")
        
        return saved_files
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate compliance report plots: {str(e)}")