#!/usr/bin/env python3
"""
Building Data Loader for Notebooks

This module provides functions to load and work with real building data
extracted from IFC files for use in Jupyter notebooks and tutorials.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildingDataLoader:
    """Loader for building data extracted from IFC files."""
    
    def __init__(self, data_path: Path):
        """Initialize with path to extracted building data JSON."""
        self.data_path = data_path
        self.data = None
        self.metadata = None
        self.levels = []
        self.all_rooms = []
        self.all_doors = []
        self.all_walls = []
        
    def load_data(self) -> Dict[str, Any]:
        """Load building data from JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Building data file not found: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get('metadata', {})
        self.levels = self.data.get('levels', [])
        
        # Flatten all elements for easy access
        self.all_rooms = []
        self.all_doors = []
        self.all_walls = []
        
        for level in self.levels:
            self.all_rooms.extend(level.get('rooms', []))
            self.all_doors.extend(level.get('doors', []))
            self.all_walls.extend(level.get('walls', []))
        
        logger.info(f"Loaded building data: {self.get_summary()}")
        return self.data
    
    def get_summary(self) -> str:
        """Get a summary of the loaded building data."""
        if not self.data:
            return "No data loaded"
        
        return (f"{self.metadata.get('project_name', 'Unknown Project')} - "
                f"{len(self.levels)} levels, {len(self.all_rooms)} rooms, "
                f"{len(self.all_doors)} doors, {len(self.all_walls)} walls")
    
    def get_level_data(self, level_name: str) -> Optional[Dict]:
        """Get data for a specific level."""
        for level in self.levels:
            if level['name'] == level_name:
                return level
        return None
    
    def get_rooms_by_level(self, level_name: str) -> List[Dict]:
        """Get all rooms for a specific level."""
        level_data = self.get_level_data(level_name)
        return level_data.get('rooms', []) if level_data else []
    
    def get_doors_by_level(self, level_name: str) -> List[Dict]:
        """Get all doors for a specific level."""
        level_data = self.get_level_data(level_name)
        return level_data.get('doors', []) if level_data else []
    
    def get_walls_by_level(self, level_name: str) -> List[Dict]:
        """Get all walls for a specific level."""
        level_data = self.get_level_data(level_name)
        return level_data.get('walls', []) if level_data else []
    
    def get_room_areas(self) -> Dict[str, float]:
        """Get areas of all rooms."""
        return {room['name']: room['area'] for room in self.all_rooms}
    
    def get_room_types(self) -> Dict[str, int]:
        """Get count of different room types."""
        types = {}
        for room in self.all_rooms:
            room_type = room.get('use', 'unknown')
            types[room_type] = types.get(room_type, 0) + 1
        return types
    
    def get_door_widths(self) -> List[float]:
        """Get list of all door widths in mm."""
        return [door['width_mm'] for door in self.all_doors]
    
    def get_wall_coordinates(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get start and end coordinates for all walls."""
        coords = []
        for wall in self.all_walls:
            start = (wall['start_point']['x'], wall['start_point']['y'])
            end = (wall['end_point']['x'], wall['end_point']['y'])
            coords.append((start, end))
        return coords
    
    def calculate_building_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate overall building bounds from walls."""
        if not self.all_walls:
            return 0, 0, 100, 100
        
        x_coords = []
        y_coords = []
        for wall in self.all_walls:
            x_coords.extend([wall['start_point']['x'], wall['end_point']['x']])
            y_coords.extend([wall['start_point']['y'], wall['end_point']['y']])
        
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    
    def visualize_level(self, level_name: str, figsize: Tuple[int, int] = (12, 8)):
        """Visualize a specific building level."""
        level_data = self.get_level_data(level_name)
        if not level_data:
            print(f"Level '{level_name}' not found")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot walls
        walls = level_data.get('walls', [])
        for wall in walls:
            start = wall['start_point']
            end = wall['end_point']
            ax.plot([start['x'], end['x']], [start['y'], end['y']], 
                   'k-', linewidth=2, alpha=0.7)
        
        # Plot doors
        doors = level_data.get('doors', [])
        for door in doors:
            pos = door['position']
            width = door['width_mm'] / 1000  # Convert to meters for visualization
            color = 'red' if door.get('is_emergency_exit', False) else 'blue'
            
            # Draw door as a small rectangle
            rect = patches.Rectangle((pos['x'] - width/2, pos['y'] - 0.1), 
                                   width, 0.2, linewidth=1, 
                                   edgecolor=color, facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add door label
            ax.text(pos['x'], pos['y'] + 0.3, door['id'], 
                   ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Plot room labels (at center of room area)
        rooms = level_data.get('rooms', [])
        for room in rooms:
            # For now, place room labels at building center
            # In a real implementation, you'd calculate room centroid
            bounds = self.calculate_building_bounds()
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            
            ax.text(center_x, center_y, f"{room['name']}\n{room['area']:.0f} m²",
                   ha='center', va='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        ax.set_title(f"Level: {level_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='black', lw=2, label='Walls'),
            plt.Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.7, label='Doors'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, label='Emergency Exits'),
            plt.Rectangle((0, 0), 1, 1, facecolor='yellow', alpha=0.7, label='Rooms')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_statistics_dashboard(self):
        """Create a dashboard with building statistics."""
        if not self.data:
            print("No data loaded. Call load_data() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Room areas by level
        level_names = [level['name'] for level in self.levels]
        level_areas = []
        for level in self.levels:
            total_area = sum(room['area'] for room in level.get('rooms', []))
            level_areas.append(total_area)
        
        ax1.bar(range(len(level_names)), level_areas, color='steelblue')
        ax1.set_title('Total Room Area by Level')
        ax1.set_xlabel('Level')
        ax1.set_ylabel('Area (m²)')
        ax1.set_xticks(range(len(level_names)))
        ax1.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                            for name in level_names], rotation=45)
        
        # 2. Room types distribution
        room_types = self.get_room_types()
        ax2.pie(room_types.values(), labels=room_types.keys(), autopct='%1.1f%%')
        ax2.set_title('Distribution of Room Types')
        
        # 3. Door widths histogram
        door_widths = self.get_door_widths()
        ax3.hist(door_widths, bins=10, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Door Widths')
        ax3.set_xlabel('Width (mm)')
        ax3.set_ylabel('Count')
        ax3.axvline(800, color='red', linestyle='--', label='Min Emergency Width')
        ax3.legend()
        
        # 4. Elements count by level
        level_data = []
        for level in self.levels:
            level_data.append({
                'Level': level['name'][:10] + '...' if len(level['name']) > 10 else level['name'],
                'Rooms': len(level.get('rooms', [])),
                'Doors': len(level.get('doors', [])),
                'Walls': len(level.get('walls', []))
            })
        
        df = pd.DataFrame(level_data)
        x = np.arange(len(df))
        width = 0.25
        
        ax4.bar(x - width, df['Rooms'], width, label='Rooms', color='green', alpha=0.7)
        ax4.bar(x, df['Doors'], width, label='Doors', color='blue', alpha=0.7)
        ax4.bar(x + width, df['Walls'], width, label='Walls', color='red', alpha=0.7)
        
        ax4.set_title('Building Elements by Level')
        ax4.set_xlabel('Level')
        ax4.set_ylabel('Count')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['Level'], rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def export_to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Export building data to pandas DataFrames for analysis."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Rooms DataFrame
        rooms_data = []
        for room in self.all_rooms:
            rooms_data.append({
                'id': room['id'],
                'name': room['name'],
                'level': room['level'],
                'area_m2': room['area'],
                'use_type': room['use'],
                'occupancy_load': room['occupancy_load']
            })
        rooms_df = pd.DataFrame(rooms_data)
        
        # Doors DataFrame
        doors_data = []
        for door in self.all_doors:
            doors_data.append({
                'id': door['id'],
                'name': door['name'],
                'width_mm': door['width_mm'],
                'height_mm': door['height_mm'],
                'door_type': door['door_type'],
                'is_emergency_exit': door['is_emergency_exit'],
                'x': door['position']['x'],
                'y': door['position']['y'],
                'z': door['position']['z']
            })
        doors_df = pd.DataFrame(doors_data)
        
        # Walls DataFrame
        walls_data = []
        for wall in self.all_walls:
            walls_data.append({
                'id': wall['id'],
                'thickness_mm': wall['thickness_mm'],
                'height_mm': wall['height_mm'],
                'material': wall['material'],
                'start_x': wall['start_point']['x'],
                'start_y': wall['start_point']['y'],
                'end_x': wall['end_point']['x'],
                'end_y': wall['end_point']['y']
            })
        walls_df = pd.DataFrame(walls_data)
        
        return {
            'rooms': rooms_df,
            'doors': doors_df,
            'walls': walls_df
        }
    
    def export_to_json(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export building data to LLM-optimized JSON format."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create LLM-optimized structure
        export_data = {
            "building_metadata": {
                "project_name": self.metadata.get('project_name'),
                "building_type": self.metadata.get('building_type'),
                "total_area_m2": self.metadata.get('total_area'),
                "created_date": self.metadata.get('created_date'),
                "summary": {
                    "levels": len(self.levels),
                    "rooms": len(self.all_rooms),
                    "doors": len(self.all_doors),
                    "walls": len(self.all_walls)
                }
            },
            "levels": [
                {
                    "name": level['name'],
                    "elevation_m": level['elevation'],
                    "element_counts": {
                        "rooms": len(level.get('rooms', [])),
                        "doors": len(level.get('doors', [])),
                        "walls": len(level.get('walls', []))
                    }
                }
                for level in self.levels
            ],
            "rooms": [
                {
                    "id": room['id'],
                    "name": room['name'],
                    "level": room['level'],
                    "area_m2": room['area'],
                    "use_type": room['use'],
                    "occupancy_load": room['occupancy_load']
                }
                for room in self.all_rooms
            ],
            "doors": [
                {
                    "id": door['id'],
                    "name": door.get('name', ''),
                    "dimensions": {
                        "width_mm": door['width_mm'],
                        "height_mm": door['height_mm']
                    },
                    "type": door['door_type'],
                    "properties": {
                        "is_emergency_exit": door.get('is_emergency_exit', False),
                        "is_accessible": door.get('is_accessible', True),
                        "fire_rating": door.get('fire_rating')
                    },
                    "position": door['position']
                }
                for door in self.all_doors
            ],
            "walls": [
                {
                    "id": wall['id'],
                    "geometry": {
                        "start_point": wall['start_point'],
                        "end_point": wall['end_point']
                    },
                    "properties": {
                        "thickness_mm": wall['thickness_mm'],
                        "height_mm": wall['height_mm'],
                        "material": wall['material'],
                        "fire_rating": wall.get('fire_rating')
                    }
                }
                for wall in self.all_walls
            ]
        }
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"LLM-optimized JSON exported to: {file_path}")
        
        return export_data
    
    def export_flat_json(self, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export building data as flat arrays - easiest for LLM parsing."""
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Flat structure - all elements in simple arrays
        flat_data = {
            "building_info": {
                "name": self.metadata.get('project_name'),
                "type": self.metadata.get('building_type'),
                "total_area": self.metadata.get('total_area'),
                "element_counts": {
                    "levels": len(self.levels),
                    "rooms": len(self.all_rooms),
                    "doors": len(self.all_doors),
                    "walls": len(self.all_walls)
                }
            },
            "all_elements": []
        }
        
        # Add all rooms as flat elements
        for room in self.all_rooms:
            flat_data["all_elements"].append({
                "element_type": "room",
                "id": room['id'],
                "name": room['name'],
                "level": room['level'],
                "area_m2": room['area'],
                "use_type": room['use'],
                "occupancy": room['occupancy_load']
            })
        
        # Add all doors as flat elements
        for door in self.all_doors:
            flat_data["all_elements"].append({
                "element_type": "door",
                "id": door['id'],
                "name": door.get('name', ''),
                "width_mm": door['width_mm'],
                "height_mm": door['height_mm'],
                "door_type": door['door_type'],
                "is_emergency": door.get('is_emergency_exit', False),
                "position_x": door['position']['x'],
                "position_y": door['position']['y'],
                "position_z": door['position']['z']
            })
        
        # Add all walls as flat elements
        for wall in self.all_walls:
            flat_data["all_elements"].append({
                "element_type": "wall",
                "id": wall['id'],
                "thickness_mm": wall['thickness_mm'],
                "height_mm": wall['height_mm'],
                "material": wall['material'],
                "start_x": wall['start_point']['x'],
                "start_y": wall['start_point']['y'],
                "start_z": wall['start_point']['z'],
                "end_x": wall['end_point']['x'],
                "end_y": wall['end_point']['y'],
                "end_z": wall['end_point']['z']
            })
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(flat_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Flat JSON exported to: {file_path}")
        
        return flat_data


# Convenience functions for easy use in notebooks
def load_vilamalla_building() -> BuildingDataLoader:
    """Load the Vilamalla building data."""
    # Try multiple possible paths to find the data file
    possible_paths = [
        Path("data/extracted/vilamalla_building.json"),
        Path("../data/extracted/vilamalla_building.json"),
        Path("../../data/extracted/vilamalla_building.json"),
        Path(__file__).parent.parent / "data" / "extracted" / "vilamalla_building.json"
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(
            f"Building data file not found. Tried paths: {[str(p) for p in possible_paths]}\n"
            f"Make sure to run the data extraction first or use sample data."
        )
    
    loader = BuildingDataLoader(data_path)
    loader.load_data()
    return loader


def quick_building_overview(loader: BuildingDataLoader):
    """Print a quick overview of the building data."""
    print("🏢 Building Overview")
    print("=" * 50)
    print(f"Project: {loader.metadata.get('project_name', 'Unknown')}")
    print(f"Total Area: {loader.metadata.get('total_area', 0):.1f} m²")
    print(f"Levels: {len(loader.levels)}")
    print(f"Rooms: {len(loader.all_rooms)}")
    print(f"Doors: {len(loader.all_doors)}")
    print(f"Walls: {len(loader.all_walls)}")
    print()
    
    # Level breakdown
    print("📊 Level Breakdown:")
    for level in loader.levels:
        level_name = level['name']
        rooms_count = len(level.get('rooms', []))
        doors_count = len(level.get('doors', []))
        walls_count = len(level.get('walls', []))
        total_area = sum(room['area'] for room in level.get('rooms', []))
        
        print(f"  {level_name[:25]:25} | "
              f"Rooms: {rooms_count:2} | "
              f"Doors: {doors_count:2} | "
              f"Walls: {walls_count:3} | "
              f"Area: {total_area:5.0f} m²")


def analyze_door_compliance(loader: BuildingDataLoader) -> Dict[str, Any]:
    """Analyze door compliance with building codes."""
    compliant_doors = []
    non_compliant_doors = []
    
    min_emergency_width = 800  # mm
    min_standard_width = 700   # mm
    
    for door in loader.all_doors:
        width = door['width_mm']
        is_emergency = door.get('is_emergency_exit', False)
        min_width = min_emergency_width if is_emergency else min_standard_width
        
        door_analysis = {
            'id': door['id'],
            'name': door['name'],
            'width_mm': width,
            'is_emergency': is_emergency,
            'required_width': min_width,
            'compliant': width >= min_width,
            'deficit_mm': max(0, min_width - width)
        }
        
        if door_analysis['compliant']:
            compliant_doors.append(door_analysis)
        else:
            non_compliant_doors.append(door_analysis)
    
    return {
        'total_doors': len(loader.all_doors),
        'compliant_count': len(compliant_doors),
        'non_compliant_count': len(non_compliant_doors),
        'compliance_rate': len(compliant_doors) / len(loader.all_doors) * 100,
        'compliant_doors': compliant_doors,
        'non_compliant_doors': non_compliant_doors
    }


if __name__ == "__main__":
    # Example usage
    try:
        loader = load_vilamalla_building()
        quick_building_overview(loader)
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        
        # Show statistics dashboard
        loader.create_statistics_dashboard()
        plt.show()
        
        # Show a level visualization
        if loader.levels:
            first_level = loader.levels[0]['name']
            loader.visualize_level(first_level)
            plt.show()
        
        # Analyze door compliance
        compliance = analyze_door_compliance(loader)
        print(f"\n🚪 Door Compliance Analysis:")
        print(f"Compliance Rate: {compliance['compliance_rate']:.1f}%")
        print(f"Compliant Doors: {compliance['compliant_count']}/{compliance['total_doors']}")
        
        if compliance['non_compliant_doors']:
            print("\nNon-compliant doors:")
            for door in compliance['non_compliant_doors']:
                print(f"  {door['id']}: {door['width_mm']}mm "
                      f"(needs {door['deficit_mm']}mm more)")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure to run the IFC extraction first:")
        print("python scripts/extract_ifc_files.py -f data/blueprints/VILAMALLA_ARQ_V6_TALLER_arq_20251032.ifc")
