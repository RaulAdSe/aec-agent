"""
IFC Data Manager - Interface for agent access to processed IFC data
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class IFCDataManager:
    """Manages access to processed IFC data for the compliance agent."""
    
    def __init__(self, processed_data_dir: str = "data/processed_ifc"):
        """Initialize with directory containing processed IFC JSON files."""
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_all_processed_files(self) -> List[str]:
        """Get list of all processed IFC files."""
        json_files = list(self.processed_data_dir.glob("*.json"))
        return [f.stem for f in json_files]
    
    def get_building_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get processed building data for a specific IFC file."""
        json_path = self.processed_data_dir / f"{filename}.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_all_spaces(self) -> List[Dict[str, Any]]:
        """Get all spaces from all processed buildings."""
        all_spaces = []
        for file in self.get_all_processed_files():
            data = self.get_building_data(file)
            if data and 'spaces' in data:
                # Add source file info to each space
                for space in data['spaces']:
                    space['source_file'] = file
                all_spaces.extend(data['spaces'])
        return all_spaces
    
    def get_all_doors(self) -> List[Dict[str, Any]]:
        """Get all doors from all processed buildings."""
        all_doors = []
        for file in self.get_all_processed_files():
            data = self.get_building_data(file)
            if data and 'doors' in data:
                # Add source file info to each door
                for door in data['doors']:
                    door['source_file'] = file
                all_doors.extend(data['doors'])
        return all_doors
    
    def get_all_walls(self) -> List[Dict[str, Any]]:
        """Get all walls from all processed buildings."""
        all_walls = []
        for file in self.get_all_processed_files():
            data = self.get_building_data(file)
            if data and 'walls' in data:
                # Add source file info to each wall
                for wall in data['walls']:
                    wall['source_file'] = file
                all_walls.extend(data['walls'])
        return all_walls
    
    def search_spaces_by_name(self, name_pattern: str) -> List[Dict[str, Any]]:
        """Search for spaces containing the given name pattern."""
        all_spaces = self.get_all_spaces()
        return [
            space for space in all_spaces
            if name_pattern.lower() in space.get('name', '').lower()
        ]
    
    def get_building_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all processed buildings."""
        all_files = self.get_all_processed_files()
        
        total_spaces = len(self.get_all_spaces())
        total_doors = len(self.get_all_doors())
        total_walls = len(self.get_all_walls())
        
        return {
            "processed_files": len(all_files),
            "file_names": all_files,
            "total_spaces": total_spaces,
            "total_doors": total_doors,
            "total_walls": total_walls
        }
    
    def get_compliance_data_for_agent(self) -> Dict[str, Any]:
        """Get structured data formatted for compliance agent queries."""
        return {
            "building_summary": self.get_building_summary(),
            "spaces": self.get_all_spaces(),
            "doors": self.get_all_doors(),
            "walls": self.get_all_walls()
        }


# Utility functions for direct use
def get_processed_ifc_data() -> Dict[str, Any]:
    """Quick access to all processed IFC data for agent integration."""
    manager = IFCDataManager()
    return manager.get_compliance_data_for_agent()


def search_building_elements(element_type: str, search_term: str = "") -> List[Dict[str, Any]]:
    """Search for specific building elements across all processed files."""
    manager = IFCDataManager()
    
    if element_type.lower() == "spaces":
        if search_term:
            return manager.search_spaces_by_name(search_term)
        return manager.get_all_spaces()
    elif element_type.lower() == "doors":
        return manager.get_all_doors()
    elif element_type.lower() == "walls":
        return manager.get_all_walls()
    else:
        return []