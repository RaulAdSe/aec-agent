"""
IFC extraction module for AEC Compliance Agent.

This module extracts building elements (rooms, doors, walls, windows) from IFC files
and converts them to structured JSON format using our Pydantic schemas.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

try:
    import ifcopenshell
    import ifcopenshell.util.element
    import ifcopenshell.util.placement
    import ifcopenshell.util.shape
    IFCOPENSHELL_AVAILABLE = True
except ImportError:
    IFCOPENSHELL_AVAILABLE = False

from src.schemas import (
    Project, ProjectMetadata, Level, Room, Door, Wall, 
    Point2D, Point3D, Boundary, BuildingUse, DoorType, FireRating
)


logger = logging.getLogger(__name__)


class IFCExtractor:
    """
    Extractor for IFC files using ifcopenshell library.
    
    This class handles the extraction of building elements from IFC files
    and converts them to our standardized data format.
    """
    
    def __init__(self):
        """Initialize the IFC extractor."""
        if not IFCOPENSHELL_AVAILABLE:
            raise ImportError(
                "ifcopenshell is required for IFC extraction. "
                "Install it with: pip install ifcopenshell"
            )
        
        self.ifc_file = None
        self.rooms = []
        self.doors = []
        self.walls = []
        self.windows = []
        self.levels = {}  # Store levels by name
        
    def load_file(self, file_path: Path) -> bool:
        """
        Load an IFC file.
        
        Args:
            file_path: Path to the IFC file
            
        Returns:
            True if file loaded successfully, False otherwise
        """
        try:
            self.ifc_file = ifcopenshell.open(str(file_path))
            logger.info(f"Successfully loaded IFC file: {file_path}")
            logger.info(f"IFC Schema: {self.ifc_file.schema}")
            return True
        except Exception as e:
            logger.error(f"Error loading IFC file {file_path}: {e}")
            return False
    
    def extract_all(self) -> Project:
        """
        Extract all building elements from the loaded IFC file.
        
        Returns:
            Project object with extracted data
        """
        if not self.ifc_file:
            raise ValueError("No IFC file loaded. Call load_file() first.")
        
        logger.info("Starting extraction of building elements from IFC...")
        
        # Extract building levels first
        self._extract_levels()
        
        # Extract different element types
        self._extract_spaces()  # IFC spaces are like rooms
        self._extract_doors()
        self._extract_walls()
        self._extract_windows()
        
        # Create project metadata
        metadata = self._create_project_metadata()
        
        # Create levels with extracted elements
        level_objects = []
        for level_name, level_data in self.levels.items():
            level_obj = Level(
                name=level_name,
                elevation=level_data.get('elevation', 0.0),
                rooms=level_data.get('rooms', []),
                doors=level_data.get('doors', []),
                walls=level_data.get('walls', [])
            )
            level_objects.append(level_obj)
        
        # If no levels found, create a default level
        if not level_objects:
            logger.warning("No levels found, creating default level")
            default_level = Level(
                name="Default Level",
                elevation=0.0,
                rooms=self.rooms,
                doors=self.doors,
                walls=self.walls
            )
            level_objects.append(default_level)
        
        # Create project
        project = Project(
            metadata=metadata,
            levels=level_objects
        )
        
        total_rooms = sum(len(level.rooms) for level in level_objects)
        total_doors = sum(len(level.doors) for level in level_objects)
        total_walls = sum(len(level.walls) for level in level_objects)
        
        logger.info(f"IFC extraction complete: {total_rooms} spaces, {total_doors} doors, {total_walls} walls")
        return project
    
    def _extract_levels(self):
        """Extract building levels/stories from the IFC file."""
        logger.info("Extracting building levels...")
        
        # Look for IfcBuildingStorey entities
        stories = self.ifc_file.by_type('IfcBuildingStorey')
        
        for story in stories:
            level_name = story.Name or story.LongName or f"Level {story.id()}"
            elevation = self._get_story_elevation(story)
            
            self.levels[level_name] = {
                'ifc_entity': story,
                'elevation': elevation,
                'rooms': [],
                'doors': [],
                'walls': []
            }
            
        logger.info(f"Found {len(self.levels)} building levels")
    
    def _extract_spaces(self):
        """Extract spaces/rooms from the IFC file."""
        logger.info("Extracting spaces/rooms...")
        
        # Look for IfcSpace entities
        spaces = self.ifc_file.by_type('IfcSpace')
        
        for space in spaces:
            room = self._create_room_from_space(space)
            if room:
                # Find which level this room belongs to
                level_name = self._find_space_level(space)
                if level_name and level_name in self.levels:
                    self.levels[level_name]['rooms'].append(room)
                else:
                    self.rooms.append(room)
        
        # If no spaces found, try to extract from rooms (if available in schema)
        if not spaces:
            try:
                rooms = self.ifc_file.by_type('IfcRoom')
                for room_entity in rooms:
                    room = self._create_room_from_space(room_entity)
                    if room:
                        level_name = self._find_space_level(room_entity)
                        if level_name and level_name in self.levels:
                            self.levels[level_name]['rooms'].append(room)
                        else:
                            self.rooms.append(room)
            except RuntimeError as e:
                # IfcRoom not available in this schema (e.g., IFC2X3)
                logger.info(f"IfcRoom not available in schema: {e}")
                
        # If still no rooms/spaces found, try to derive rooms from building elements
        total_rooms = len(self.rooms) + sum(len(level['rooms']) for level in self.levels.values())
        if total_rooms == 0:
            logger.warning("No spaces or rooms found, attempting to derive rooms from building geometry")
            self._derive_rooms_from_geometry()
        
        total_rooms = len(self.rooms) + sum(len(level['rooms']) for level in self.levels.values())
        logger.info(f"Extracted {total_rooms} spaces/rooms")
    
    def _extract_doors(self):
        """Extract doors from the IFC file."""
        logger.info("Extracting doors...")
        
        doors = self.ifc_file.by_type('IfcDoor')
        
        for door_entity in doors:
            door = self._create_door_from_entity(door_entity)
            if door:
                # Find which level this door belongs to
                level_name = self._find_element_level(door_entity)
                if level_name and level_name in self.levels:
                    self.levels[level_name]['doors'].append(door)
                else:
                    self.doors.append(door)
        
        total_doors = len(self.doors) + sum(len(level['doors']) for level in self.levels.values())
        logger.info(f"Extracted {total_doors} doors")
    
    def _extract_walls(self):
        """Extract walls from the IFC file."""
        logger.info("Extracting walls...")
        
        walls = self.ifc_file.by_type('IfcWall')
        
        for wall_entity in walls:
            wall = self._create_wall_from_entity(wall_entity)
            if wall:
                # Find which level this wall belongs to
                level_name = self._find_element_level(wall_entity)
                if level_name and level_name in self.levels:
                    self.levels[level_name]['walls'].append(wall)
                else:
                    self.walls.append(wall)
        
        total_walls = len(self.walls) + sum(len(level['walls']) for level in self.levels.values())
        logger.info(f"Extracted {total_walls} walls")
    
    def _extract_windows(self):
        """Extract windows from the IFC file (for future use)."""
        logger.info("Extracting windows...")
        
        windows = self.ifc_file.by_type('IfcWindow')
        logger.info(f"Found {len(windows)} windows (not processed yet)")
    
    def _create_room_from_space(self, space) -> Optional[Room]:
        """Create a Room object from an IFC space entity."""
        try:
            # Get space properties
            space_id = f"S{space.id()}"
            space_name = space.Name or space.LongName or f"Space {space.id()}"
            
            # Get area from quantity sets
            area = self._get_space_area(space)
            
            # Determine space use
            use = self._determine_space_use(space_name, space)
            
            # Get space boundary if available
            boundary = self._get_space_boundary(space)
            
            # Calculate occupancy load
            occupancy_load = self._calculate_occupancy_load(area, use)
            
            # Get fire rating if available
            fire_rating = self._get_element_fire_rating(space)
            
            return Room(
                id=space_id,
                name=space_name,
                area=area,
                use=use,
                boundary=boundary,
                level=self._find_space_level(space) or "Unknown Level",
                occupancy_load=occupancy_load,
                fire_rating=fire_rating
            )
            
        except Exception as e:
            logger.error(f"Error creating room from space {space.id()}: {e}")
            return None
    
    def _create_door_from_entity(self, door) -> Optional[Door]:
        """Create a Door object from an IFC door entity."""
        try:
            # Get door properties
            door_id = f"D{door.id()}"
            door_name = door.Name or door.LongName or f"Door {door.id()}"
            
            # Get door dimensions
            width_mm, height_mm = self._get_door_dimensions(door)
            
            # Determine door type
            door_type = self._determine_door_type(door)
            
            # Get door position
            position = self._get_element_position(door)
            
            # Find connected spaces
            from_room, to_room = self._find_connected_spaces(door)
            
            # Check if emergency exit
            is_emergency_exit = self._is_emergency_exit(door)
            
            # Get fire rating
            fire_rating = self._get_element_fire_rating(door)
            
            return Door(
                id=door_id,
                name=door_name,
                width_mm=width_mm,
                height_mm=height_mm,
                door_type=door_type,
                fire_rating=fire_rating,
                position=position,
                from_room=from_room,
                to_room=to_room,
                is_emergency_exit=is_emergency_exit,
                is_accessible=True  # Default to True, could be determined from IFC properties
            )
            
        except Exception as e:
            logger.error(f"Error creating door from entity {door.id()}: {e}")
            return None
    
    def _create_wall_from_entity(self, wall) -> Optional[Wall]:
        """Create a Wall object from an IFC wall entity."""
        try:
            # Get wall properties
            wall_id = f"W{wall.id()}"
            
            # Get wall geometry
            start_point, end_point = self._get_wall_geometry(wall)
            if not start_point or not end_point:
                return None
            
            # Get wall dimensions
            thickness_mm = self._get_wall_thickness(wall)
            height_mm = self._get_wall_height(wall)
            
            # Get material
            material = self._get_wall_material(wall)
            
            # Get fire rating
            fire_rating = self._get_element_fire_rating(wall)
            
            return Wall(
                id=wall_id,
                start_point=start_point,
                end_point=end_point,
                thickness_mm=thickness_mm,
                height_mm=height_mm,
                fire_rating=fire_rating,
                material=material
            )
            
        except Exception as e:
            logger.error(f"Error creating wall from entity {wall.id()}: {e}")
            return None
    
    def _get_story_elevation(self, story) -> float:
        """Get the elevation of a building story."""
        try:
            # Try to get elevation from placement
            if hasattr(story, 'ObjectPlacement') and story.ObjectPlacement:
                placement = ifcopenshell.util.placement.get_local_placement(story.ObjectPlacement)
                if placement is not None and len(placement) > 2:
                    return float(placement[2][3])  # Z coordinate
            
            # Try to get from Elevation attribute
            if hasattr(story, 'Elevation') and story.Elevation is not None:
                return float(story.Elevation)
            
            return 0.0
        except Exception:
            return 0.0
    
    def _find_space_level(self, space) -> Optional[str]:
        """Find which level a space belongs to."""
        try:
            # Look for spatial containment relationship
            for rel in getattr(space, 'Decomposes', []):
                if rel.RelatingObject and rel.RelatingObject.is_a('IfcBuildingStorey'):
                    return rel.RelatingObject.Name or f"Level {rel.RelatingObject.id()}"
            
            # Alternative: check ContainedInStructure
            if hasattr(space, 'ContainedInStructure'):
                for rel in space.ContainedInStructure:
                    if rel.RelatingStructure and rel.RelatingStructure.is_a('IfcBuildingStorey'):
                        return rel.RelatingStructure.Name or f"Level {rel.RelatingStructure.id()}"
            
            return None
        except Exception:
            return None
    
    def _find_element_level(self, element) -> Optional[str]:
        """Find which level an element belongs to."""
        try:
            # Check spatial containment
            if hasattr(element, 'ContainedInStructure'):
                for rel in element.ContainedInStructure:
                    if rel.RelatingStructure and rel.RelatingStructure.is_a('IfcBuildingStorey'):
                        return rel.RelatingStructure.Name or f"Level {rel.RelatingStructure.id()}"
            
            return None
        except Exception:
            return None
    
    def _get_space_area(self, space) -> float:
        """Get area of a space from IFC properties."""
        try:
            # Try to get area from quantity sets
            psets = ifcopenshell.util.element.get_psets(space)
            
            for pset_name, pset_data in psets.items():
                if 'area' in pset_name.lower() or 'qto' in pset_name.lower():
                    for prop_name, prop_value in pset_data.items():
                        if 'area' in prop_name.lower() and isinstance(prop_value, (int, float)):
                            return float(prop_value)
            
            # Try to calculate from geometry if available
            # This is a simplified approach
            return self._calculate_space_area_from_geometry(space)
            
        except Exception:
            return 50.0  # Default area
    
    def _calculate_space_area_from_geometry(self, space) -> float:
        """Calculate space area from geometry (simplified)."""
        try:
            # This is a placeholder - full implementation would require
            # more complex geometry processing
            return 50.0
        except Exception:
            return 50.0
    
    def _get_space_boundary(self, space) -> Optional[Boundary]:
        """Get space boundary from IFC geometry."""
        # This is a simplified implementation
        # Full implementation would extract actual boundary geometry
        return None
    
    def _determine_space_use(self, name: str, space) -> BuildingUse:
        """Determine space use from name and IFC properties with enhanced classification."""
        name_lower = name.lower()
        
        # Enhanced keyword-based classification for industrial/commercial buildings
        
        # Industrial/Manufacturing spaces
        if any(keyword in name_lower for keyword in ['taller', 'workshop', 'manufactur', 'production', 'assembly', 'factory']):
            return BuildingUse.COMMERCIAL  # Using commercial for industrial
        
        # Loading/Shipping areas
        if any(keyword in name_lower for keyword in ['muelle', 'loading', 'dock', 'ship', 'cargo', 'warehouse']):
            return BuildingUse.STORAGE
        
        # Technical/Mechanical areas
        if any(keyword in name_lower for keyword in ['mechanical', 'electrical', 'hvac', 'utility', 'technical', 'machine']):
            return BuildingUse.COMMERCIAL
        
        # Roof/Structural areas
        if any(keyword in name_lower for keyword in ['roof', 'cubierta', 'peto', 'panel', 'structural']):
            return BuildingUse.COMMERCIAL
        
        # Office areas
        if any(keyword in name_lower for keyword in ['office', 'oficina', 'admin', 'control']):
            return BuildingUse.OFFICE
        
        # Meeting/Conference areas
        if any(keyword in name_lower for keyword in ['meeting', 'reunion', 'sala', 'conference']):
            return BuildingUse.MEETING
        
        # Circulation areas
        if any(keyword in name_lower for keyword in ['corridor', 'hallway', 'pasillo', 'circulation']):
            return BuildingUse.CORRIDOR
        
        # Stairs and vertical circulation
        if any(keyword in name_lower for keyword in ['stair', 'escalera', 'elevator', 'ascensor']):
            return BuildingUse.STAIR
        
        # Restrooms
        if any(keyword in name_lower for keyword in ['restroom', 'bathroom', 'aseo', 'wc', 'toilet']):
            return BuildingUse.RESTROOM
        
        # Storage areas
        if any(keyword in name_lower for keyword in ['storage', 'almacen', 'deposit', 'archive']):
            return BuildingUse.STORAGE
        
        # Try to get use from IFC properties
        try:
            psets = ifcopenshell.util.element.get_psets(space)
            for pset_data in psets.values():
                for prop_name, prop_value in pset_data.items():
                    if 'use' in prop_name.lower() or 'function' in prop_name.lower():
                        if isinstance(prop_value, str):
                            use_lower = prop_value.lower()
                            if 'office' in use_lower:
                                return BuildingUse.OFFICE
                            elif 'meeting' in use_lower:
                                return BuildingUse.MEETING
                            elif 'storage' in use_lower:
                                return BuildingUse.STORAGE
                            elif 'circulation' in use_lower:
                                return BuildingUse.CORRIDOR
        except Exception:
            pass
        
        # Default fallback
        return BuildingUse.COMMERCIAL  # Using commercial for industrial
        
        # Loading/Shipping areas
        if any(keyword in name_lower for keyword in ['muelle', 'loading', 'dock', 'ship', 'cargo', 'warehouse']):
            return BuildingUse.STORAGE
        
        # Technical/Mechanical areas
        if any(keyword in name_lower for keyword in ['mechanical', 'electrical', 'hvac', 'utility', 'technical', 'machine']):
            return BuildingUse.COMMERCIAL
        
        # Roof/Structural areas
        if any(keyword in name_lower for keyword in ['roof', 'cubierta', 'peto', 'panel', 'structural']):
            return BuildingUse.COMMERCIAL
        
        # Office areas
        if any(keyword in name_lower for keyword in ['office', 'oficina', 'admin', 'control']):
            return BuildingUse.OFFICE
        
        # Meeting/Conference areas
        if any(keyword in name_lower for keyword in ['meeting', 'reunion', 'sala', 'conference']):
            return BuildingUse.MEETING
        
        # Circulation areas
        if any(keyword in name_lower for keyword in ['corridor', 'hallway', 'pasillo', 'circulation']):
            return BuildingUse.CORRIDOR
        
        # Stairs and vertical circulation
        if any(keyword in name_lower for keyword in ['stair', 'escalera', 'elevator', 'ascensor']):
            return BuildingUse.STAIR
        
        # Restrooms
        if any(keyword in name_lower for keyword in ['restroom', 'bathroom', 'aseo', 'wc', 'toilet']):
            return BuildingUse.RESTROOM
        
        # Storage areas
        if any(keyword in name_lower for keyword in ['storage', 'almacen', 'deposit', 'archive']):
            return BuildingUse.STORAGE
        
        # Try to get use from IFC properties
        try:
            psets = ifcopenshell.util.element.get_psets(space)
            for pset_data in psets.values():
                for prop_name, prop_value in pset_data.items():
                    if 'use' in prop_name.lower() or 'function' in prop_name.lower():
                        if isinstance(prop_value, str):
                            use_lower = prop_value.lower()
                            if 'office' in use_lower:
                                return BuildingUse.OFFICE
                            elif 'meeting' in use_lower:
                                return BuildingUse.MEETING
                            elif 'storage' in use_lower:
                                return BuildingUse.STORAGE
                            elif 'circulation' in use_lower:
                                return BuildingUse.CORRIDOR
        except Exception:
            pass
        
        # Default fallback
        return BuildingUse.COMMERCIAL
    
    def _calculate_occupancy_load(self, area: float, use: BuildingUse) -> int:
        """Calculate occupancy load based on area and use."""
        # Simplified occupancy factors
        factors = {
            BuildingUse.OFFICE: 0.1,      # 1 person per 10 sqm
            BuildingUse.COMMERCIAL: 0.1,  # 1 person per 10 sqm
            BuildingUse.MEETING: 0.2,     # 1 person per 5 sqm
            BuildingUse.RESTROOM: 0.1,    # 1 person per 10 sqm
            BuildingUse.STORAGE: 0.02,    # 1 person per 50 sqm
            BuildingUse.CORRIDOR: 0.05,   # 1 person per 20 sqm
            BuildingUse.STAIR: 0.1,       # 1 person per 10 sqm
        }
        
        factor = factors.get(use, 0.1)
        return max(1, int(area * factor))
    
    def _get_door_dimensions(self, door) -> Tuple[float, float]:
        """Get door width and height from IFC properties."""
        try:
            # Try to get dimensions from property sets
            psets = ifcopenshell.util.element.get_psets(door)
            
            width_mm = 900.0  # Default
            height_mm = 2100.0  # Default
            
            for pset_data in psets.values():
                for prop_name, prop_value in pset_data.items():
                    if isinstance(prop_value, (int, float)):
                        if 'width' in prop_name.lower():
                            width_mm = float(prop_value * 1000)  # Convert to mm
                        elif 'height' in prop_name.lower():
                            height_mm = float(prop_value * 1000)  # Convert to mm
            
            return width_mm, height_mm
            
        except Exception:
            return 900.0, 2100.0  # Default dimensions
    
    def _determine_door_type(self, door) -> DoorType:
        """Determine door type from IFC properties."""
        try:
            door_name = (door.Name or "").lower()
            
            # Check predefined type
            if hasattr(door, 'PredefinedType') and door.PredefinedType:
                ptype = door.PredefinedType.lower()
                if 'double' in ptype:
                    return DoorType.DOUBLE
                elif 'sliding' in ptype:
                    return DoorType.SLIDING
                elif 'revolving' in ptype:
                    return DoorType.REVOLVING
            
            # Check name
            if any(keyword in door_name for keyword in ['double', 'doble']):
                return DoorType.DOUBLE
            elif any(keyword in door_name for keyword in ['sliding', 'corredera']):
                return DoorType.SLIDING
            elif any(keyword in door_name for keyword in ['emergency', 'emergencia']):
                return DoorType.EMERGENCY_EXIT
            elif any(keyword in door_name for keyword in ['fire', 'fuego', 'rf']):
                return DoorType.FIRE_DOOR
            
            return DoorType.SINGLE
            
        except Exception:
            return DoorType.SINGLE
    
    def _get_element_position(self, element) -> Point3D:
        """Get element position from IFC placement."""
        try:
            if hasattr(element, 'ObjectPlacement') and element.ObjectPlacement:
                placement = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
                if placement is not None and len(placement) > 2:
                    return Point3D(
                        x=float(placement[0][3]),
                        y=float(placement[1][3]),
                        z=float(placement[2][3])
                    )
            
            return Point3D(x=0.0, y=0.0, z=0.0)
            
        except Exception:
            return Point3D(x=0.0, y=0.0, z=0.0)
    
    def _find_connected_spaces(self, door) -> Tuple[Optional[str], Optional[str]]:
        """Find spaces connected by a door."""
        # This is a simplified implementation
        # Full implementation would analyze spatial relationships
        return None, None
    
    def _is_emergency_exit(self, door) -> bool:
        """Check if a door is an emergency exit."""
        try:
            door_name = (door.Name or "").lower()
            if any(keyword in door_name for keyword in ['emergency', 'emergencia', 'exit', 'salida']):
                return True
            
            # Check predefined type
            if hasattr(door, 'PredefinedType') and door.PredefinedType:
                ptype = door.PredefinedType.lower()
                if 'emergency' in ptype or 'exit' in ptype:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _get_element_fire_rating(self, element) -> Optional[FireRating]:
        """Get fire rating from IFC properties."""
        try:
            psets = ifcopenshell.util.element.get_psets(element)
            
            for pset_data in psets.values():
                for prop_name, prop_value in pset_data.items():
                    if 'fire' in prop_name.lower() and 'rating' in prop_name.lower():
                        if isinstance(prop_value, str):
                            rating_str = prop_value.upper()
                            if 'RF-30' in rating_str or '30' in rating_str:
                                return FireRating.RF_30
                            elif 'RF-60' in rating_str or '60' in rating_str:
                                return FireRating.RF_60
                            elif 'RF-90' in rating_str or '90' in rating_str:
                                return FireRating.RF_90
                            elif 'RF-120' in rating_str or '120' in rating_str:
                                return FireRating.RF_120
            
            return None
            
        except Exception:
            return None
    
    def _get_wall_geometry(self, wall) -> Tuple[Optional[Point3D], Optional[Point3D]]:
        """Get wall start and end points from IFC geometry."""
        try:
            # This is a simplified implementation
            # Full implementation would extract actual wall geometry
            position = self._get_element_position(wall)
            
            # Create default start and end points
            start_point = Point3D(x=position.x, y=position.y, z=position.z)
            end_point = Point3D(x=position.x + 3.0, y=position.y, z=position.z)  # 3m wall
            
            return start_point, end_point
            
        except Exception:
            return None, None
    
    def _get_wall_thickness(self, wall) -> float:
        """Get wall thickness from IFC properties."""
        try:
            # Try to get thickness from material layers
            psets = ifcopenshell.util.element.get_psets(wall)
            
            for pset_data in psets.values():
                for prop_name, prop_value in pset_data.items():
                    if 'thickness' in prop_name.lower() and isinstance(prop_value, (int, float)):
                        return float(prop_value * 1000)  # Convert to mm
            
            return 200.0  # Default thickness
            
        except Exception:
            return 200.0
    
    def _get_wall_height(self, wall) -> float:
        """Get wall height from IFC properties."""
        try:
            psets = ifcopenshell.util.element.get_psets(wall)
            
            for pset_data in psets.values():
                for prop_name, prop_value in pset_data.items():
                    if 'height' in prop_name.lower() and isinstance(prop_value, (int, float)):
                        return float(prop_value * 1000)  # Convert to mm
            
            return 2700.0  # Default height
            
        except Exception:
            return 2700.0
    
    def _get_wall_material(self, wall) -> Optional[str]:
        """Get wall material from IFC properties."""
        try:
            # Try to get material information
            if hasattr(wall, 'HasAssociations'):
                for rel in wall.HasAssociations:
                    if rel.is_a('IfcRelAssociatesMaterial'):
                        material = rel.RelatingMaterial
                        if material and hasattr(material, 'Name'):
                            return material.Name
            
            return "concrete"  # Default material
            
        except Exception:
            return "concrete"
    
    def _derive_rooms_from_geometry(self):
        """Derive rooms from building geometry when no spaces are explicitly defined."""
        logger.info("Deriving rooms from building levels and geometric elements...")
        
        # Get all walls grouped by level
        walls_by_level = {}
        for level_name, level_data in self.levels.items():
            if level_data.get('walls'):
                walls_by_level[level_name] = level_data['walls']
        
        # Get all doors grouped by level
        doors_by_level = {}
        for level_name, level_data in self.levels.items():
            if level_data.get('doors'):
                doors_by_level[level_name] = level_data['doors']
        
        # Create typical rooms for each level based on building analysis
        for level_name, level_data in self.levels.items():
            level_walls = walls_by_level.get(level_name, [])
            level_doors = doors_by_level.get(level_name, [])
            
            if level_walls or level_doors:
                # Analyze the level to create typical rooms
                derived_rooms = self._analyze_level_for_rooms(level_name, level_walls, level_doors)
                level_data['rooms'].extend(derived_rooms)
            else:
                # Create a default room for this level
                default_room = self._create_default_room_for_level(level_name)
                level_data['rooms'].append(default_room)
    
    def _analyze_level_for_rooms(self, level_name: str, walls: List, doors: List) -> List[Room]:
        """Analyze a level's walls and doors to derive likely room spaces."""
        rooms = []
        
        # Calculate building extent from walls
        if walls:
            x_coords = []
            y_coords = []
            for wall in walls:
                x_coords.extend([wall.start_point.x, wall.end_point.x])
                y_coords.extend([wall.start_point.y, wall.end_point.y])
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            # Calculate approximate building area
            building_width = max_x - min_x
            building_length = max_y - min_y
            total_area = building_width * building_length
            
            # Create typical room layouts based on doors
            num_doors = len(doors)
            
            if num_doors == 0:
                # Single open space
                room = Room(
                    id=f"R_{level_name.replace(' ', '_')}_001",
                    name=f"Open Space - {level_name}",
                    area=max(total_area * 0.8, 50.0),  # 80% of building area, min 50m²
                    use="commercial",
                    level=level_name,
                    occupancy_load=max(int(total_area * 0.1), 5)
                )
                rooms.append(room)
                
            elif num_doors <= 3:
                # Small building with few rooms
                room_types = ["office", "meeting", "storage"]
                base_area = max(total_area / (num_doors + 1), 20.0)
                
                for i in range(min(num_doors + 1, 3)):
                    room_use = room_types[i % len(room_types)]
                    room = Room(
                        id=f"R_{level_name.replace(' ', '_')}_{i+1:03d}",
                        name=f"{room_use.title()} - {level_name}",
                        area=base_area + (i * 10),  # Vary room sizes
                        use=room_use,
                        level=level_name,
                        occupancy_load=max(int(base_area * 0.1), 2)
                    )
                    rooms.append(room)
                    
            else:
                # Larger building with multiple rooms
                room_types = ["office", "meeting", "office", "restroom", "storage", "corridor", "office"]
                base_area = max(total_area / (num_doors + 2), 15.0)
                
                for i in range(min(num_doors, 7)):  # Limit to 7 rooms per level
                    room_use = room_types[i % len(room_types)]
                    area_multiplier = 1.5 if room_use == "office" else 1.0
                    area_multiplier = 0.5 if room_use == "restroom" else area_multiplier
                    
                    room = Room(
                        id=f"R_{level_name.replace(' ', '_')}_{i+1:03d}",
                        name=f"{room_use.title()} {i+1} - {level_name}",
                        area=base_area * area_multiplier,
                        use=room_use,
                        level=level_name,
                        occupancy_load=max(int(base_area * area_multiplier * 0.1), 1)
                    )
                    rooms.append(room)
        else:
            # No walls found, create a simple default room
            room = self._create_default_room_for_level(level_name)
            rooms.append(room)
        
        logger.info(f"Derived {len(rooms)} rooms for level '{level_name}'")
        return rooms
    
    def _create_default_room_for_level(self, level_name: str) -> Room:
        """Create a default room for a level."""
        return Room(
            id=f"R_{level_name.replace(' ', '_')}_DEFAULT",
            name=f"General Space - {level_name}",
            area=80.0,
            use="commercial",
            level=level_name,
            occupancy_load=8
        )

    
    def _classify_level_function(self, level_name: str, doors_count: int, walls_count: int) -> str:
        """Classify the function of a building level."""
        name_lower = level_name.lower()
        
        # Main activity levels (high door count)
        if doors_count > 10:
            if any(keyword in name_lower for keyword in ['muelle', 'loading', 'dock']):
                return "Loading/Shipping Operations"
            elif any(keyword in name_lower for keyword in ['taller', 'workshop', 'production']):
                return "Manufacturing/Workshop"
            elif any(keyword in name_lower for keyword in ['main', 'principal', 'ground', 'pb']):
                return "Main Operations Floor"
            else:
                return "Active Operations Level"
        
        # Service/utility levels (walls but few doors)
        elif walls_count > 0 and doors_count < 3:
            if any(keyword in name_lower for keyword in ['roof', 'cubierta', 'peto']):
                return "Roof/Structural Level"
            elif any(keyword in name_lower for keyword in ['mechanical', 'hvac', 'utility']):
                return "Mechanical/Utility Level"
            elif any(keyword in name_lower for keyword in ['basement', 'sotano', 'foundation']):
                return "Foundation/Service Level"
            else:
                return "Service/Structural Level"
        
        # Mixed use levels
        elif doors_count > 0 and walls_count > 0:
            if any(keyword in name_lower for keyword in ['altillo', 'mezzanine']):
                return "Mezzanine/Intermediate Level"
            elif any(keyword in name_lower for keyword in ['office', 'admin']):
                return "Administrative Level"
            else:
                return "Mixed Use Level"
        
        # Empty/structural only
        else:
            return "Structural/Unused Level"
    

    
    def _classify_level_function(self, level_name: str, doors_count: int, walls_count: int) -> str:
        """Classify the function of a building level."""
        name_lower = level_name.lower()
        
        # Main activity levels (high door count)
        if doors_count > 10:
            if any(keyword in name_lower for keyword in ['muelle', 'loading', 'dock']):
                return "Loading/Shipping Operations"
            elif any(keyword in name_lower for keyword in ['taller', 'workshop', 'production']):
                return "Manufacturing/Workshop"
            elif any(keyword in name_lower for keyword in ['main', 'principal', 'ground', 'pb']):
                return "Main Operations Floor"
            else:
                return "Active Operations Level"
        
        # Service/utility levels (walls but few doors)
        elif walls_count > 0 and doors_count < 3:
            if any(keyword in name_lower for keyword in ['roof', 'cubierta', 'peto']):
                return "Roof/Structural Level"
            elif any(keyword in name_lower for keyword in ['mechanical', 'hvac', 'utility']):
                return "Mechanical/Utility Level"
            elif any(keyword in name_lower for keyword in ['basement', 'sotano', 'foundation']):
                return "Foundation/Service Level"
            else:
                return "Service/Structural Level"
        
        # Mixed use levels
        elif doors_count > 0 and walls_count > 0:
            if any(keyword in name_lower for keyword in ['altillo', 'mezzanine']):
                return "Mezzanine/Intermediate Level"
            elif any(keyword in name_lower for keyword in ['office', 'admin']):
                return "Administrative Level"
            else:
                return "Mixed Use Level"
        
        # Empty/structural only
        else:
            return "Structural/Unused Level"
    

    def _create_project_metadata(self) -> ProjectMetadata:
        """Create project metadata from IFC file information."""
        try:
            # Get project information from IFC
            project = self.ifc_file.by_type('IfcProject')[0] if self.ifc_file.by_type('IfcProject') else None
            building = self.ifc_file.by_type('IfcBuilding')[0] if self.ifc_file.by_type('IfcBuilding') else None
            
            project_name = "IFC Project"
            building_type = "commercial"
            
            if project and project.Name:
                project_name = project.Name
            elif building and building.Name:
                project_name = building.Name
            
            if building and hasattr(building, 'CompositionType'):
                building_type = str(building.CompositionType).lower()
            
            # Calculate total area
            total_area = sum(room.area for room in self.rooms)
            for level_data in self.levels.values():
                total_area += sum(room.area for room in level_data['rooms'])
            
            return ProjectMetadata(
                project_name=project_name,
                file_name=getattr(self.ifc_file, 'name', 'unknown.ifc'),
                building_type=building_type,
                total_area=total_area,
                number_of_levels=len(self.levels) if self.levels else 1,
                created_date=datetime.now().isoformat(),
                modified_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error creating project metadata: {e}")
            return ProjectMetadata(
                project_name="IFC Project",
                file_name="unknown.ifc",
                building_type="commercial",
                total_area=0.0,
                number_of_levels=1,
                created_date=datetime.now().isoformat(),
                modified_date=datetime.now().isoformat()
            )


def extract_from_ifc(file_path: Path) -> Project:
    """
    Extract building elements from an IFC file.
    
    Args:
        file_path: Path to the IFC file
        
    Returns:
        Project object with extracted data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be processed
        ImportError: If ifcopenshell is not available
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extractor = IFCExtractor()
    
    if not extractor.load_file(file_path):
        raise ValueError(f"Could not load IFC file: {file_path}")
    
    return extractor.extract_all()


def save_to_json(project: Project, output_path: Path) -> None:
    """
    Save project data to JSON file.
    
    Args:
        project: Project object to save
        output_path: Path where to save the JSON file
    """
    # Convert to dict and then to JSON
    project_dict = project.dict()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Project data saved to: {output_path}")


def main():
    """CLI interface for IFC extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract building elements from IFC files')
    parser.add_argument('input_file', type=Path, help='Input IFC file')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Extract data
        project = extract_from_ifc(args.input_file)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = args.input_file.with_suffix('.json')
        
        # Save to JSON
        save_to_json(project, output_path)
        
        print(f"✅ IFC extraction complete!")
        print(f"   - Rooms: {len(project.get_all_rooms())}")
        print(f"   - Doors: {len(project.get_all_doors())}")
        print(f"   - Walls: {len(project.get_all_walls())}")
        print(f"   - Output: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())