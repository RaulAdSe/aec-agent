"""
Revit extraction module for AEC Compliance Agent.

This module extracts building data from Revit files using the Revit API.
It can be run as a standalone script or integrated into the extraction pipeline.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Revit API imports (only available when running inside Revit)
try:
    import clr
    clr.AddReference('RevitAPI')
    clr.AddReference('RevitServices')
    clr.AddReference('RevitNodes')
    
    from Autodesk.Revit.DB import *
    from Autodesk.Revit.UI import *
    from RevitServices.Persistence import DocumentManager
    from RevitServices.Transactions import TransactionManager
    
    REVIT_AVAILABLE = True
except ImportError:
    REVIT_AVAILABLE = False
    # Mock classes for development outside Revit
    class MockElement:
        def __init__(self, id_val, name=""):
            self.Id = id_val
            self.Name = name
    
    class MockDocument:
        def __init__(self):
            self.Title = "Mock Document"
    
    class MockFilteredElementCollector:
        def __init__(self, doc, element_type):
            self.doc = doc
            self.element_type = element_type
            self.elements = []
        
        def WhereElementIsNotElementType(self):
            return self
        
        def ToElements(self):
            return self.elements

try:
    from ..schemas import (
        Project, ProjectMetadata, Room, Door, Wall, 
        FireEquipment, Sector, Level
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from schemas import (
        Project, ProjectMetadata, Room, Door, Wall, 
        FireEquipment, Sector, Level
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RevitExtractor:
    """Extract building data from Revit files using the Revit API."""
    
    def __init__(self):
        self.rooms: List[Room] = []
        self.doors: List[Door] = []
        self.walls: List[Wall] = []
        self.fire_equipment: List[FireEquipment] = []
        self.sectors: List[Sector] = []
        self.doc = None
        
    def extract_from_document(self, doc=None, project_name: str = None, level_name: str = "Level 1") -> Project:
        """
        Extract building data from Revit document.
        
        Args:
            doc: Revit document (if None, uses active document)
            project_name: Name of the project
            level_name: Name of the building level
            
        Returns:
            Project object with extracted data
        """
        if not REVIT_AVAILABLE:
            logger.warning("Revit API not available. Returning mock data.")
            return self._create_mock_project(project_name, level_name)
        
        # Get document
        if doc is None:
            try:
                self.doc = DocumentManager.Instance.CurrentDBDocument
            except:
                logger.error("No active Revit document found")
                return self._create_mock_project(project_name, level_name)
        else:
            self.doc = doc
        
        logger.info(f"Extracting from Revit document: {self.doc.Title}")
        
        # Extract project name from document if not provided
        if not project_name:
            project_name = self.doc.Title
        
        # Extract building elements
        self._extract_rooms(level_name)
        self._extract_doors(level_name)
        self._extract_walls(level_name)
        self._extract_fire_equipment(level_name)
        self._extract_sectors(level_name)
        
        # Create metadata
        metadata = ProjectMetadata(
            project_name=project_name,
            level_name=level_name,
            extraction_date=datetime.now(),
            source_file=self.doc.Title,
            building_use="commercial"  # Default, can be updated
        )
        
        # Create level
        level = Level(
            name=level_name,
            number=0,  # Ground floor by default
            height=0.0,
            rooms=self.rooms,
            doors=self.doors,
            walls=self.walls,
            fire_equipment=self.fire_equipment,
            sectors=self.sectors
        )
        
        # Create and return project
        project = Project(
            metadata=metadata,
            levels=[level],
            rooms=self.rooms,
            doors=self.doors,
            walls=self.walls,
            fire_equipment=self.fire_equipment,
            sectors=self.sectors
        )
        
        logger.info(f"Extraction complete: {len(self.rooms)} rooms, {len(self.doors)} doors, "
                   f"{len(self.walls)} walls, {len(self.fire_equipment)} fire equipment, "
                   f"{len(self.sectors)} sectors")
        
        return project
    
    def _extract_rooms(self, level_name: str):
        """Extract rooms from Revit document."""
        if not REVIT_AVAILABLE:
            return
        
        try:
            # Get all room elements
            room_collector = FilteredElementCollector(self.doc).OfClass(Room)
            rooms = room_collector.WhereElementIsNotElementType().ToElements()
            
            room_id = 1
            for room in rooms:
                try:
                    # Get room properties
                    room_name = room.get_Parameter(BuiltInParameter.ROOM_NAME).AsString()
                    if not room_name:
                        room_name = f"Room {room_id}"
                    
                    # Get room area
                    area_param = room.get_Parameter(BuiltInParameter.ROOM_AREA)
                    area = area_param.AsDouble() * 0.092903  # Convert from sq ft to sq m
                    
                    # Get room boundary
                    boundary = self._get_room_boundary(room)
                    
                    # Get room use type
                    use_type = self._get_room_use_type(room)
                    
                    # Create room object
                    room_obj = Room(
                        id=f"R{room_id:03d}",
                        name=room_name,
                        level=level_name,
                        boundary=boundary,
                        use_type=use_type,
                        area=area
                    )
                    
                    self.rooms.append(room_obj)
                    room_id += 1
                    logger.debug(f"Extracted room: {room_name} ({area:.2f} m²)")
                    
                except Exception as e:
                    logger.warning(f"Error extracting room: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting rooms: {e}")
    
    def _extract_doors(self, level_name: str):
        """Extract doors from Revit document."""
        if not REVIT_AVAILABLE:
            return
        
        try:
            # Get all door elements
            door_collector = FilteredElementCollector(self.doc).OfClass(FamilyInstance)
            doors = door_collector.WhereElementIsNotElementType().ToElements()
            
            door_id = 1
            for door in doors:
                try:
                    # Check if it's a door family
                    if door.Symbol.Family.FamilyCategory.Name == "Doors":
                        # Get door properties
                        door_name = door.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS).AsString()
                        if not door_name:
                            door_name = f"Door {door_id}"
                        
                        # Get door width
                        width_param = door.get_Parameter(BuiltInParameter.DOOR_WIDTH)
                        width = width_param.AsDouble() * 0.3048  # Convert from feet to meters
                        
                        # Get door position
                        location = door.Location
                        if hasattr(location, 'Point'):
                            position = [location.Point.X * 0.3048, location.Point.Y * 0.3048]
                        else:
                            position = [0, 0]
                        
                        # Determine door type
                        door_type = self._get_door_type(door)
                        is_egress = self._is_egress_door(door)
                        fire_rating = self._get_door_fire_rating(door)
                        
                        # Create door object
                        door_obj = Door(
                            id=f"D{door_id:03d}",
                            position=position,
                            width=width,
                            door_type=door_type,
                            is_egress=is_egress,
                            fire_rating=fire_rating
                        )
                        
                        self.doors.append(door_obj)
                        door_id += 1
                        logger.debug(f"Extracted door: {door_name} ({width:.2f}m)")
                        
                except Exception as e:
                    logger.warning(f"Error extracting door: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting doors: {e}")
    
    def _extract_walls(self, level_name: str):
        """Extract walls from Revit document."""
        if not REVIT_AVAILABLE:
            return
        
        try:
            # Get all wall elements
            wall_collector = FilteredElementCollector(self.doc).OfClass(Wall)
            walls = wall_collector.WhereElementIsNotElementType().ToElements()
            
            wall_id = 1
            for wall in walls:
                try:
                    # Get wall properties
                    wall_name = wall.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS).AsString()
                    if not wall_name:
                        wall_name = f"Wall {wall_id}"
                    
                    # Get wall location curve
                    location_curve = wall.Location.Curve
                    start_point = location_curve.GetEndPoint(0)
                    end_point = location_curve.GetEndPoint(1)
                    
                    start = [start_point.X * 0.3048, start_point.Y * 0.3048]
                    end = [end_point.X * 0.3048, end_point.Y * 0.3048]
                    
                    # Calculate length
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    
                    # Get wall thickness
                    thickness_param = wall.get_Parameter(BuiltInParameter.WALL_ATTR_WIDTH_PARAM)
                    thickness = thickness_param.AsDouble() * 0.3048 if thickness_param else 0.15
                    
                    # Get fire rating
                    fire_rating = self._get_wall_fire_rating(wall)
                    
                    # Check if exterior wall
                    is_exterior = self._is_exterior_wall(wall)
                    
                    # Create wall object
                    wall_obj = Wall(
                        id=f"W{wall_id:03d}",
                        start=start,
                        end=end,
                        length=length,
                        thickness=thickness,
                        fire_rating=fire_rating,
                        is_exterior=is_exterior
                    )
                    
                    self.walls.append(wall_obj)
                    wall_id += 1
                    logger.debug(f"Extracted wall: {wall_name} ({length:.2f}m)")
                    
                except Exception as e:
                    logger.warning(f"Error extracting wall: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting walls: {e}")
    
    def _extract_fire_equipment(self, level_name: str):
        """Extract fire safety equipment from Revit document."""
        if not REVIT_AVAILABLE:
            return
        
        try:
            # Get all family instances (equipment)
            equipment_collector = FilteredElementCollector(self.doc).OfClass(FamilyInstance)
            equipment = equipment_collector.WhereElementIsNotElementType().ToElements()
            
            equipment_id = 1
            for eq in equipment:
                try:
                    # Check if it's fire safety equipment
                    category_name = eq.Symbol.Family.FamilyCategory.Name
                    if any(keyword in category_name.upper() for keyword in 
                          ['FIRE', 'FUEGO', 'EXTINGUISHER', 'EXTINTOR', 'ALARM', 'ALARMA']):
                        
                        # Get equipment properties
                        eq_name = eq.get_Parameter(BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS).AsString()
                        if not eq_name:
                            eq_name = f"Equipment {equipment_id}"
                        
                        # Get position
                        location = eq.Location
                        if hasattr(location, 'Point'):
                            position = [location.Point.X * 0.3048, location.Point.Y * 0.3048]
                        else:
                            position = [0, 0]
                        
                        # Determine equipment type
                        equipment_type = self._classify_fire_equipment(category_name, eq_name)
                        
                        # Create equipment object
                        eq_obj = FireEquipment(
                            id=f"FE{equipment_id:03d}",
                            equipment_type=equipment_type,
                            position=position,
                            floor_level=level_name,
                            status="active"
                        )
                        
                        # Set coverage radius based on type
                        if equipment_type == "extinguisher":
                            eq_obj.coverage_radius = 15.0
                        elif equipment_type == "hydrant":
                            eq_obj.coverage_radius = 25.0
                        elif equipment_type == "sprinkler":
                            eq_obj.coverage_radius = 3.5
                        
                        self.fire_equipment.append(eq_obj)
                        equipment_id += 1
                        logger.debug(f"Extracted {equipment_type}: {eq_name}")
                        
                except Exception as e:
                    logger.warning(f"Error extracting equipment: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error extracting fire equipment: {e}")
    
    def _extract_sectors(self, level_name: str):
        """Extract fire sectors from Revit document."""
        if not REVIT_AVAILABLE:
            return
        
        # Fire sectors are typically represented as areas or rooms with specific properties
        # This is a simplified implementation
        sector_id = 1
        
        # Group rooms by fire sector if they have sector information
        sector_rooms = {}
        for room in self.rooms:
            # In a real implementation, you would check room parameters for sector info
            # For now, we'll create sectors based on room proximity or other criteria
            pass
    
    def _get_room_boundary(self, room) -> List[List[float]]:
        """Get room boundary points from Revit room."""
        if not REVIT_AVAILABLE:
            return [[0, 0], [5, 0], [5, 4], [0, 4]]
        
        try:
            # Get room boundary
            boundary = room.GetBoundarySegments(SpatialElementBoundaryOptions())
            if boundary and len(boundary) > 0:
                points = []
                for segment in boundary[0]:  # Take first boundary
                    curve = segment.GetCurve()
                    start = curve.GetEndPoint(0)
                    points.append([start.X * 0.3048, start.Y * 0.3048])
                return points
        except:
            pass
        
        # Fallback to simple rectangle
        return [[0, 0], [5, 0], [5, 4], [0, 4]]
    
    def _get_room_use_type(self, room) -> str:
        """Get room use type from Revit room."""
        if not REVIT_AVAILABLE:
            return "general"
        
        try:
            # Get room use type parameter
            use_param = room.get_Parameter(BuiltInParameter.ROOM_USE_TYPE)
            if use_param:
                use_type = use_param.AsString()
                return self._map_room_use_type(use_type)
        except:
            pass
        
        return "general"
    
    def _map_room_use_type(self, revit_use_type: str) -> str:
        """Map Revit room use type to our schema."""
        use_type_mapping = {
            "Office": "office",
            "Conference Room": "meeting_room",
            "Storage": "storage",
            "Restroom": "bathroom",
            "Corridor": "corridor",
            "Lobby": "lobby",
            "Stairs": "stairs",
            "Retail": "retail",
            "Reception": "reception"
        }
        
        return use_type_mapping.get(revit_use_type, "general")
    
    def _get_door_type(self, door) -> str:
        """Get door type from Revit door."""
        if not REVIT_AVAILABLE:
            return "single"
        
        try:
            # Check door family name for type hints
            family_name = door.Symbol.Family.Name.upper()
            if "DOUBLE" in family_name or "DOBLE" in family_name:
                return "double"
            elif "SLIDING" in family_name or "CORRED" in family_name:
                return "sliding"
            elif "EMERGENCY" in family_name or "EMERGENCIA" in family_name:
                return "emergency"
        except:
            pass
        
        return "single"
    
    def _is_egress_door(self, door) -> bool:
        """Check if door is an egress door."""
        if not REVIT_AVAILABLE:
            return False
        
        try:
            # Check door family name
            family_name = door.Symbol.Family.Name.upper()
            if any(keyword in family_name for keyword in ['EMERGENCY', 'EMERGENCIA', 'EXIT', 'SALIDA']):
                return True
            
            # Check door parameters
            egress_param = door.get_Parameter(BuiltInParameter.DOOR_EGRESS)
            if egress_param and egress_param.AsInteger() == 1:
                return True
        except:
            pass
        
        return False
    
    def _get_door_fire_rating(self, door) -> Optional[str]:
        """Get door fire rating."""
        if not REVIT_AVAILABLE:
            return None
        
        try:
            # Check door family name for fire rating
            family_name = door.Symbol.Family.Name.upper()
            if "EI-120" in family_name:
                return "EI-120"
            elif "EI-90" in family_name:
                return "EI-90"
            elif "EI-60" in family_name:
                return "EI-60"
            elif "EI-30" in family_name:
                return "EI-30"
        except:
            pass
        
        return None
    
    def _get_wall_fire_rating(self, wall) -> Optional[str]:
        """Get wall fire rating."""
        if not REVIT_AVAILABLE:
            return None
        
        try:
            # Check wall type name for fire rating
            wall_type = wall.WallType.Name.upper()
            if "EI-120" in wall_type:
                return "EI-120"
            elif "EI-90" in wall_type:
                return "EI-90"
            elif "EI-60" in wall_type:
                return "EI-60"
            elif "EI-30" in wall_type:
                return "EI-30"
        except:
            pass
        
        return None
    
    def _is_exterior_wall(self, wall) -> bool:
        """Check if wall is exterior."""
        if not REVIT_AVAILABLE:
            return False
        
        try:
            # Check wall function parameter
            function_param = wall.get_Parameter(BuiltInParameter.WALL_FUNCTION)
            if function_param:
                function = function_param.AsInteger()
                return function == 1  # 1 = Exterior
        except:
            pass
        
        return False
    
    def _classify_fire_equipment(self, category_name: str, equipment_name: str) -> str:
        """Classify fire equipment type."""
        name_upper = (category_name + " " + equipment_name).upper()
        
        if any(keyword in name_upper for keyword in ['EXTINTOR', 'EXTINGUISHER']):
            return "extinguisher"
        elif any(keyword in name_upper for keyword in ['BIE', 'HIDRANT', 'HYDRANT']):
            return "hydrant"
        elif any(keyword in name_upper for keyword in ['SPRINKLER', 'ROCIADOR']):
            return "sprinkler"
        elif any(keyword in name_upper for keyword in ['ALARMA', 'ALARM', 'DETECTOR']):
            return "alarm"
        elif any(keyword in name_upper for keyword in ['EMERGENCIA', 'EMERGENCY']):
            return "emergency_light"
        elif any(keyword in name_upper for keyword in ['SEÑAL', 'SIGN', 'SALIDA']):
            return "exit_sign"
        
        return "general"
    
    def _create_mock_project(self, project_name: str = None, level_name: str = "Level 1") -> Project:
        """Create mock project data when Revit API is not available."""
        if not project_name:
            project_name = "Mock Revit Project"
        
        # Create mock data similar to the existing mock files
        metadata = ProjectMetadata(
            project_name=project_name,
            level_name=level_name,
            extraction_date=datetime.now(),
            source_file="mock_revit_file.rvt",
            building_use="commercial"
        )
        
        # Create mock rooms
        rooms = [
            Room(
                id="R001",
                name="Recepción",
                level=level_name,
                boundary=[[0, 0], [10, 0], [10, 8], [0, 8]],
                use_type="reception",
                area=80.0
            ),
            Room(
                id="R002",
                name="Zona Venta 1",
                level=level_name,
                boundary=[[10, 0], [25, 0], [25, 15], [10, 15]],
                use_type="retail",
                area=225.0
            ),
            Room(
                id="R003",
                name="Almacén",
                level=level_name,
                boundary=[[25, 0], [35, 0], [35, 10], [25, 10]],
                use_type="storage",
                area=100.0
            )
        ]
        
        # Create mock doors
        doors = [
            Door(
                id="D001",
                position=[5, 0],
                width=1.2,
                door_type="entrance",
                is_egress=True
            ),
            Door(
                id="D002",
                position=[10, 7.5],
                width=0.9,
                door_type="single",
                is_egress=False
            ),
            Door(
                id="D003",
                position=[30, 5],
                width=1.0,
                door_type="single",
                is_egress=False
            )
        ]
        
        # Create mock walls
        walls = [
            Wall(
                id="W001",
                start=[0, 0],
                end=[35, 0],
                length=35.0,
                thickness=0.2,
                is_exterior=True,
                fire_rating="EI-90"
            ),
            Wall(
                id="W002",
                start=[0, 15],
                end=[35, 15],
                length=35.0,
                thickness=0.2,
                is_exterior=True,
                fire_rating="EI-90"
            )
        ]
        
        # Create mock fire equipment
        fire_equipment = [
            FireEquipment(
                id="FE001",
                equipment_type="extinguisher",
                position=[15, 12],
                coverage_radius=15.0,
                floor_level=level_name,
                status="active"
            ),
            FireEquipment(
                id="FE002",
                equipment_type="hydrant",
                position=[20, 2],
                coverage_radius=25.0,
                floor_level=level_name,
                status="active"
            )
        ]
        
        # Create level
        level = Level(
            name=level_name,
            number=0,
            height=0.0,
            rooms=rooms,
            doors=doors,
            walls=walls,
            fire_equipment=fire_equipment,
            sectors=[]
        )
        
        # Create project
        project = Project(
            metadata=metadata,
            levels=[level],
            rooms=rooms,
            doors=doors,
            walls=walls,
            fire_equipment=fire_equipment,
            sectors=[]
        )
        
        logger.info("Created mock Revit project data")
        return project


def extract_and_save_revit(doc=None, output_path: Optional[Path] = None, 
                          project_name: Optional[str] = None) -> Project:
    """
    Extract data from Revit document and save to JSON.
    
    Args:
        doc: Revit document (if None, uses active document)
        output_path: Optional path for JSON output
        project_name: Optional project name
        
    Returns:
        Extracted Project object
    """
    extractor = RevitExtractor()
    project = extractor.extract_from_document(doc, project_name)
    
    # Save to JSON if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(project.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved extracted data to: {output_path}")
    
    return project


# Script for running inside Revit
if __name__ == "__main__":
    # This script can be run inside Revit using pyRevit or similar tools
    try:
        project = extract_and_save_revit()
        print(f"Extraction completed: {len(project.rooms)} rooms, {len(project.doors)} doors")
    except Exception as e:
        print(f"Error during extraction: {e}")
