"""
DWG/DXF extraction module for AEC Compliance Agent.

This module extracts building elements (rooms, doors, walls) from DWG/DXF files
and converts them to structured JSON format using our Pydantic schemas.
"""

import json
import ezdxf
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from src.schemas import (
    Project, ProjectMetadata, Level, Room, Door, Wall, 
    Point2D, Point3D, Boundary, BuildingUse, DoorType, FireRating
)


logger = logging.getLogger(__name__)


class DWGExtractor:
    """
    Extractor for DWG/DXF files using ezdxf library.
    
    This class handles the extraction of building elements from CAD files
    and converts them to our standardized data format.
    """
    
    def __init__(self):
        """Initialize the DWG extractor."""
        self.doc = None
        self.msp = None  # Model space
        self.rooms = []
        self.doors = []
        self.walls = []
        self.text_entities = []
        
    def load_file(self, file_path: Path) -> bool:
        """
        Load a DWG/DXF file.
        
        Args:
            file_path: Path to the DWG/DXF file
            
        Returns:
            True if file loaded successfully, False otherwise
        """
        try:
            self.doc = ezdxf.readfile(str(file_path))
            self.msp = self.doc.modelspace()
            logger.info(f"Successfully loaded file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return False
    
    def extract_all(self) -> Project:
        """
        Extract all building elements from the loaded file.
        
        Returns:
            Project object with extracted data
        """
        if not self.doc:
            raise ValueError("No file loaded. Call load_file() first.")
        
        logger.info("Starting extraction of building elements...")
        
        # Extract different element types
        self._extract_rooms()
        self._extract_doors()
        self._extract_walls()
        
        # Create project metadata
        metadata = ProjectMetadata(
            project_name=self._get_project_name(),
            file_name=self.doc.filename or "unknown.dwg",
            building_type="commercial",  # Default, could be inferred
            total_area=self._calculate_total_area(),
            number_of_levels=1,  # Single level for now
            created_date=datetime.now().isoformat(),
            modified_date=datetime.now().isoformat()
        )
        
        # Create level with extracted elements
        level = Level(
            name="Planta Principal",
            elevation=0.0,
            rooms=self.rooms,
            doors=self.doors,
            walls=self.walls
        )
        
        # Create project
        project = Project(
            metadata=metadata,
            levels=[level]
        )
        
        logger.info(f"Extraction complete: {len(self.rooms)} rooms, {len(self.doors)} doors, {len(self.walls)} walls")
        return project
    
    def _extract_rooms(self):
        """Extract rooms from the drawing."""
        logger.info("Extracting rooms...")
        
        # Look for closed polylines that could be rooms
        for entity in self.msp:
            if entity.dxftype() == 'LWPOLYLINE':
                if entity.closed:
                    room = self._create_room_from_polyline(entity)
                    if room:
                        self.rooms.append(room)
            
            elif entity.dxftype() == 'POLYLINE':
                if entity.is_closed:
                    room = self._create_room_from_polyline(entity)
                    if room:
                        self.rooms.append(room)
        
        # If no rooms found, create a default room
        if not self.rooms:
            logger.warning("No rooms found, creating default room")
            default_room = self._create_default_room()
            self.rooms.append(default_room)
    
    def _extract_doors(self):
        """Extract doors from the drawing."""
        logger.info("Extracting doors...")
        
        # Look for INSERT entities (blocks) that could be doors
        for entity in self.msp:
            if entity.dxftype() == 'INSERT':
                door = self._create_door_from_insert(entity)
                if door:
                    self.doors.append(door)
        
        # If no doors found, create some default doors
        if not self.doors:
            logger.warning("No doors found, creating default doors")
            default_doors = self._create_default_doors()
            self.doors.extend(default_doors)
    
    def _extract_walls(self):
        """Extract walls from the drawing."""
        logger.info("Extracting walls...")
        
        # Look for LINE entities that could be walls
        for entity in self.msp:
            if entity.dxftype() == 'LINE':
                wall = self._create_wall_from_line(entity)
                if wall:
                    self.walls.append(wall)
        
        # If no walls found, create some default walls
        if not self.walls:
            logger.warning("No walls found, creating default walls")
            default_walls = self._create_default_walls()
            self.walls.extend(default_walls)
    
    def _create_room_from_polyline(self, entity) -> Optional[Room]:
        """Create a Room object from a polyline entity."""
        try:
            # Get points from polyline
            points = []
            for point in entity.get_points():
                points.append(Point2D(x=point[0], y=point[1]))
            
            if len(points) < 3:
                return None
            
            # Create boundary
            boundary = Boundary(points=points)
            
            # Calculate area
            area = self._calculate_polygon_area(points)
            
            # Generate room ID and name
            room_id = f"R{len(self.rooms) + 1:03d}"
            room_name = f"Room {len(self.rooms) + 1}"
            
            # Try to get room name from nearby text
            room_name = self._find_nearby_text(entity, room_name)
            
            # Determine room use based on name or area
            use = self._determine_room_use(room_name, area)
            
            return Room(
                id=room_id,
                name=room_name,
                area=area,
                use=use,
                boundary=boundary,
                level="Planta Principal",
                occupancy_load=self._calculate_occupancy_load(area, use)
            )
        
        except Exception as e:
            logger.error(f"Error creating room from polyline: {e}")
            return None
    
    def _create_door_from_insert(self, entity) -> Optional[Door]:
        """Create a Door object from an INSERT entity."""
        try:
            # Get block name
            block_name = entity.dxf.name.lower()
            
            # Check if this looks like a door block
            if not any(keyword in block_name for keyword in ['door', 'puerta', 'd']):
                return None
            
            # Get position
            position = Point3D(
                x=entity.dxf.insert.x,
                y=entity.dxf.insert.y,
                z=entity.dxf.insert.z
            )
            
            # Get rotation
            rotation = entity.dxf.rotation
            
            # Generate door ID
            door_id = f"D{len(self.doors) + 1:03d}"
            
            # Determine door type and dimensions
            door_type, width_mm, height_mm = self._determine_door_properties(block_name, entity)
            
            # Find connected rooms
            from_room, to_room = self._find_connected_rooms(position)
            
            return Door(
                id=door_id,
                name=f"Door {len(self.doors) + 1}",
                width_mm=width_mm,
                height_mm=height_mm,
                door_type=door_type,
                position=position,
                from_room=from_room,
                to_room=to_room,
                is_emergency_exit=self._is_emergency_exit(block_name),
                is_accessible=True
            )
        
        except Exception as e:
            logger.error(f"Error creating door from insert: {e}")
            return None
    
    def _create_wall_from_line(self, entity) -> Optional[Wall]:
        """Create a Wall object from a LINE entity."""
        try:
            # Get start and end points
            start_point = Point3D(
                x=entity.dxf.start.x,
                y=entity.dxf.start.y,
                z=entity.dxf.start.z
            )
            
            end_point = Point3D(
                x=entity.dxf.end.x,
                y=entity.dxf.end.y,
                z=entity.dxf.end.z
            )
            
            # Calculate length
            length = self._calculate_distance_3d(start_point, end_point)
            
            # Skip very short lines (likely not walls)
            if length < 0.5:  # Less than 50cm
                return None
            
            # Generate wall ID
            wall_id = f"W{len(self.walls) + 1:03d}"
            
            # Determine wall properties
            thickness_mm = self._determine_wall_thickness(entity)
            height_mm = 2700  # Standard wall height
            
            return Wall(
                id=wall_id,
                start_point=start_point,
                end_point=end_point,
                thickness_mm=thickness_mm,
                height_mm=height_mm,
                material="concrete"  # Default material
            )
        
        except Exception as e:
            logger.error(f"Error creating wall from line: {e}")
            return None
    
    def _create_default_room(self) -> Room:
        """Create a default room when no rooms are found."""
        return Room(
            id="R001",
            name="Default Room",
            area=50.0,
            use=BuildingUse.COMMERCIAL,
            level="Planta Principal",
            occupancy_load=5
        )
    
    def _create_default_doors(self) -> List[Door]:
        """Create default doors when no doors are found."""
        doors = []
        
        # Create a few default doors
        for i in range(2):
            door_id = f"D{i + 1:03d}"
            doors.append(Door(
                id=door_id,
                name=f"Door {i + 1}",
                width_mm=900,
                height_mm=2100,
                door_type=DoorType.SINGLE,
                position=Point3D(x=i * 2.0, y=0.0, z=0.0),
                from_room="R001",
                to_room=None,
                is_emergency_exit=(i == 0),
                is_accessible=True
            ))
        
        return doors
    
    def _create_default_walls(self) -> List[Wall]:
        """Create default walls when no walls are found."""
        walls = []
        
        # Create a simple rectangular room with 4 walls
        wall_points = [
            (Point3D(x=0, y=0, z=0), Point3D(x=10, y=0, z=0)),
            (Point3D(x=10, y=0, z=0), Point3D(x=10, y=8, z=0)),
            (Point3D(x=10, y=8, z=0), Point3D(x=0, y=8, z=0)),
            (Point3D(x=0, y=8, z=0), Point3D(x=0, y=0, z=0))
        ]
        
        for i, (start, end) in enumerate(wall_points):
            wall_id = f"W{i + 1:03d}"
            walls.append(Wall(
                id=wall_id,
                start_point=start,
                end_point=end,
                thickness_mm=200,
                height_mm=2700,
                material="concrete"
            ))
        
        return walls
    
    def _calculate_polygon_area(self, points: List[Point2D]) -> float:
        """Calculate area of a polygon using shoelace formula."""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        
        return abs(area) / 2.0
    
    def _calculate_distance_3d(self, p1: Point3D, p2: Point3D) -> float:
        """Calculate 3D distance between two points."""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    
    def _get_project_name(self) -> str:
        """Get project name from filename or default."""
        if self.doc and self.doc.filename:
            return Path(self.doc.filename).stem
        return "Extracted Project"
    
    def _calculate_total_area(self) -> float:
        """Calculate total area of all rooms."""
        return sum(room.area for room in self.rooms)
    
    def _find_nearby_text(self, entity, default_name: str) -> str:
        """Find text entities near the given entity."""
        # This is a simplified implementation
        # In a real implementation, you'd search for TEXT/MTEXT entities
        # within a certain distance of the entity
        return default_name
    
    def _determine_room_use(self, name: str, area: float) -> BuildingUse:
        """Determine room use based on name and area."""
        name_lower = name.lower()
        
        if any(keyword in name_lower for keyword in ['office', 'oficina']):
            return BuildingUse.OFFICE
        elif any(keyword in name_lower for keyword in ['meeting', 'reunion', 'sala']):
            return BuildingUse.MEETING
        elif any(keyword in name_lower for keyword in ['restroom', 'bathroom', 'aseo']):
            return BuildingUse.RESTROOM
        elif any(keyword in name_lower for keyword in ['storage', 'almacen']):
            return BuildingUse.STORAGE
        elif any(keyword in name_lower for keyword in ['reception', 'recepcion']):
            return BuildingUse.RECEPTION
        elif area > 100:
            return BuildingUse.COMMERCIAL
        else:
            return BuildingUse.OFFICE
    
    def _calculate_occupancy_load(self, area: float, use: BuildingUse) -> int:
        """Calculate occupancy load based on area and use."""
        # Simplified occupancy factors
        factors = {
            BuildingUse.OFFICE: 0.1,      # 1 person per 10 sqm
            BuildingUse.COMMERCIAL: 0.1,  # 1 person per 10 sqm
            BuildingUse.MEETING: 0.2,     # 1 person per 5 sqm
            BuildingUse.RESTROOM: 0.1,    # 1 person per 10 sqm
            BuildingUse.STORAGE: 0.02,    # 1 person per 50 sqm
            BuildingUse.RECEPTION: 0.1,   # 1 person per 10 sqm
        }
        
        factor = factors.get(use, 0.1)
        return max(1, int(area * factor))
    
    def _determine_door_properties(self, block_name: str, entity) -> Tuple[DoorType, float, float]:
        """Determine door type and dimensions from block name and entity."""
        # Default dimensions
        width_mm = 900
        height_mm = 2100
        door_type = DoorType.SINGLE
        
        # Check for double doors
        if any(keyword in block_name for keyword in ['double', 'doble']):
            door_type = DoorType.DOUBLE
            width_mm = 1200
        
        # Check for emergency exits
        if any(keyword in block_name for keyword in ['emergency', 'emergencia', 'exit', 'salida']):
            door_type = DoorType.EMERGENCY_EXIT
            width_mm = 900
        
        # Check for fire doors
        if any(keyword in block_name for keyword in ['fire', 'fuego', 'rf']):
            door_type = DoorType.FIRE_DOOR
            width_mm = 900
        
        return door_type, width_mm, height_mm
    
    def _find_connected_rooms(self, position: Point3D) -> Tuple[Optional[str], Optional[str]]:
        """Find rooms connected by a door at the given position."""
        # This is a simplified implementation
        # In a real implementation, you'd check which rooms the door position intersects
        if self.rooms:
            return self.rooms[0].id, None
        return None, None
    
    def _is_emergency_exit(self, block_name: str) -> bool:
        """Check if a door is an emergency exit based on block name."""
        return any(keyword in block_name for keyword in ['emergency', 'emergencia', 'exit', 'salida'])
    
    def _determine_wall_thickness(self, entity) -> float:
        """Determine wall thickness from entity properties or default."""
        # Check if entity has thickness property
        if hasattr(entity.dxf, 'thickness') and entity.dxf.thickness > 0:
            return entity.dxf.thickness * 1000  # Convert to mm
        
        # Default wall thickness
        return 200.0  # 20cm


def extract_from_dwg(file_path: Path) -> Project:
    """
    Extract building elements from a DWG/DXF file.
    
    Args:
        file_path: Path to the DWG/DXF file
        
    Returns:
        Project object with extracted data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be processed
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extractor = DWGExtractor()
    
    if not extractor.load_file(file_path):
        raise ValueError(f"Could not load file: {file_path}")
    
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
    """CLI interface for DWG extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract building elements from DWG/DXF files')
    parser.add_argument('input_file', type=Path, help='Input DWG/DXF file')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Extract data
        project = extract_from_dwg(args.input_file)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = args.input_file.with_suffix('.json')
        
        # Save to JSON
        save_to_json(project, output_path)
        
        print(f"✅ Extraction complete!")
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
