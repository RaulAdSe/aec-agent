"""
DWG/DXF extraction module for AEC Compliance Agent.

This module extracts building data from AutoCAD DWG/DXF files,
focusing on fire protection systems and building geometry.
"""

import ezdxf
from ezdxf.document import Drawing
from ezdxf.entities import Insert, Line, LWPolyline, Text, MText, Hatch, Circle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime

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


class DWGExtractor:
    """Extract building data from DWG/DXF files."""
    
    def __init__(self):
        self.rooms: List[Room] = []
        self.doors: List[Door] = []
        self.walls: List[Wall] = []
        self.fire_equipment: List[FireEquipment] = []
        self.sectors: List[Sector] = []
        self.doc: Optional[Drawing] = None
        
    def extract_from_file(self, file_path: Path, project_name: str = None, level_name: str = "Planta Baja") -> Project:
        """
        Extract building data from DWG/DXF file.
        
        Args:
            file_path: Path to DWG/DXF file
            project_name: Name of the project
            level_name: Name of the building level
            
        Returns:
            Project object with extracted data
        """
        logger.info(f"Starting extraction from: {file_path}")
        
        # Load the DXF file
        try:
            self.doc = ezdxf.readfile(str(file_path))
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
        
        # Get model space
        msp = self.doc.modelspace()
        
        # Extract project name from file if not provided
        if not project_name:
            project_name = file_path.stem
        
        # Detect file type from name
        file_type = self._detect_file_type(file_path.name)
        logger.info(f"Detected file type: {file_type}")
        
        # Extract based on file type
        if "PCI" in file_path.name.upper():
            if "EXTINCIÓN" in file_path.name.upper():
                self._extract_fire_equipment(msp, level_name)
            elif "SECTORIZACIÓN" in file_path.name.upper():
                self._extract_sectorization(msp, level_name)
        else:
            # General extraction
            self._extract_walls(msp)
            self._extract_doors(msp)
            self._extract_rooms(msp, level_name)
        
        # Create metadata
        metadata = ProjectMetadata(
            project_name=project_name,
            level_name=level_name,
            extraction_date=datetime.now(),
            source_file=str(file_path),
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
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect the type of CAD file based on filename."""
        filename_upper = filename.upper()
        if "EXTINCIÓN" in filename_upper or "EXTINCION" in filename_upper:
            return "fire_extinguishing"
        elif "SECTORIZACIÓN" in filename_upper or "SECTORIZACION" in filename_upper:
            return "sectorization"
        elif "EVACUACIÓN" in filename_upper or "EVACUACION" in filename_upper:
            return "evacuation"
        else:
            return "general"
    
    def _extract_fire_equipment(self, msp, level_name: str):
        """Extract fire protection equipment from model space."""
        equipment_id = 1
        
        # Look for INSERT blocks (equipment symbols)
        for entity in msp:
            if entity.dxftype() == 'INSERT':
                block_name = entity.dxf.name.upper()
                position = [entity.dxf.insert.x, entity.dxf.insert.y]
                
                equipment_type = self._classify_fire_equipment(block_name)
                if equipment_type:
                    equipment = FireEquipment(
                        id=f"FE{equipment_id:03d}",
                        equipment_type=equipment_type,
                        position=position,
                        floor_level=level_name,
                        status="active"
                    )
                    
                    # Set coverage radius based on type
                    if equipment_type == "extinguisher":
                        equipment.coverage_radius = 15.0
                    elif equipment_type == "hydrant":
                        equipment.coverage_radius = 25.0
                    elif equipment_type == "sprinkler":
                        equipment.coverage_radius = 3.5
                    
                    self.fire_equipment.append(equipment)
                    equipment_id += 1
                    logger.debug(f"Found {equipment_type} at {position}")
            
            # Also look for CIRCLE entities (alternative representation)
            elif entity.dxftype() == 'CIRCLE':
                layer_name = entity.dxf.layer.upper()
                if any(keyword in layer_name for keyword in ['EXTINTOR', 'BIE', 'SPRINKLER', 'ALARMA']):
                    position = [entity.dxf.center.x, entity.dxf.center.y]
                    equipment_type = self._classify_fire_equipment(layer_name)
                    
                    if equipment_type:
                        equipment = FireEquipment(
                            id=f"FE{equipment_id:03d}",
                            equipment_type=equipment_type,
                            position=position,
                            floor_level=level_name,
                            status="active"
                        )
                        self.fire_equipment.append(equipment)
                        equipment_id += 1
    
    def _classify_fire_equipment(self, name: str) -> Optional[str]:
        """Classify fire equipment type from block or layer name."""
        name_upper = name.upper()
        
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
        
        return None
    
    def _extract_sectorization(self, msp, level_name: str):
        """Extract fire compartments/sectors from model space."""
        sector_id = 1
        
        # Look for HATCH entities (sectors are often represented as hatches)
        for entity in msp:
            if entity.dxftype() == 'HATCH':
                # Extract boundary
                boundary_points = []
                try:
                    paths = entity.paths
                    for path in paths:
                        for edge in path.edges:
                            if hasattr(edge, 'start'):
                                boundary_points.append([edge.start.x, edge.start.y])
                except:
                    continue
                
                if len(boundary_points) >= 3:
                    # Determine fire resistance from layer or pattern
                    layer_name = entity.dxf.layer.upper()
                    fire_resistance = self._determine_fire_resistance(layer_name)
                    
                    sector = Sector(
                        id=f"S{sector_id:03d}",
                        name=f"Sector {sector_id}",
                        boundary=boundary_points,
                        fire_resistance=fire_resistance
                    )
                    
                    # Calculate area if possible
                    try:
                        from shapely.geometry import Polygon
                        poly = Polygon(boundary_points)
                        sector.area = poly.area
                    except:
                        pass
                    
                    self.sectors.append(sector)
                    sector_id += 1
                    logger.debug(f"Found sector with {len(boundary_points)} boundary points")
            
            # Also look for closed LWPOLYLINE entities with specific layers
            elif entity.dxftype() == 'LWPOLYLINE':
                layer_name = entity.dxf.layer.upper()
                if any(keyword in layer_name for keyword in ['SECTOR', 'COMPARTMENT', 'FIRE']):
                    if entity.is_closed:
                        boundary_points = [[p[0], p[1]] for p in entity.get_points()]
                        
                        if len(boundary_points) >= 3:
                            fire_resistance = self._determine_fire_resistance(layer_name)
                            
                            sector = Sector(
                                id=f"S{sector_id:03d}",
                                name=f"Sector {sector_id}",
                                boundary=boundary_points,
                                fire_resistance=fire_resistance
                            )
                            self.sectors.append(sector)
                            sector_id += 1
    
    def _determine_fire_resistance(self, layer_name: str) -> str:
        """Determine fire resistance rating from layer name."""
        layer_upper = layer_name.upper()
        
        if "EI-120" in layer_upper or "EI120" in layer_upper:
            return "EI-120"
        elif "EI-90" in layer_upper or "EI90" in layer_upper:
            return "EI-90"
        elif "EI-60" in layer_upper or "EI60" in layer_upper:
            return "EI-60"
        elif "EI-30" in layer_upper or "EI30" in layer_upper:
            return "EI-30"
        else:
            return "EI-60"  # Default
    
    def _extract_walls(self, msp):
        """Extract walls from model space."""
        wall_id = 1
        
        for entity in msp:
            if entity.dxftype() == 'LINE':
                # Check if on wall layer
                layer_name = entity.dxf.layer.upper()
                if any(keyword in layer_name for keyword in ['WALL', 'MURO', 'PARED']):
                    start = [entity.dxf.start.x, entity.dxf.start.y]
                    end = [entity.dxf.end.x, entity.dxf.end.y]
                    
                    # Calculate length
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    
                    if length > 0.1:  # Ignore very small lines
                        wall = Wall(
                            id=f"W{wall_id:03d}",
                            start=start,
                            end=end,
                            length=length,
                            thickness=0.15,  # Default
                            is_exterior='EXT' in layer_name
                        )
                        
                        # Check for fire rating in layer name
                        if 'EI' in layer_name:
                            wall.fire_rating = self._determine_fire_resistance(layer_name)
                        
                        self.walls.append(wall)
                        wall_id += 1
    
    def _extract_doors(self, msp):
        """Extract doors from model space."""
        door_id = 1
        
        for entity in msp:
            if entity.dxftype() == 'INSERT':
                block_name = entity.dxf.name.upper()
                
                # Check if it's a door block
                if any(keyword in block_name for keyword in ['DOOR', 'PUERTA', 'GATE', 'PORTE']):
                    position = [entity.dxf.insert.x, entity.dxf.insert.y]
                    
                    # Try to get width from block attributes or scale
                    width = 0.9  # Default single door
                    if entity.dxf.xscale:
                        width = abs(entity.dxf.xscale) * 0.9
                    
                    # Check for width in attributes
                    if hasattr(entity, 'attribs'):
                        for attrib in entity.attribs:
                            if attrib.dxf.tag.upper() in ['WIDTH', 'ANCHO', 'LARGO']:
                                try:
                                    width = float(attrib.dxf.text)
                                    break
                                except:
                                    pass
                    
                    # Determine door type
                    door_type = "single"
                    is_egress = False
                    fire_rating = None
                    
                    if 'DOUBLE' in block_name or 'DOBLE' in block_name:
                        door_type = "double"
                        width = max(width, 1.2)  # Ensure minimum width for double doors
                    elif 'EMERGENCY' in block_name or 'EMERGENCIA' in block_name or 'SALIDA' in block_name:
                        is_egress = True
                        door_type = "emergency"
                        width = max(width, 0.8)  # Ensure minimum egress width
                    elif 'SLIDING' in block_name or 'CORRED' in block_name:
                        door_type = "sliding"
                    elif 'FIRE' in block_name or 'FUEGO' in block_name:
                        fire_rating = "EI-60"  # Default fire rating
                    
                    # Check layer for additional properties
                    layer_name = entity.dxf.layer.upper()
                    if 'EI' in layer_name:
                        fire_rating = self._determine_fire_resistance(layer_name)
                    if 'EGRESS' in layer_name or 'SALIDA' in layer_name:
                        is_egress = True
                    
                    door = Door(
                        id=f"D{door_id:03d}",
                        position=position,
                        width=width,
                        door_type=door_type,
                        is_egress=is_egress,
                        fire_rating=fire_rating
                    )
                    
                    self.doors.append(door)
                    door_id += 1
                    logger.debug(f"Found {door_type} door at {position} (width: {width}m)")
            
            # Also check for door symbols represented as lines or arcs
            elif entity.dxftype() == 'LINE':
                layer_name = entity.dxf.layer.upper()
                if any(keyword in layer_name for keyword in ['DOOR', 'PUERTA', 'PORTE']):
                    # Calculate door position as midpoint
                    start = [entity.dxf.start.x, entity.dxf.start.y]
                    end = [entity.dxf.end.x, entity.dxf.end.y]
                    position = [(start[0] + end[0])/2, (start[1] + end[1])/2]
                    
                    # Estimate width from line length
                    length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
                    width = max(0.8, min(2.0, length))  # Reasonable door width range
                    
                    door = Door(
                        id=f"D{door_id:03d}",
                        position=position,
                        width=width,
                        door_type="single",
                        is_egress='EGRESS' in layer_name or 'SALIDA' in layer_name
                    )
                    
                    self.doors.append(door)
                    door_id += 1
                    logger.debug(f"Found door line at {position} (width: {width}m)")
    
    def _extract_rooms(self, msp, level_name: str):
        """Extract rooms from model space."""
        room_id = 1
        
        # Look for closed polylines representing rooms
        for entity in msp:
            if entity.dxftype() == 'LWPOLYLINE':
                if entity.is_closed:
                    layer_name = entity.dxf.layer.upper()
                    
                    # Check if it's a room layer
                    if any(keyword in layer_name for keyword in ['ROOM', 'SPACE', 'HABITACION', 'SALA']):
                        boundary = [[p[0], p[1]] for p in entity.get_points()]
                        
                        if len(boundary) >= 3:
                            # Look for room name from nearby text
                            room_name = self._find_room_name(msp, boundary) or f"Room {room_id}"
                            
                            room = Room(
                                id=f"R{room_id:03d}",
                                name=room_name,
                                level=level_name,
                                boundary=boundary
                            )
                            
                            # Calculate area
                            try:
                                from shapely.geometry import Polygon
                                poly = Polygon(boundary)
                                room.area = poly.area
                            except:
                                pass
                            
                            # Determine use type from name
                            room.use_type = self._determine_room_type(room_name)
                            
                            self.rooms.append(room)
                            room_id += 1
    
    def _find_room_name(self, msp, boundary: List[List[float]]) -> Optional[str]:
        """Find room name from text entities within boundary."""
        # Calculate boundary center
        if not boundary:
            return None
        
        center_x = sum(p[0] for p in boundary) / len(boundary)
        center_y = sum(p[1] for p in boundary) / len(boundary)
        
        # Look for text entities near center
        for entity in msp:
            if entity.dxftype() in ['TEXT', 'MTEXT']:
                text_pos = None
                if entity.dxftype() == 'TEXT':
                    text_pos = entity.dxf.insert
                    text_content = entity.dxf.text
                else:  # MTEXT
                    text_pos = entity.dxf.insert
                    text_content = entity.text
                
                if text_pos:
                    # Check if text is near room center
                    dist = ((text_pos.x - center_x)**2 + (text_pos.y - center_y)**2)**0.5
                    if dist < 5.0:  # Within 5 meters of center
                        return text_content
        
        return None
    
    def _determine_room_type(self, room_name: str) -> str:
        """Determine room use type from name."""
        name_upper = room_name.upper()
        
        if any(keyword in name_upper for keyword in ['OFFICE', 'OFICINA', 'DESPACHO']):
            return "office"
        elif any(keyword in name_upper for keyword in ['CORRIDOR', 'PASILLO', 'HALL']):
            return "corridor"
        elif any(keyword in name_upper for keyword in ['STAIR', 'ESCALERA']):
            return "stairs"
        elif any(keyword in name_upper for keyword in ['STORAGE', 'ALMACEN', 'ARCHIVO']):
            return "storage"
        elif any(keyword in name_upper for keyword in ['BATHROOM', 'BAÑO', 'ASEO', 'WC']):
            return "bathroom"
        elif any(keyword in name_upper for keyword in ['MEETING', 'REUNION', 'SALA']):
            return "meeting_room"
        elif any(keyword in name_upper for keyword in ['LOBBY', 'RECEPCION', 'ENTRADA']):
            return "lobby"
        else:
            return "general"


def extract_and_save(file_path: Path, output_path: Optional[Path] = None, 
                     project_name: Optional[str] = None) -> Project:
    """
    Extract data from DWG/DXF file and save to JSON.
    
    Args:
        file_path: Path to DWG/DXF file
        output_path: Optional path for JSON output
        project_name: Optional project name
        
    Returns:
        Extracted Project object
    """
    extractor = DWGExtractor()
    project = extractor.extract_from_file(file_path, project_name)
    
    # Save to JSON if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(project.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved extracted data to: {output_path}")
    
    return project