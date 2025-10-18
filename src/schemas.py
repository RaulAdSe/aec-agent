"""
Pydantic schemas for AEC Compliance Agent data models.

This module defines the data structures for building information extracted
from CAD/Revit files, with validation and default values.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ProjectMetadata(BaseModel):
    """Metadata for the building project."""
    project_name: str = Field(..., description="Name of the building project")
    project_id: Optional[str] = Field(None, description="Unique project identifier")
    level_name: str = Field(..., description="Building level/floor name")
    level_number: Optional[int] = Field(None, description="Level number (0 = ground floor)")
    extraction_date: datetime = Field(default_factory=datetime.now, description="When data was extracted")
    source_file: Optional[str] = Field(None, description="Source CAD/Revit file path")
    building_use: Optional[str] = Field(None, description="Building use type (commercial, residential, etc.)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Wall(BaseModel):
    """Wall entity from building plans."""
    id: str = Field(..., description="Unique wall identifier")
    start: List[float] = Field(..., description="Start point [x, y] coordinates in meters")
    end: List[float] = Field(..., description="End point [x, y] coordinates in meters")
    length: float = Field(..., description="Wall length in meters")
    thickness: Optional[float] = Field(0.15, description="Wall thickness in meters")
    fire_rating: Optional[str] = Field(None, description="Fire resistance rating (e.g., EI-60, EI-90)")
    material: Optional[str] = Field(None, description="Wall material type")
    is_exterior: bool = Field(False, description="Whether this is an exterior wall")
    
    @field_validator('start', 'end')
    def validate_coordinates(cls, v):
        if len(v) != 2:
            raise ValueError("Coordinates must have exactly 2 values [x, y]")
        return v
    
    @field_validator('length')
    def validate_positive_length(cls, v):
        if v <= 0:
            raise ValueError("Wall length must be positive")
        return v


class Door(BaseModel):
    """Door entity from building plans."""
    id: str = Field(..., description="Unique door identifier")
    position: List[float] = Field(..., description="Door center position [x, y] in meters")
    width: float = Field(..., description="Door width in meters")
    height: Optional[float] = Field(2.1, description="Door height in meters")
    door_type: str = Field("single", description="Door type: single, double, sliding, emergency")
    is_egress: bool = Field(False, description="Whether this is an emergency exit door")
    fire_rating: Optional[str] = Field(None, description="Fire resistance rating")
    opening_direction: Optional[str] = Field(None, description="Opening direction: inward, outward")
    connected_rooms: Optional[List[str]] = Field(default_factory=list, description="IDs of connected rooms")
    
    @field_validator('width')
    def validate_door_width(cls, v):
        if v <= 0:
            raise ValueError("Door width must be positive")
        if v < 0.6:
            raise ValueError("Door width below minimum (0.6m)")
        return v


class Room(BaseModel):
    """Room entity from building plans."""
    id: str = Field(..., description="Unique room identifier")
    name: str = Field(..., description="Room name or number")
    level: str = Field(..., description="Level where room is located")
    boundary: List[List[float]] = Field(..., description="Room boundary polygon points [[x,y], ...]")
    use_type: Optional[str] = Field(None, description="Room use: office, corridor, stairs, etc.")
    area: Optional[float] = Field(None, description="Room area in square meters")
    occupancy_load: Optional[int] = Field(None, description="Maximum occupancy")
    has_emergency_lighting: bool = Field(False, description="Emergency lighting present")
    has_fire_detection: bool = Field(False, description="Fire detection system present")
    ceiling_height: Optional[float] = Field(2.5, description="Ceiling height in meters")
    
    @field_validator('boundary')
    def validate_boundary(cls, v):
        if len(v) < 3:
            raise ValueError("Room boundary must have at least 3 points")
        for point in v:
            if len(point) != 2:
                raise ValueError("Each boundary point must have [x, y] coordinates")
        return v


class FireEquipment(BaseModel):
    """Fire safety equipment from building plans."""
    id: str = Field(..., description="Unique equipment identifier")
    equipment_type: str = Field(..., description="Type: extinguisher, hydrant, alarm, sprinkler, etc.")
    position: List[float] = Field(..., description="Equipment position [x, y] in meters")
    coverage_radius: Optional[float] = Field(None, description="Coverage radius in meters")
    last_inspection: Optional[datetime] = Field(None, description="Last inspection date")
    status: str = Field("active", description="Equipment status: active, inactive, maintenance")
    floor_level: str = Field(..., description="Floor where equipment is located")
    
    @field_validator('position')
    def validate_position(cls, v):
        if len(v) != 2:
            raise ValueError("Position must have exactly 2 coordinates [x, y]")
        return v


class EvacuationRoute(BaseModel):
    """Evacuation route information."""
    id: str = Field(..., description="Route identifier")
    from_room: str = Field(..., description="Starting room ID")
    to_exit: str = Field(..., description="Exit door ID")
    path_nodes: List[List[float]] = Field(..., description="Path coordinates [[x,y], ...]")
    distance: float = Field(..., description="Total route distance in meters")
    is_accessible: bool = Field(True, description="Whether route is accessible")
    has_emergency_lighting: bool = Field(False, description="Emergency lighting along route")
    width_minimum: Optional[float] = Field(None, description="Minimum width along route")


class Sector(BaseModel):
    """Fire compartment/sector from sectorization plans."""
    id: str = Field(..., description="Sector identifier")
    name: str = Field(..., description="Sector name or code")
    boundary: List[List[float]] = Field(..., description="Sector boundary polygon")
    area: Optional[float] = Field(None, description="Sector area in square meters")
    fire_resistance: str = Field(..., description="Fire resistance rating (e.g., EI-60)")
    rooms: List[str] = Field(default_factory=list, description="Room IDs in this sector")
    max_evacuation_distance: Optional[float] = Field(None, description="Maximum evacuation distance in meters")
    compartment_type: Optional[str] = Field(None, description="Type: risk_special, general, protected")


class Level(BaseModel):
    """Building level/floor."""
    name: str = Field(..., description="Level name")
    number: int = Field(..., description="Level number (0 = ground)")
    height: float = Field(0.0, description="Height above ground in meters")
    rooms: List[Room] = Field(default_factory=list, description="Rooms on this level")
    doors: List[Door] = Field(default_factory=list, description="Doors on this level")
    walls: List[Wall] = Field(default_factory=list, description="Walls on this level")
    fire_equipment: List[FireEquipment] = Field(default_factory=list, description="Fire equipment on this level")
    sectors: List[Sector] = Field(default_factory=list, description="Fire sectors on this level")


class Project(BaseModel):
    """Main project schema containing all building data."""
    metadata: ProjectMetadata
    levels: List[Level] = Field(default_factory=list, description="Building levels")
    rooms: List[Room] = Field(default_factory=list, description="All rooms (backward compatibility)")
    doors: List[Door] = Field(default_factory=list, description="All doors (backward compatibility)")
    walls: List[Wall] = Field(default_factory=list, description="All walls (backward compatibility)")
    fire_equipment: List[FireEquipment] = Field(default_factory=list, description="All fire equipment")
    evacuation_routes: List[EvacuationRoute] = Field(default_factory=list, description="Calculated evacuation routes")
    sectors: List[Sector] = Field(default_factory=list, description="Fire compartments/sectors")
    
    def get_level(self, level_name: str) -> Optional[Level]:
        """Get a specific level by name."""
        for level in self.levels:
            if level.name == level_name:
                return level
        return None
    
    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Get a room by its ID."""
        for room in self.rooms:
            if room.id == room_id:
                return room
        return None
    
    def get_door_by_id(self, door_id: str) -> Optional[Door]:
        """Get a door by its ID."""
        for door in self.doors:
            if door.id == door_id:
                return door
        return None
    
    def get_egress_doors(self) -> List[Door]:
        """Get all emergency exit doors."""
        return [door for door in self.doors if door.is_egress]
    
    def get_fire_equipment_by_type(self, equipment_type: str) -> List[FireEquipment]:
        """Get fire equipment by type."""
        return [eq for eq in self.fire_equipment if eq.equipment_type == equipment_type]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Validation helpers
def validate_project_json(json_data: Dict[str, Any]) -> Project:
    """
    Validate JSON data against Project schema.
    
    Args:
        json_data: Dictionary with project data
        
    Returns:
        Validated Project object
        
    Raises:
        ValidationError: If data doesn't match schema
    """
    return Project(**json_data)


def create_example_project() -> Project:
    """Create an example project for testing."""
    metadata = ProjectMetadata(
        project_name="BAUHAUS Leganés",
        level_name="Planta Baja",
        level_number=0,
        building_use="commercial"
    )
    
    room1 = Room(
        id="R001",
        name="Oficina A",
        level="Planta Baja",
        boundary=[[0, 0], [5, 0], [5, 4], [0, 4]],
        use_type="office",
        area=20.0
    )
    
    room2 = Room(
        id="R002", 
        name="Pasillo",
        level="Planta Baja",
        boundary=[[5, 0], [8, 0], [8, 4], [5, 4]],
        use_type="corridor",
        area=12.0,
        has_emergency_lighting=True
    )
    
    door1 = Door(
        id="D001",
        position=[5, 2],
        width=0.9,
        door_type="single",
        connected_rooms=["R001", "R002"]
    )
    
    door2 = Door(
        id="D002",
        position=[8, 2],
        width=1.2,
        door_type="double",
        is_egress=True,
        fire_rating="EI-60"
    )
    
    wall1 = Wall(
        id="W001",
        start=[0, 0],
        end=[5, 0],
        length=5.0,
        fire_rating="EI-90",
        is_exterior=True
    )
    
    equipment1 = FireEquipment(
        id="FE001",
        equipment_type="extinguisher",
        position=[2.5, 3.8],
        coverage_radius=15.0,
        floor_level="Planta Baja"
    )
    
    return Project(
        metadata=metadata,
        rooms=[room1, room2],
        doors=[door1, door2],
        walls=[wall1],
        fire_equipment=[equipment1]
    )