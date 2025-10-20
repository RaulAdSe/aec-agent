"""
Pydantic schemas for AEC Compliance Agent data models.

This module defines the data structures used throughout the application
for representing building elements, project metadata, and compliance checks.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class BuildingUse(str, Enum):
    """Building use types for compliance classification."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    OFFICE = "office"
    RETAIL = "retail"
    INDUSTRIAL = "industrial"
    EDUCATIONAL = "educational"
    HEALTHCARE = "healthcare"
    ASSEMBLY = "assembly"
    STORAGE = "storage"
    RECEPTION = "reception"
    MEETING = "meeting"
    RESTROOM = "restroom"
    CORRIDOR = "corridor"
    STAIR = "stair"
    ELEVATOR = "elevator"
    EXIT = "exit"


class DoorType(str, Enum):
    """Door types for compliance checking."""
    SINGLE = "single"
    DOUBLE = "double"
    SLIDING = "sliding"
    REVOLVING = "revolving"
    FIRE_DOOR = "fire_door"
    EMERGENCY_EXIT = "emergency_exit"


class FireRating(str, Enum):
    """Fire resistance ratings."""
    NO_RATING = "no_rating"
    RF_30 = "RF_30"  # 30 minutes
    RF_60 = "RF_60"  # 60 minutes
    RF_90 = "RF_90"  # 90 minutes
    RF_120 = "RF_120"  # 120 minutes


class Point2D(BaseModel):
    """2D point with x, y coordinates."""
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")


class Point3D(BaseModel):
    """3D point with x, y, z coordinates."""
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
    z: float = Field(..., description="Z coordinate in meters")


class Boundary(BaseModel):
    """Room boundary as a list of 2D points."""
    points: List[Point2D] = Field(..., min_length=3, description="List of boundary points")
    
    @field_validator('points')
    @classmethod
    def validate_closed_polygon(cls, v):
        """Ensure the polygon is closed (first point == last point)."""
        if len(v) < 3:
            raise ValueError("Boundary must have at least 3 points")
        
        # Check if polygon is closed
        first_point = v[0]
        last_point = v[-1]
        
        if abs(first_point.x - last_point.x) > 1e-6 or abs(first_point.y - last_point.y) > 1e-6:
            # Auto-close the polygon
            v.append(Point2D(x=first_point.x, y=first_point.y))
        
        return v


class Room(BaseModel):
    """Room representation with geometric and functional properties."""
    id: str = Field(..., description="Unique room identifier")
    name: str = Field(..., description="Room name")
    area: float = Field(..., gt=0, description="Room area in square meters")
    use: BuildingUse = Field(..., description="Room use type")
    boundary: Optional[Boundary] = Field(None, description="Room boundary polygon")
    level: str = Field(..., description="Building level/floor")
    occupancy_load: Optional[int] = Field(None, ge=0, description="Maximum occupancy load")
    fire_rating: Optional[FireRating] = Field(None, description="Required fire rating")
    
    class Config:
        use_enum_values = True


class Door(BaseModel):
    """Door representation with dimensional and functional properties."""
    id: str = Field(..., description="Unique door identifier")
    name: Optional[str] = Field(None, description="Door name")
    width_mm: float = Field(..., gt=0, description="Door width in millimeters")
    height_mm: float = Field(..., gt=0, description="Door height in millimeters")
    door_type: DoorType = Field(..., description="Type of door")
    fire_rating: Optional[FireRating] = Field(None, description="Fire resistance rating")
    position: Point3D = Field(..., description="Door position in 3D space")
    from_room: Optional[str] = Field(None, description="Source room ID")
    to_room: Optional[str] = Field(None, description="Destination room ID")
    is_emergency_exit: bool = Field(False, description="Is this an emergency exit door")
    is_accessible: bool = Field(True, description="Is door accessible (ADA compliance)")
    
    class Config:
        use_enum_values = True


class Wall(BaseModel):
    """Wall representation with geometric and material properties."""
    id: str = Field(..., description="Unique wall identifier")
    start_point: Point3D = Field(..., description="Wall start point")
    end_point: Point3D = Field(..., description="Wall end point")
    thickness_mm: float = Field(..., gt=0, description="Wall thickness in millimeters")
    height_mm: float = Field(..., gt=0, description="Wall height in millimeters")
    fire_rating: Optional[FireRating] = Field(None, description="Fire resistance rating")
    material: Optional[str] = Field(None, description="Wall material")
    
    class Config:
        use_enum_values = True


class Level(BaseModel):
    """Building level/floor representation."""
    name: str = Field(..., description="Level name")
    elevation: float = Field(..., description="Level elevation in meters")
    rooms: List[Room] = Field(default_factory=list, description="Rooms on this level")
    doors: List[Door] = Field(default_factory=list, description="Doors on this level")
    walls: List[Wall] = Field(default_factory=list, description="Walls on this level")


class ProjectMetadata(BaseModel):
    """Project metadata and general information."""
    project_name: str = Field(..., description="Project name")
    file_name: str = Field(..., description="Source file name")
    building_type: str = Field(..., description="Type of building")
    total_area: Optional[float] = Field(None, ge=0, description="Total building area")
    number_of_levels: Optional[int] = Field(None, ge=1, description="Number of building levels")
    created_date: Optional[str] = Field(None, description="Creation date")
    modified_date: Optional[str] = Field(None, description="Last modification date")


class Project(BaseModel):
    """Complete project representation."""
    metadata: ProjectMetadata = Field(..., description="Project metadata")
    levels: List[Level] = Field(..., min_length=1, description="Building levels")
    
    def get_all_rooms(self) -> List[Room]:
        """Get all rooms from all levels."""
        rooms = []
        for level in self.levels:
            rooms.extend(level.rooms)
        return rooms
    
    def get_all_doors(self) -> List[Door]:
        """Get all doors from all levels."""
        doors = []
        for level in self.levels:
            doors.extend(level.doors)
        return doors
    
    def get_all_walls(self) -> List[Wall]:
        """Get all walls from all levels."""
        walls = []
        for level in self.levels:
            walls.extend(level.walls)
        return walls
    
    def get_room_by_id(self, room_id: str) -> Optional[Room]:
        """Get room by ID."""
        for level in self.levels:
            for room in level.rooms:
                if room.id == room_id:
                    return room
        return None
    
    def get_door_by_id(self, door_id: str) -> Optional[Door]:
        """Get door by ID."""
        for level in self.levels:
            for door in level.doors:
                if door.id == door_id:
                    return door
        return None


class ComplianceCheck(BaseModel):
    """Result of a compliance check."""
    check_id: str = Field(..., description="Unique check identifier")
    check_type: str = Field(..., description="Type of compliance check")
    element_id: str = Field(..., description="ID of the element being checked")
    element_type: str = Field(..., description="Type of element (room, door, wall)")
    is_compliant: bool = Field(..., description="Whether the element is compliant")
    requirement: str = Field(..., description="Regulation requirement being checked")
    actual_value: Optional[float] = Field(None, description="Actual measured value")
    required_value: Optional[float] = Field(None, description="Required value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    message: str = Field(..., description="Human-readable result message")
    regulation_reference: Optional[str] = Field(None, description="Reference to regulation")
    severity: str = Field("warning", description="Severity level: info, warning, error")
    
    class Config:
        use_enum_values = True


class ComplianceReport(BaseModel):
    """Complete compliance verification report."""
    project_id: str = Field(..., description="Project identifier")
    report_date: str = Field(..., description="Report generation date")
    total_checks: int = Field(..., ge=0, description="Total number of checks performed")
    compliant_checks: int = Field(..., ge=0, description="Number of compliant checks")
    non_compliant_checks: int = Field(..., ge=0, description="Number of non-compliant checks")
    compliance_percentage: float = Field(..., ge=0, le=100, description="Overall compliance percentage")
    checks: List[ComplianceCheck] = Field(..., description="List of all compliance checks")
    summary: str = Field(..., description="Executive summary of compliance status")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    
    @field_validator('compliance_percentage')
    @classmethod
    def calculate_compliance_percentage(cls, v, info):
        """Calculate compliance percentage if not provided."""
        if info.data and 'total_checks' in info.data and 'compliant_checks' in info.data:
            total = info.data['total_checks']
            compliant = info.data['compliant_checks']
            if total > 0:
                return round((compliant / total) * 100, 2)
        return v
