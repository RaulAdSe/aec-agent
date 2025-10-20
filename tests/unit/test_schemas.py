"""
Unit tests for schemas module.
"""
import pytest
from datetime import datetime
from src.schemas import (
    Project, ProjectMetadata, Room, Door, Wall, Level
)


class TestProjectMetadata:
    """Test ProjectMetadata schema."""
    
    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = ProjectMetadata(
            project_name="Test Project",
            level_name="Ground Floor",
            extraction_date=datetime.now(),
            source_file="test.dwg",
            building_use="commercial"
        )
        
        assert metadata.project_name == "Test Project"
        assert metadata.level_name == "Ground Floor"
        assert metadata.building_use == "commercial"
    
    def test_metadata_defaults(self):
        """Test metadata with minimal fields."""
        metadata = ProjectMetadata(
            project_name="Test",
            level_name="P1",
            extraction_date=datetime.now(),
            source_file="test.dwg"
        )
        
        assert metadata.project_name == "Test"
        assert metadata.building_use is None  # No default value


class TestRoom:
    """Test Room schema."""
    
    def test_valid_room(self):
        """Test creating valid room."""
        room = Room(
            id="ROOM_001",
            name="Office",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 5], [0, 5]]
        )
        
        assert room.id == "ROOM_001"
        assert room.name == "Office"
        assert len(room.boundary) == 4
    
    def test_room_with_optional_fields(self):
        """Test room with use_type."""
        room = Room(
            id="ROOM_001",
            name="Office",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 5], [0, 5]],
            use_type="office"
        )
        
        assert room.use_type == "office"
    
    def test_room_with_area(self):
        """Test room with calculated area."""
        room = Room(
            id="ROOM_001",
            name="Office",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 5], [0, 5]],
            area=50.0
        )
        
        assert room.area == 50.0


class TestDoor:
    """Test Door schema."""
    
    def test_valid_door(self):
        """Test creating valid door."""
        door = Door(
            id="DOOR_001",
            position=[5, 0],
            width=0.90,
            door_type="single"
        )
        
        assert door.width == 0.90
        assert door.door_type == "single"
        assert door.position == [5, 0]
    
    def test_door_with_fire_rating(self):
        """Test door with fire rating."""
        door = Door(
            id="DOOR_001",
            position=[0, 0],
            width=0.80,
            door_type="single",
            is_egress=True,
            fire_rating="EI-30"
        )
        
        assert door.is_egress is True
        assert door.fire_rating == "EI-30"
    
    def test_door_defaults(self):
        """Test door with default values."""
        door = Door(
            id="DOOR_001",
            position=[0, 0],
            width=0.90
        )
        
        assert door.door_type == "single"  # Default
        assert door.is_egress is False  # Default


class TestWall:
    """Test Wall schema."""
    
    def test_valid_wall(self):
        """Test creating valid wall."""
        wall = Wall(
            id="WALL_001",
            start=[0, 0],
            end=[10, 0],
            length=10.0,
            thickness=0.15
        )
        
        assert wall.length == 10.0
        assert wall.thickness == 0.15
        assert wall.start == [0, 0]
        assert wall.end == [10, 0]
    
    def test_wall_with_fire_rating(self):
        """Test wall with fire rating."""
        wall = Wall(
            id="WALL_001",
            start=[0, 0],
            end=[10, 0],
            length=10.0,
            thickness=0.15,
            is_exterior=True,
            fire_rating="EI-90"
        )
        
        assert wall.is_exterior is True
        assert wall.fire_rating == "EI-90"


# FireEquipment and Sector classes removed as they don't exist in current schemas


class TestLevel:
    """Test Level schema."""
    
    def test_valid_level(self):
        """Test creating valid level."""
        level = Level(
            name="Ground Floor",
            number=0,
            height=0.0
        )
        
        assert level.name == "Ground Floor"
        assert level.number == 0
        assert level.height == 0.0


class TestProject:
    """Test complete Project schema."""
    
    def test_valid_project(self):
        """Test creating valid project."""
        metadata = ProjectMetadata(
            project_name="Test",
            level_name="P1",
            extraction_date=datetime.now(),
            source_file="test.dwg"
        )
        
        level = Level(name="P1", number=0, height=0.0)
        
        room = Room(
            id="ROOM_001",
            name="Office",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 5], [0, 5]]
        )
        
        door = Door(
            id="DOOR_001",
            position=[5, 0],
            width=0.90
        )
        
        project = Project(
            metadata=metadata,
            levels=[level],
            rooms=[room],
            doors=[door],
            walls=[],
            fire_equipment=[],
            sectors=[]
        )
        
        assert len(project.rooms) == 1
        assert len(project.doors) == 1
        assert len(project.levels) == 1
    
    def test_empty_project(self):
        """Test project with no rooms/doors."""
        metadata = ProjectMetadata(
            project_name="Empty",
            level_name="P1",
            extraction_date=datetime.now(),
            source_file="empty.dwg"
        )
        
        project = Project(
            metadata=metadata,
            levels=[],
            rooms=[],
            doors=[],
            walls=[],
            fire_equipment=[],
            sectors=[]
        )
        
        assert len(project.rooms) == 0
        assert len(project.doors) == 0
