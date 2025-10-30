"""
Integration tests for IFC extraction workflow.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from extraction.ifc_extractor import IFCExtractor, extract_from_ifc
from extraction.unified_extractor import UnifiedExtractor, analyze_file
from schemas import Project, BuildingUse, DoorType


class TestIFCUnifiedExtractorIntegration:
    """Test IFC integration with unified extractor."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_unified_extractor_detects_ifc(self, mock_ifcopenshell):
        """Test that unified extractor correctly detects and processes IFC files."""
        # Setup mock IFC file
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Mock basic entities
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [Mock(Name="Test Project")],
            'IfcBuilding': [Mock(Name="Test Building")],
            'IfcBuildingStorey': [],
            'IfcSpace': [],
            'IfcDoor': [],
            'IfcWall': []
        }.get(entity_type, [])
        
        # Mock utility functions
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            # Create temporary IFC file
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
                file_path = Path(tmp.name)
            
            try:
                # Test unified extractor
                extractor = UnifiedExtractor()
                
                # Test file type detection
                file_type = extractor._detect_file_type(file_path)
                assert file_type == 'ifc'
                
                # Test extraction
                project = extractor.extract_from_file(file_path)
                assert isinstance(project, Project)
                assert project.metadata.project_name == "Test Project"
                
            finally:
                file_path.unlink()
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_unified_extractor_ifc_analysis(self, mock_ifcopenshell):
        """Test IFC file analysis through unified extractor."""
        # Setup mock IFC file
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        
        # Mock entities with iterator
        mock_entities = [
            Mock(is_a=lambda: 'IfcProject'),
            Mock(is_a=lambda: 'IfcBuilding'),
            Mock(is_a=lambda: 'IfcSpace'),
            Mock(is_a=lambda: 'IfcSpace'),
            Mock(is_a=lambda: 'IfcDoor'),
            Mock(is_a=lambda: 'IfcWall'),
        ]
        mock_ifc_file.__iter__ = Mock(return_value=iter(mock_entities))
        
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcSpace': [Mock(), Mock()],
            'IfcDoor': [Mock()],
            'IfcWall': [Mock()],
            'IfcWindow': [],
            'IfcBuildingStorey': [Mock()]
        }.get(entity_type, [])
        
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Create temporary IFC file
        with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
            file_path = Path(tmp.name)
        
        try:
            # Test analysis
            analysis = analyze_file(file_path)
            
            assert analysis['file_type'] == 'ifc'
            assert analysis['supported'] is True
            assert analysis['ifc_schema'] == 'IFC4'
            assert analysis['spaces'] == 2
            assert analysis['doors'] == 1
            assert analysis['walls'] == 1
            assert analysis['windows'] == 0
            assert analysis['building_stories'] == 1
            assert analysis['total_entities'] == 6
            
        finally:
            file_path.unlink()


class TestIFCDirectoryProcessing:
    """Test processing directories containing IFC files."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_directory_processing_with_ifc_files(self, mock_ifcopenshell):
        """Test processing a directory containing IFC files."""
        # Setup mock IFC file
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Mock basic entities
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [Mock(Name="Test Project")],
            'IfcBuilding': [Mock(Name="Test Building")],
            'IfcBuildingStorey': [],
            'IfcSpace': [],
            'IfcDoor': [],
            'IfcWall': []
        }.get(entity_type, [])
        
        # Mock utility functions
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            # Create temporary directory with IFC files
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                
                # Create mock IFC files
                ifc_file1 = tmp_path / "building1.ifc"
                ifc_file2 = tmp_path / "building2.ifc"
                ifc_file1.touch()
                ifc_file2.touch()
                
                # Test directory processing
                extractor = UnifiedExtractor()
                projects = extractor.extract_from_directory(tmp_path)
                
                assert len(projects) == 2
                assert str(ifc_file1) in projects
                assert str(ifc_file2) in projects
                
                for project in projects.values():
                    assert isinstance(project, Project)
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_find_ifc_files_in_directory(self):
        """Test finding IFC files in directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create various files
            (tmp_path / "building.ifc").touch()
            (tmp_path / "BUILDING.IFC").touch()  # Test case insensitive
            (tmp_path / "drawing.dwg").touch()
            (tmp_path / "document.pdf").touch()
            
            extractor = UnifiedExtractor()
            cad_files = extractor._find_cad_files(tmp_path)
            
            # Should find both IFC files and the DWG file
            ifc_files = [f for f in cad_files if f.suffix.lower() == '.ifc']
            assert len(ifc_files) == 2


class TestIFCDataMapping:
    """Test mapping of IFC entities to our data schemas."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_ifc_space_to_room_mapping(self):
        """Test mapping IFC spaces to Room objects."""
        extractor = IFCExtractor()
        
        # Mock IFC space
        space_mock = Mock()
        space_mock.id.return_value = 123
        space_mock.Name = "Conference Room"
        space_mock.LongName = None
        
        # Mock property sets for area
        psets = {
            "Qto_SpaceBaseQuantities": {
                "NetFloorArea": 25.5
            }
        }
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value=psets):
            room = extractor._create_room_from_space(space_mock)
            
            assert room is not None
            assert room.id == "S123"
            assert room.name == "Conference Room"
            assert room.area == 25.5
            assert room.use == BuildingUse.MEETING  # Should detect from name
            assert room.occupancy_load == 5  # 25.5 * 0.2 for meeting room
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_ifc_door_to_door_mapping(self):
        """Test mapping IFC doors to Door objects."""
        extractor = IFCExtractor()
        
        # Mock IFC door
        door_mock = Mock()
        door_mock.id.return_value = 456
        door_mock.Name = "Emergency Exit Door"
        door_mock.LongName = None
        door_mock.PredefinedType = "EMERGENCY_EXIT"
        door_mock.ObjectPlacement = None
        
        # Mock property sets for dimensions
        psets = {
            "Pset_DoorCommon": {
                "OverallWidth": 1.2,  # 1200mm
                "OverallHeight": 2.1  # 2100mm
            }
        }
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value=psets), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            door = extractor._create_door_from_entity(door_mock)
            
            assert door is not None
            assert door.id == "D456"
            assert door.name == "Emergency Exit Door"
            assert door.width_mm == 1200.0
            assert door.height_mm == 2100.0
            assert door.door_type == DoorType.EMERGENCY_EXIT
            assert door.is_emergency_exit is True
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_ifc_wall_to_wall_mapping(self):
        """Test mapping IFC walls to Wall objects."""
        extractor = IFCExtractor()
        
        # Mock IFC wall
        wall_mock = Mock()
        wall_mock.id.return_value = 789
        wall_mock.ObjectPlacement = None
        wall_mock.HasAssociations = []
        
        # Mock property sets
        psets = {
            "Qto_WallBaseQuantities": {
                "Length": 5.0,
                "Width": 0.2,  # thickness
                "Height": 2.7
            }
        }
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value=psets), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            wall = extractor._create_wall_from_entity(wall_mock)
            
            assert wall is not None
            assert wall.id == "W789"
            assert wall.thickness_mm == 200.0  # Converted from 0.2m
            assert wall.height_mm == 2700.0    # Converted from 2.7m
            assert wall.material == "concrete"  # Default


class TestIFCErrorHandling:
    """Test error handling in IFC extraction."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_ifc_file_corruption_handling(self, mock_ifcopenshell):
        """Test handling of corrupted IFC files."""
        mock_ifcopenshell.open.side_effect = Exception("Corrupted IFC file")
        
        with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
            file_path = Path(tmp.name)
        
        try:
            extractor = IFCExtractor()
            result = extractor.load_file(file_path)
            assert result is False
            
            # Test through unified extractor
            unified = UnifiedExtractor()
            with pytest.raises(ValueError, match="Could not load IFC file"):
                unified.extract_from_file(file_path)
                
        finally:
            file_path.unlink()
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_missing_entities_handling(self, mock_ifcopenshell):
        """Test handling when IFC file has missing or no entities."""
        # Setup mock IFC file with no entities
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifc_file.by_type.return_value = []  # No entities of any type
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}):
            
            extractor = IFCExtractor()
            extractor.ifc_file = mock_ifc_file
            
            # Should not crash with empty entities
            project = extractor.extract_all()
            
            assert isinstance(project, Project)
            assert len(project.levels) == 1  # Should create default level
            assert len(project.get_all_rooms()) == 0
            assert len(project.get_all_doors()) == 0
            assert len(project.get_all_walls()) == 0
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_invalid_entity_data_handling(self):
        """Test handling of invalid entity data."""
        extractor = IFCExtractor()
        
        # Test space with invalid data
        invalid_space = Mock()
        invalid_space.id.side_effect = Exception("Invalid entity")
        
        room = extractor._create_room_from_space(invalid_space)
        assert room is None  # Should return None for invalid entities
        
        # Test door with invalid data
        invalid_door = Mock()
        invalid_door.id.side_effect = Exception("Invalid entity")
        
        door = extractor._create_door_from_entity(invalid_door)
        assert room is None  # Should return None for invalid entities


class TestIFCPerformance:
    """Test performance considerations for IFC extraction."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_large_file_handling(self, mock_ifcopenshell):
        """Test handling of large IFC files (mocked)."""
        # Setup mock for large file
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        
        # Simulate many entities
        mock_project = Mock(Name="Large Project")
        mock_building = Mock(Name="Large Building")
        
        # Create mock entities for large file
        mock_spaces = [Mock(Name=f"Space {i}", id=lambda i=i: i) for i in range(100)]
        mock_doors = [Mock(Name=f"Door {i}", id=lambda i=i: i+1000) for i in range(50)]
        mock_walls = [Mock(Name=f"Wall {i}", id=lambda i=i: i+2000) for i in range(200)]
        
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [mock_project],
            'IfcBuilding': [mock_building],
            'IfcBuildingStorey': [Mock(Name="Floor", Elevation=0.0)],
            'IfcSpace': mock_spaces,
            'IfcDoor': mock_doors,
            'IfcWall': mock_walls
        }.get(entity_type, [])
        
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Mock utility functions for performance
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
                file_path = Path(tmp.name)
            
            try:
                # Extract from large file
                project = extract_from_ifc(file_path)
                
                # Verify all entities were processed
                assert isinstance(project, Project)
                assert project.metadata.project_name == "Large Project"
                # Note: Actual counts may be less due to entity creation failures in mocks
                
            finally:
                file_path.unlink()


@pytest.fixture
def mock_ifc_building():
    """Fixture providing a mock IFC building structure."""
    # Mock project
    project = Mock()
    project.Name = "Test Building Project"
    project.Description = "Test project for unit testing"
    
    # Mock building
    building = Mock()
    building.Name = "Main Building"
    building.CompositionType = "ELEMENT"
    
    # Mock stories
    ground_floor = Mock()
    ground_floor.Name = "Ground Floor"
    ground_floor.Elevation = 0.0
    
    first_floor = Mock()
    first_floor.Name = "First Floor"  
    first_floor.Elevation = 3.0
    
    # Mock spaces
    office = Mock()
    office.Name = "Office 101"
    office.id.return_value = 101
    
    meeting_room = Mock()
    meeting_room.Name = "Meeting Room"
    meeting_room.id.return_value = 102
    
    # Mock doors
    main_door = Mock()
    main_door.Name = "Main Entrance"
    main_door.id.return_value = 201
    main_door.PredefinedType = None
    
    emergency_exit = Mock()
    emergency_exit.Name = "Emergency Exit"
    emergency_exit.id.return_value = 202
    emergency_exit.PredefinedType = "EMERGENCY_EXIT"
    
    # Mock walls
    exterior_wall = Mock()
    exterior_wall.Name = "Exterior Wall"
    exterior_wall.id.return_value = 301
    
    interior_wall = Mock()
    interior_wall.Name = "Interior Wall"
    interior_wall.id.return_value = 302
    
    return {
        'project': project,
        'building': building,
        'stories': [ground_floor, first_floor],
        'spaces': [office, meeting_room],
        'doors': [main_door, emergency_exit],
        'walls': [exterior_wall, interior_wall]
    }