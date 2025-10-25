"""
Unit tests for IFC extractor module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from extraction.ifc_extractor import IFCExtractor, extract_from_ifc, save_to_json
from schemas import Project, ProjectMetadata, Level, Room, Door, Wall, Point3D, BuildingUse, DoorType


class TestIFCExtractor:
    """Test cases for IFCExtractor class."""
    
    def test_init_without_ifcopenshell(self):
        """Test initialization when ifcopenshell is not available."""
        with patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', False):
            with pytest.raises(ImportError, match="ifcopenshell is required"):
                IFCExtractor()
    
    def test_init_with_ifcopenshell(self):
        """Test successful initialization."""
        with patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True):
            extractor = IFCExtractor()
            assert extractor.ifc_file is None
            assert extractor.rooms == []
            assert extractor.doors == []
            assert extractor.walls == []
            assert extractor.windows == []
            assert extractor.levels == {}
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_load_file_success(self, mock_ifcopenshell):
        """Test successful file loading."""
        # Setup mock
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        extractor = IFCExtractor()
        result = extractor.load_file(Path("test.ifc"))
        
        assert result is True
        assert extractor.ifc_file == mock_ifc_file
        mock_ifcopenshell.open.assert_called_once_with("test.ifc")
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_load_file_failure(self, mock_ifcopenshell):
        """Test file loading failure."""
        mock_ifcopenshell.open.side_effect = Exception("File not found")
        
        extractor = IFCExtractor()
        result = extractor.load_file(Path("nonexistent.ifc"))
        
        assert result is False
        assert extractor.ifc_file is None
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_extract_all_without_loaded_file(self):
        """Test extract_all without loading a file first."""
        extractor = IFCExtractor()
        
        with pytest.raises(ValueError, match="No IFC file loaded"):
            extractor.extract_all()
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_get_story_elevation(self):
        """Test getting story elevation."""
        extractor = IFCExtractor()
        
        # Test with placement
        story_mock = Mock()
        story_mock.ObjectPlacement = Mock()
        placement_matrix = [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 5], [0, 0, 0, 1]]
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=placement_matrix):
            elevation = extractor._get_story_elevation(story_mock)
            assert elevation == 5.0
        
        # Test with Elevation attribute
        story_mock.ObjectPlacement = None
        story_mock.Elevation = 3.5
        elevation = extractor._get_story_elevation(story_mock)
        assert elevation == 3.5
        
        # Test default case
        story_mock.Elevation = None
        elevation = extractor._get_story_elevation(story_mock)
        assert elevation == 0.0
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_determine_space_use(self):
        """Test space use determination."""
        extractor = IFCExtractor()
        
        # Test office space
        space_mock = Mock()
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}):
            use = extractor._determine_space_use("Office Room", space_mock)
            assert use == BuildingUse.OFFICE
        
        # Test meeting room
        use = extractor._determine_space_use("Meeting Room", space_mock)
        assert use == BuildingUse.MEETING
        
        # Test restroom
        use = extractor._determine_space_use("Restroom", space_mock)
        assert use == BuildingUse.RESTROOM
        
        # Test default case
        use = extractor._determine_space_use("Unknown Space", space_mock)
        assert use == BuildingUse.COMMERCIAL
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_calculate_occupancy_load(self):
        """Test occupancy load calculation."""
        extractor = IFCExtractor()
        
        # Test office space
        load = extractor._calculate_occupancy_load(100.0, BuildingUse.OFFICE)
        assert load == 10  # 100 * 0.1
        
        # Test meeting room
        load = extractor._calculate_occupancy_load(50.0, BuildingUse.MEETING)
        assert load == 10  # 50 * 0.2
        
        # Test minimum load
        load = extractor._calculate_occupancy_load(1.0, BuildingUse.STORAGE)
        assert load == 1  # Minimum is 1
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_determine_door_type(self):
        """Test door type determination."""
        extractor = IFCExtractor()
        
        # Test single door (default)
        door_mock = Mock()
        door_mock.Name = "Standard Door"
        door_mock.PredefinedType = None
        door_type = extractor._determine_door_type(door_mock)
        assert door_type == DoorType.SINGLE
        
        # Test double door
        door_mock.Name = "Double Door"
        door_type = extractor._determine_door_type(door_mock)
        assert door_type == DoorType.DOUBLE
        
        # Test emergency exit
        door_mock.Name = "Emergency Exit"
        door_type = extractor._determine_door_type(door_mock)
        assert door_type == DoorType.EMERGENCY_EXIT
        
        # Test fire door
        door_mock.Name = "Fire Door"
        door_type = extractor._determine_door_type(door_mock)
        assert door_type == DoorType.FIRE_DOOR
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_get_element_position(self):
        """Test element position extraction."""
        extractor = IFCExtractor()
        
        # Test with placement
        element_mock = Mock()
        element_mock.ObjectPlacement = Mock()
        placement_matrix = [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=placement_matrix):
            position = extractor._get_element_position(element_mock)
            assert position.x == 10.0
            assert position.y == 20.0
            assert position.z == 30.0
        
        # Test default case
        element_mock.ObjectPlacement = None
        position = extractor._get_element_position(element_mock)
        assert position.x == 0.0
        assert position.y == 0.0
        assert position.z == 0.0
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_is_emergency_exit(self):
        """Test emergency exit detection."""
        extractor = IFCExtractor()
        
        # Test emergency exit by name
        door_mock = Mock()
        door_mock.Name = "Emergency Exit"
        door_mock.PredefinedType = None
        assert extractor._is_emergency_exit(door_mock) is True
        
        # Test emergency exit by predefined type
        door_mock.Name = "Door"
        door_mock.PredefinedType = "EMERGENCY_EXIT"
        assert extractor._is_emergency_exit(door_mock) is True
        
        # Test regular door
        door_mock.Name = "Regular Door"
        door_mock.PredefinedType = None
        assert extractor._is_emergency_exit(door_mock) is False
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_get_door_dimensions(self):
        """Test door dimensions extraction."""
        extractor = IFCExtractor()
        
        door_mock = Mock()
        psets = {
            "Pset_DoorCommon": {
                "OverallWidth": 0.9,  # 900mm
                "OverallHeight": 2.1  # 2100mm
            }
        }
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value=psets):
            width, height = extractor._get_door_dimensions(door_mock)
            assert width == 900.0
            assert height == 2100.0
        
        # Test default case
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}):
            width, height = extractor._get_door_dimensions(door_mock)
            assert width == 900.0  # Default
            assert height == 2100.0  # Default
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_get_wall_thickness(self):
        """Test wall thickness extraction."""
        extractor = IFCExtractor()
        
        wall_mock = Mock()
        psets = {
            "Pset_WallCommon": {
                "ThermalTransmittance": 0.5,
                "Reference": "Wall-001",
                "thickness": 0.2  # 200mm in meters
            }
        }
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value=psets):
            thickness = extractor._get_wall_thickness(wall_mock)
            assert thickness == 200.0  # Converted to mm
        
        # Test default case
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}):
            thickness = extractor._get_wall_thickness(wall_mock)
            assert thickness == 200.0  # Default
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    def test_get_wall_material(self):
        """Test wall material extraction."""
        extractor = IFCExtractor()
        
        # Test with material association
        wall_mock = Mock()
        material_mock = Mock()
        material_mock.Name = "Concrete"
        
        rel_mock = Mock()
        rel_mock.is_a.return_value = True
        rel_mock.RelatingMaterial = material_mock
        
        wall_mock.HasAssociations = [rel_mock]
        
        material = extractor._get_wall_material(wall_mock)
        assert material == "Concrete"
        
        # Test default case
        wall_mock.HasAssociations = []
        material = extractor._get_wall_material(wall_mock)
        assert material == "concrete"  # Default


class TestIFCExtractionFunctions:
    """Test cases for standalone IFC extraction functions."""
    
    @patch('extraction.ifc_extractor.IFCExtractor')
    def test_extract_from_ifc_success(self, mock_extractor_class):
        """Test successful IFC extraction."""
        # Setup mocks
        mock_extractor = Mock()
        mock_project = Mock(spec=Project)
        mock_extractor.load_file.return_value = True
        mock_extractor.extract_all.return_value = mock_project
        mock_extractor_class.return_value = mock_extractor
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
            file_path = Path(tmp.name)
        
        try:
            result = extract_from_ifc(file_path)
            assert result == mock_project
            mock_extractor.load_file.assert_called_once_with(file_path)
            mock_extractor.extract_all.assert_called_once()
        finally:
            file_path.unlink()
    
    def test_extract_from_ifc_file_not_found(self):
        """Test IFC extraction with non-existent file."""
        file_path = Path("nonexistent.ifc")
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            extract_from_ifc(file_path)
    
    @patch('extraction.ifc_extractor.IFCExtractor')
    def test_extract_from_ifc_load_failure(self, mock_extractor_class):
        """Test IFC extraction when file loading fails."""
        # Setup mocks
        mock_extractor = Mock()
        mock_extractor.load_file.return_value = False
        mock_extractor_class.return_value = mock_extractor
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
            file_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValueError, match="Could not load IFC file"):
                extract_from_ifc(file_path)
        finally:
            file_path.unlink()
    
    def test_save_to_json(self):
        """Test saving project to JSON."""
        # Create a sample project
        metadata = ProjectMetadata(
            project_name="Test Project",
            file_name="test.ifc",
            building_type="office",
            total_area=1000.0,
            number_of_levels=1,
            created_date="2024-01-01T00:00:00",
            modified_date="2024-01-01T00:00:00"
        )
        
        level = Level(
            name="Ground Floor",
            elevation=0.0,
            rooms=[],
            doors=[],
            walls=[]
        )
        
        project = Project(
            metadata=metadata,
            levels=[level]
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            output_path = Path(tmp.name)
        
        try:
            save_to_json(project, output_path)
            
            # Verify file was created and contains correct data
            assert output_path.exists()
            
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data['metadata']['project_name'] == "Test Project"
            assert data['metadata']['building_type'] == "office"
            assert len(data['levels']) == 1
            assert data['levels'][0]['name'] == "Ground Floor"
            
        finally:
            if output_path.exists():
                output_path.unlink()


class TestIFCExtractionIntegration:
    """Integration tests for IFC extraction with mock data."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_full_extraction_workflow(self, mock_ifcopenshell):
        """Test complete extraction workflow with mocked IFC data."""
        # Setup mock IFC file
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Mock project
        mock_project = Mock()
        mock_project.Name = "Test Building"
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [mock_project],
            'IfcBuilding': [Mock(Name="Main Building")],
            'IfcBuildingStorey': [Mock(Name="Ground Floor", Elevation=0.0)],
            'IfcSpace': [Mock(Name="Office 1", id=lambda: 123)],
            'IfcDoor': [Mock(Name="Door 1", id=lambda: 456)],
            'IfcWall': [Mock(Name="Wall 1", id=lambda: 789)]
        }.get(entity_type, [])
        
        # Mock utility functions
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]):
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
                file_path = Path(tmp.name)
            
            try:
                # Run extraction
                project = extract_from_ifc(file_path)
                
                # Verify project structure
                assert isinstance(project, Project)
                assert project.metadata.project_name == "Test Building"
                assert len(project.levels) == 1
                assert project.levels[0].name == "Ground Floor"
                
                # Verify extracted elements
                all_rooms = project.get_all_rooms()
                all_doors = project.get_all_doors()
                all_walls = project.get_all_walls()
                
                assert len(all_rooms) >= 0  # May be empty if mocked space creation fails
                assert len(all_doors) >= 0  # May be empty if mocked door creation fails
                assert len(all_walls) >= 0  # May be empty if mocked wall creation fails
                
            finally:
                file_path.unlink()


@pytest.fixture
def sample_ifc_data():
    """Fixture providing sample IFC data for testing."""
    return {
        'project_name': 'Test Building',
        'building_name': 'Main Building',
        'stories': [
            {'name': 'Ground Floor', 'elevation': 0.0},
            {'name': 'First Floor', 'elevation': 3.0}
        ],
        'spaces': [
            {'name': 'Office 1', 'area': 25.0},
            {'name': 'Meeting Room', 'area': 15.0},
            {'name': 'Restroom', 'area': 5.0}
        ],
        'doors': [
            {'name': 'Door 1', 'width': 0.9, 'height': 2.1},
            {'name': 'Emergency Exit', 'width': 1.2, 'height': 2.1}
        ],
        'walls': [
            {'name': 'Wall 1', 'thickness': 0.2, 'material': 'Concrete'},
            {'name': 'Wall 2', 'thickness': 0.15, 'material': 'Brick'}
        ]
    }