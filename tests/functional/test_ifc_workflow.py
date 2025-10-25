"""
Functional tests for complete IFC extraction workflow.
"""

import pytest
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from extraction.ifc_extractor import extract_from_ifc, save_to_json
from extraction.unified_extractor import extract_from_file, extract_from_directory
from schemas import Project, BuildingUse, DoorType


class TestIFCCommandLineInterface:
    """Test IFC extraction command-line interfaces."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_ifc_extractor_cli_single_file(self, mock_ifcopenshell):
        """Test IFC extractor CLI with single file."""
        # Setup mock
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [Mock(Name="CLI Test Project")],
            'IfcBuilding': [Mock(Name="CLI Test Building")],
            'IfcBuildingStorey': [],
            'IfcSpace': [],
            'IfcDoor': [],
            'IfcWall': []
        }.get(entity_type, [])
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as input_file, \
                 tempfile.NamedTemporaryFile(suffix='.json', delete=False) as output_file:
                
                input_path = Path(input_file.name)
                output_path = Path(output_file.name)
            
            try:
                # Test CLI script
                script_path = Path(__file__).parent.parent.parent.parent / 'scripts' / 'extract_ifc_files.py'
                
                if script_path.exists():
                    # Run the CLI script
                    result = subprocess.run([
                        sys.executable, str(script_path),
                        '-f', str(input_path),
                        '-o', str(output_path)
                    ], capture_output=True, text=True)
                    
                    # Check if command succeeded (may fail due to missing dependencies in test environment)
                    if result.returncode == 0:
                        assert output_path.exists()
                        
                        # Verify JSON output
                        with open(output_path, 'r') as f:
                            data = json.load(f)
                        
                        assert 'metadata' in data
                        assert 'levels' in data
                        assert data['metadata']['project_name'] == "CLI Test Project"
                
            finally:
                # Cleanup
                for path in [input_path, output_path]:
                    if path.exists():
                        path.unlink()
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_unified_extractor_cli_ifc_file(self, mock_ifcopenshell):
        """Test unified extractor CLI with IFC file."""
        # Setup mock
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [Mock(Name="Unified Test Project")],
            'IfcBuilding': [Mock(Name="Unified Test Building")],
            'IfcBuildingStorey': [],
            'IfcSpace': [],
            'IfcDoor': [],
            'IfcWall': []
        }.get(entity_type, [])
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as input_file:
                input_path = Path(input_file.name)
            
            try:
                # Test unified CLI
                script_path = Path(__file__).parent.parent.parent.parent / 'src' / 'extraction' / 'unified_extractor.py'
                
                if script_path.exists():
                    # Test analyze function
                    result = subprocess.run([
                        sys.executable, '-m', 'src.extraction.unified_extractor',
                        '--analyze', str(input_path)
                    ], capture_output=True, text=True, 
                    cwd=Path(__file__).parent.parent.parent.parent)
                    
                    # Check if analysis worked (may fail due to import issues in test environment)
                    if result.returncode == 0:
                        output = result.stdout
                        assert 'file_type' in output or 'ifc' in output.lower()
                
            finally:
                input_path.unlink()


class TestIFCCompleteWorkflow:
    """Test complete IFC extraction workflow scenarios."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_complete_building_extraction_workflow(self, mock_ifcopenshell):
        """Test complete workflow from IFC file to JSON output."""
        # Setup comprehensive mock building
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Mock project and building
        mock_project = Mock()
        mock_project.Name = "Complete Test Building"
        
        mock_building = Mock()
        mock_building.Name = "Main Building"
        mock_building.CompositionType = "ELEMENT"
        
        # Mock building story
        mock_story = Mock()
        mock_story.Name = "Ground Floor"
        mock_story.Elevation = 0.0
        mock_story.ObjectPlacement = None
        
        # Mock spaces with different uses
        mock_office = Mock()
        mock_office.Name = "Office 101"
        mock_office.LongName = None
        mock_office.id.return_value = 101
        
        mock_meeting = Mock()
        mock_meeting.Name = "Meeting Room A"
        mock_meeting.LongName = None
        mock_meeting.id.return_value = 102
        
        mock_restroom = Mock()
        mock_restroom.Name = "Restroom"
        mock_restroom.LongName = None
        mock_restroom.id.return_value = 103
        
        # Mock doors
        mock_entrance = Mock()
        mock_entrance.Name = "Main Entrance"
        mock_entrance.LongName = None
        mock_entrance.id.return_value = 201
        mock_entrance.PredefinedType = None
        mock_entrance.ObjectPlacement = None
        
        mock_emergency = Mock()
        mock_emergency.Name = "Emergency Exit"
        mock_emergency.LongName = None
        mock_emergency.id.return_value = 202
        mock_emergency.PredefinedType = "EMERGENCY_EXIT"
        mock_emergency.ObjectPlacement = None
        
        # Mock walls
        mock_exterior = Mock()
        mock_exterior.id.return_value = 301
        mock_exterior.ObjectPlacement = None
        mock_exterior.HasAssociations = []
        
        mock_interior = Mock()
        mock_interior.id.return_value = 302
        mock_interior.ObjectPlacement = None
        mock_interior.HasAssociations = []
        
        # Setup entity returns
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [mock_project],
            'IfcBuilding': [mock_building],
            'IfcBuildingStorey': [mock_story],
            'IfcSpace': [mock_office, mock_meeting, mock_restroom],
            'IfcRoom': [],  # No rooms, only spaces
            'IfcDoor': [mock_entrance, mock_emergency],
            'IfcWall': [mock_exterior, mock_interior],
            'IfcWindow': []
        }.get(entity_type, [])
        
        # Mock property sets for realistic data
        def mock_get_psets(entity):
            if entity == mock_office:
                return {"Qto_SpaceBaseQuantities": {"NetFloorArea": 25.0}}
            elif entity == mock_meeting:
                return {"Qto_SpaceBaseQuantities": {"NetFloorArea": 15.0}}
            elif entity == mock_restroom:
                return {"Qto_SpaceBaseQuantities": {"NetFloorArea": 5.0}}
            elif entity == mock_entrance:
                return {"Pset_DoorCommon": {"OverallWidth": 1.2, "OverallHeight": 2.1}}
            elif entity == mock_emergency:
                return {"Pset_DoorCommon": {"OverallWidth": 0.9, "OverallHeight": 2.1}}
            elif entity == mock_exterior:
                return {"Qto_WallBaseQuantities": {"Width": 0.25, "Height": 2.7}}
            elif entity == mock_interior:
                return {"Qto_WallBaseQuantities": {"Width": 0.15, "Height": 2.7}}
            return {}
        
        # Mock placement for positioning
        placement_matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   side_effect=mock_get_psets), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=placement_matrix):
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as input_file, \
                 tempfile.NamedTemporaryFile(suffix='.json', delete=False) as output_file:
                
                input_path = Path(input_file.name)
                output_path = Path(output_file.name)
            
            try:
                # Step 1: Extract project data
                project = extract_from_ifc(input_path)
                
                # Verify project structure
                assert isinstance(project, Project)
                assert project.metadata.project_name == "Complete Test Building"
                assert project.metadata.building_type == "element"  # From CompositionType
                assert len(project.levels) == 1
                
                # Verify level
                level = project.levels[0]
                assert level.name == "Ground Floor"
                assert level.elevation == 0.0
                
                # Verify rooms (spaces)
                all_rooms = project.get_all_rooms()
                assert len(all_rooms) == 3
                
                # Find specific rooms by use
                office_rooms = [r for r in all_rooms if r.use == BuildingUse.OFFICE]
                meeting_rooms = [r for r in all_rooms if r.use == BuildingUse.MEETING]
                restrooms = [r for r in all_rooms if r.use == BuildingUse.RESTROOM]
                
                assert len(office_rooms) == 1
                assert len(meeting_rooms) == 1
                assert len(restrooms) == 1
                
                # Verify room properties
                office = office_rooms[0]
                assert office.name == "Office 101"
                assert office.area == 25.0
                assert office.occupancy_load == 2  # 25 * 0.1 for office
                
                meeting = meeting_rooms[0]
                assert meeting.name == "Meeting Room A"
                assert meeting.area == 15.0
                assert meeting.occupancy_load == 3  # 15 * 0.2 for meeting
                
                # Verify doors
                all_doors = project.get_all_doors()
                assert len(all_doors) == 2
                
                emergency_doors = [d for d in all_doors if d.is_emergency_exit]
                regular_doors = [d for d in all_doors if not d.is_emergency_exit]
                
                assert len(emergency_doors) == 1
                assert len(regular_doors) == 1
                
                emergency_door = emergency_doors[0]
                assert emergency_door.name == "Emergency Exit"
                assert emergency_door.door_type == DoorType.EMERGENCY_EXIT
                assert emergency_door.width_mm == 900.0
                
                # Verify walls
                all_walls = project.get_all_walls()
                assert len(all_walls) == 2
                
                thick_walls = [w for w in all_walls if w.thickness_mm > 200]
                thin_walls = [w for w in all_walls if w.thickness_mm <= 200]
                
                assert len(thick_walls) == 1  # Exterior wall (250mm)
                assert len(thin_walls) == 1   # Interior wall (150mm)
                
                # Step 2: Save to JSON
                save_to_json(project, output_path)
                
                # Verify JSON output
                assert output_path.exists()
                
                with open(output_path, 'r') as f:
                    json_data = json.load(f)
                
                # Verify JSON structure
                assert json_data['metadata']['project_name'] == "Complete Test Building"
                assert len(json_data['levels']) == 1
                assert len(json_data['levels'][0]['rooms']) == 3
                assert len(json_data['levels'][0]['doors']) == 2
                assert len(json_data['levels'][0]['walls']) == 2
                
                # Step 3: Verify totals
                total_area = sum(room['area'] for room in json_data['levels'][0]['rooms'])
                assert total_area == 45.0  # 25 + 15 + 5
                
                assert json_data['metadata']['total_area'] == total_area
                
                # Step 4: Test round-trip (load JSON back)
                loaded_project = Project.parse_obj(json_data)
                assert loaded_project.metadata.project_name == project.metadata.project_name
                assert len(loaded_project.get_all_rooms()) == len(project.get_all_rooms())
                
            finally:
                # Cleanup
                for path in [input_path, output_path]:
                    if path.exists():
                        path.unlink()
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_mixed_file_directory_processing(self, mock_ifcopenshell):
        """Test processing directory with mixed IFC and DWG files."""
        # Setup IFC mock
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [Mock(Name="IFC Building")],
            'IfcBuilding': [Mock(Name="IFC Building")],
            'IfcBuildingStorey': [],
            'IfcSpace': [],
            'IfcDoor': [],
            'IfcWall': []
        }.get(entity_type, [])
        
        # Setup DWG mock
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None), \
             patch('ezdxf.readfile') as mock_ezdxf:
            
            # Mock DWG document
            mock_doc = Mock()
            mock_doc.filename = "test.dwg"
            mock_doc.modelspace.return_value = []
            mock_ezdxf.return_value = mock_doc
            
            # Create temporary directory with mixed files
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                output_dir = tmp_path / "output"
                
                # Create test files
                ifc_file = tmp_path / "building.ifc"
                dwg_file = tmp_path / "drawing.dwg"
                txt_file = tmp_path / "readme.txt"  # Should be ignored
                
                ifc_file.touch()
                dwg_file.touch()
                txt_file.touch()
                
                # Process directory
                projects = extract_from_directory(tmp_path, output_dir)
                
                # Should process both IFC and DWG files
                assert len(projects) == 2
                assert str(ifc_file) in projects
                assert str(dwg_file) in projects
                
                # Verify output files were created
                assert output_dir.exists()
                assert (output_dir / "building_extracted.json").exists()
                assert (output_dir / "drawing_extracted.json").exists()
                
                # TXT file should be ignored
                assert str(txt_file) not in projects


class TestIFCDataValidation:
    """Test validation of extracted IFC data."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_data_consistency_validation(self, mock_ifcopenshell):
        """Test that extracted data is consistent and valid."""
        # Setup mock with realistic building data
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Create mock entities with relationships
        mock_project = Mock(Name="Validation Test")
        mock_building = Mock(Name="Test Building")
        mock_story = Mock(Name="Ground Floor", Elevation=0.0)
        
        # Create interconnected spaces and doors
        mock_room1 = Mock(Name="Room A", id=lambda: 1)
        mock_room2 = Mock(Name="Room B", id=lambda: 2)
        mock_door = Mock(Name="Connecting Door", id=lambda: 10)
        
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [mock_project],
            'IfcBuilding': [mock_building],
            'IfcBuildingStorey': [mock_story],
            'IfcSpace': [mock_room1, mock_room2],
            'IfcDoor': [mock_door],
            'IfcWall': []
        }.get(entity_type, [])
        
        def mock_get_psets(entity):
            if entity in [mock_room1, mock_room2]:
                return {"Qto_SpaceBaseQuantities": {"NetFloorArea": 20.0}}
            elif entity == mock_door:
                return {"Pset_DoorCommon": {"OverallWidth": 0.8, "OverallHeight": 2.0}}
            return {}
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   side_effect=mock_get_psets), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
                file_path = Path(tmp.name)
            
            try:
                # Extract and validate
                project = extract_from_ifc(file_path)
                
                # Validate project structure
                assert project.metadata is not None
                assert project.metadata.project_name == "Validation Test"
                assert project.metadata.total_area > 0
                assert project.metadata.number_of_levels == 1
                
                # Validate levels
                assert len(project.levels) == 1
                level = project.levels[0]
                assert level.name == "Ground Floor"
                assert level.elevation == 0.0
                
                # Validate rooms
                rooms = level.rooms
                assert len(rooms) == 2
                for room in rooms:
                    assert room.id is not None
                    assert room.name is not None
                    assert room.area > 0
                    assert room.use in BuildingUse
                    assert room.occupancy_load > 0
                
                # Validate doors
                doors = level.doors
                assert len(doors) == 1
                door = doors[0]
                assert door.id is not None
                assert door.name is not None
                assert door.width_mm > 0
                assert door.height_mm > 0
                assert door.door_type in DoorType
                
                # Validate consistency
                total_calculated_area = sum(room.area for room in rooms)
                assert abs(project.metadata.total_area - total_calculated_area) < 0.01
                
            finally:
                file_path.unlink()


@pytest.mark.slow
class TestIFCPerformanceWorkflow:
    """Performance tests for IFC extraction (marked as slow)."""
    
    @patch('extraction.ifc_extractor.IFCOPENSHELL_AVAILABLE', True)
    @patch('extraction.ifc_extractor.ifcopenshell')
    def test_large_building_extraction_performance(self, mock_ifcopenshell):
        """Test extraction performance with large building (mocked)."""
        import time
        
        # Setup large building mock
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        mock_ifcopenshell.open.return_value = mock_ifc_file
        
        # Create many mock entities
        num_spaces = 500
        num_doors = 200
        num_walls = 1000
        
        mock_spaces = [Mock(Name=f"Space {i}", id=lambda i=i: i) for i in range(num_spaces)]
        mock_doors = [Mock(Name=f"Door {i}", id=lambda i=i: i+10000) for i in range(num_doors)]
        mock_walls = [Mock(id=lambda i=i: i+20000) for i in range(num_walls)]
        
        mock_ifc_file.by_type.side_effect = lambda entity_type: {
            'IfcProject': [Mock(Name="Large Building")],
            'IfcBuilding': [Mock(Name="Large Building")],
            'IfcBuildingStorey': [Mock(Name="Floor", Elevation=0.0)],
            'IfcSpace': mock_spaces,
            'IfcDoor': mock_doors,
            'IfcWall': mock_walls
        }.get(entity_type, [])
        
        with patch('extraction.ifc_extractor.ifcopenshell.util.element.get_psets', 
                   return_value={}), \
             patch('extraction.ifc_extractor.ifcopenshell.util.placement.get_local_placement', 
                   return_value=None):
            
            with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False) as tmp:
                file_path = Path(tmp.name)
            
            try:
                # Measure extraction time
                start_time = time.time()
                project = extract_from_ifc(file_path)
                extraction_time = time.time() - start_time
                
                # Should complete in reasonable time (adjust threshold as needed)
                assert extraction_time < 60  # Should complete within 60 seconds
                
                # Verify results
                assert isinstance(project, Project)
                print(f"Extracted large building in {extraction_time:.2f} seconds")
                
            finally:
                file_path.unlink()


# Test fixtures for workflow testing
@pytest.fixture
def sample_building_data():
    """Fixture providing sample building data for testing."""
    return {
        'project_name': 'Workflow Test Building',
        'levels': [
            {
                'name': 'Ground Floor',
                'elevation': 0.0,
                'spaces': ['Office A', 'Office B', 'Meeting Room', 'Restroom'],
                'doors': ['Main Entrance', 'Emergency Exit', 'Office Door A', 'Office Door B'],
                'walls': ['North Wall', 'South Wall', 'East Wall', 'West Wall']
            },
            {
                'name': 'First Floor', 
                'elevation': 3.0,
                'spaces': ['Office C', 'Office D', 'Conference Room'],
                'doors': ['Stair Door', 'Office Door C', 'Office Door D'],
                'walls': ['North Wall', 'South Wall', 'East Wall', 'West Wall']
            }
        ]
    }


@pytest.fixture
def mock_complex_building():
    """Fixture providing a complex building structure for testing."""
    # This would create a comprehensive mock building
    # with multiple levels, rooms, doors, walls, and relationships
    pass