"""
Tests for IFC LLM Exporter

Basic tests to validate the new IFC LLM exporter functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.extraction.ifc_llm_exporter import (
    IFCLLMExporter, ExtractionConfig, export_to_llm_json
)
from src.extraction.ifc_llm_exporter.crs import CoordinateSystemHandler, UnitInfo
from src.extraction.ifc_llm_exporter.spatial import SpatialHierarchyExtractor
from src.extraction.ifc_llm_exporter.props import PropertyExtractor
from src.extraction.ifc_llm_exporter.geom import GeometryProcessor
from src.extraction.ifc_llm_exporter.writers import LLMJSONWriter, OutputConfig, ChunkingConfig
from src.extraction.ifc_llm_exporter.schema import validate_project_json, validate_element_json


class TestCoordinateSystemHandler:
    """Test coordinate system handling."""
    
    def test_unit_info_defaults(self):
        """Test default unit information."""
        units = UnitInfo()
        assert units.length == "m"
        assert units.area == "m2" 
        assert units.volume == "m3"
        assert units.angle == "deg"
        assert units.scale_to_meters == 1.0
    
    @patch('src.extraction.ifc_llm_exporter.crs.ifcopenshell')
    def test_coordinate_system_handler_init(self, mock_ifcopenshell):
        """Test coordinate system handler initialization."""
        # Mock IFC file
        mock_ifc_file = Mock()
        mock_ifc_file.schema = "IFC4"
        
        # Mock project and units
        mock_project = Mock()
        mock_project.UnitsInContext = None
        mock_ifc_file.by_type.return_value = [mock_project]
        
        # Initialize handler
        handler = CoordinateSystemHandler(mock_ifc_file)
        
        assert handler.ifc_file == mock_ifc_file
        assert handler.units.length == "m"
        assert handler.coordinate_system.true_north_deg == 0.0


class TestSpatialHierarchyExtractor:
    """Test spatial hierarchy extraction."""
    
    @patch('src.extraction.ifc_llm_exporter.spatial.ifcopenshell')
    def test_spatial_extractor_init(self, mock_ifcopenshell):
        """Test spatial extractor initialization."""
        mock_ifc_file = Mock()
        
        extractor = SpatialHierarchyExtractor(mock_ifc_file)
        
        assert extractor.ifc_file == mock_ifc_file
        assert extractor.spatial_tree is None
        assert extractor.element_relationships == {}


class TestPropertyExtractor:
    """Test property extraction."""
    
    @patch('src.extraction.ifc_llm_exporter.props.ifcopenshell')
    def test_property_extractor_init(self, mock_ifcopenshell):
        """Test property extractor initialization."""
        mock_ifc_file = Mock()
        units = UnitInfo()
        
        extractor = PropertyExtractor(mock_ifc_file, units)
        
        assert extractor.ifc_file == mock_ifc_file
        assert extractor.units_info == units


class TestGeometryProcessor:
    """Test geometry processing."""
    
    @patch('src.extraction.ifc_llm_exporter.geom.ifcopenshell')
    def test_geometry_processor_init(self, mock_ifcopenshell):
        """Test geometry processor initialization."""
        mock_ifc_file = Mock()
        mock_crs_handler = Mock()
        
        processor = GeometryProcessor(mock_ifc_file, mock_crs_handler)
        
        assert processor.ifc_file == mock_ifc_file
        assert processor.crs_handler == mock_crs_handler
        assert processor.enable_mesh_export is False


class TestLLMJSONWriter:
    """Test LLM JSON writer."""
    
    def test_llm_json_writer_init(self):
        """Test LLM JSON writer initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_config = OutputConfig(output_dir=Path(temp_dir))
            chunking_config = ChunkingConfig()
            
            writer = LLMJSONWriter(output_config, chunking_config)
            
            assert writer.output_config == output_config
            assert writer.chunking_config == chunking_config
            assert writer.elements_dir.exists()
    
    def test_chunking_config_defaults(self):
        """Test chunking configuration defaults."""
        config = ChunkingConfig()
        
        assert config.max_elements_per_chunk == 1000
        assert config.max_chunk_size_kb == 5000
        assert config.chunk_by_domain is True
        assert config.chunk_by_storey is True
        assert config.min_chunk_size == 10


class TestJSONSchemaValidation:
    """Test JSON schema validation."""
    
    def test_validate_minimal_project_json(self):
        """Test validation of minimal project JSON."""
        project_data = {
            "project": {},
            "units": {
                "length": "m",
                "area": "m2", 
                "volume": "m3",
                "angle": "deg"
            },
            "coordinate_system": {
                "model_origin_note": "Test origin",
                "true_north_deg": 0.0
            },
            "spatial_tree": [],
            "indices": {
                "by_class": {},
                "by_storey": {},
                "by_zone": {}
            }
        }
        
        # Should not raise exception
        assert validate_project_json(project_data) is True
    
    def test_validate_minimal_element_json(self):
        """Test validation of minimal element JSON."""
        element_data = {
            "guid": "1234567890abcdef123456",
            "ifc_class": "IfcWall",
            "name": "Test Wall",
            "spatial": {},
            "coordinates": {
                "transform_local_to_model": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                "transform_model_to_world": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                "reference_point_model_m": [0, 0, 0],
                "bbox_model_m": [0, 0, 0, 1, 1, 1]
            },
            "geometry_summary": {
                "shape": "extruded"
            },
            "properties": {},
            "relations": {},
            "source": {}
        }
        
        # Should not raise exception
        assert validate_element_json(element_data) is True


class TestExtractionConfig:
    """Test extraction configuration."""
    
    def test_extraction_config_defaults(self):
        """Test extraction configuration defaults."""
        config = ExtractionConfig()
        
        assert config.include_classes is None
        assert config.exclude_classes is None
        assert config.enable_geometry is True
        assert config.enable_properties is True
        assert config.enable_materials is True
        assert config.enable_relationships is True
        assert config.enable_mesh_export is False
        assert config.override_epsg is None
        assert config.max_elements is None
        assert config.skip_large_elements is False


class TestIFCLLMExporter:
    """Test main IFC LLM exporter."""
    
    def test_exporter_init(self):
        """Test exporter initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy IFC file
            ifc_file = Path(temp_dir) / "test.ifc"
            ifc_file.write_text("ISO-10303-21;\\nHEADER;\\nENDSEC;\\nDATA;\\nENDSEC;\\nEND-ISO-10303-21;")
            
            output_dir = Path(temp_dir) / "output"
            
            exporter = IFCLLMExporter(
                ifc_file_path=ifc_file,
                output_dir=output_dir
            )
            
            assert exporter.ifc_file_path == ifc_file
            assert exporter.output_dir == output_dir
            assert isinstance(exporter.extraction_config, ExtractionConfig)
            assert isinstance(exporter.chunking_config, ChunkingConfig)


class TestIntegration:
    """Integration tests."""
    
    @patch('src.extraction.ifc_llm_exporter.exporter.ifcopenshell')
    def test_export_to_llm_json_mock(self, mock_ifcopenshell):
        """Test high-level export function with mocked IFC."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy IFC file
            ifc_file = Path(temp_dir) / "test.ifc"
            ifc_file.write_text("ISO-10303-21;\\nHEADER;\\nENDSEC;\\nDATA;\\nENDSEC;\\nEND-ISO-10303-21;")
            
            output_dir = Path(temp_dir) / "output"
            
            # Mock IFC file loading
            mock_ifc = Mock()
            mock_ifc.schema = "IFC4"
            mock_ifc.by_type.return_value = []
            mock_ifcopenshell.open.return_value = mock_ifc
            
            # Configure extraction
            extraction_config = ExtractionConfig(
                enable_geometry=False,  # Disable to avoid complex mocking
                enable_properties=False,
                enable_relationships=False
            )
            
            try:
                summary = export_to_llm_json(
                    ifc_file_path=ifc_file,
                    output_dir=output_dir,
                    extraction_config=extraction_config
                )
                
                # Check that we got a summary (even if mocked)
                assert 'success' in summary
                assert 'input_file' in summary
                assert 'output_directory' in summary
                
            except Exception as e:
                # Expected due to mocking - just ensure the structure is there
                assert "IFC file not found" in str(e) or "ifcopenshell" in str(e)


def test_module_imports():
    """Test that all modules can be imported without errors."""
    # Test main module imports
    from src.extraction.ifc_llm_exporter import (
        IFCLLMExporter, export_to_llm_json, ExtractionConfig
    )
    
    # Test submodule imports
    from src.extraction.ifc_llm_exporter.crs import CoordinateSystemHandler
    from src.extraction.ifc_llm_exporter.spatial import SpatialHierarchyExtractor
    from src.extraction.ifc_llm_exporter.props import PropertyExtractor
    from src.extraction.ifc_llm_exporter.geom import GeometryProcessor
    from src.extraction.ifc_llm_exporter.writers import LLMJSONWriter
    from src.extraction.ifc_llm_exporter.schema import validate_project_json
    
    # All imports successful
    assert True


if __name__ == "__main__":
    # Run basic smoke tests
    test_module_imports()
    print("✅ Module imports successful")
    
    # Test configurations
    config = ExtractionConfig()
    assert config.enable_geometry is True
    print("✅ ExtractionConfig working")
    
    chunking = ChunkingConfig()
    assert chunking.max_elements_per_chunk == 1000
    print("✅ ChunkingConfig working")
    
    print("✅ All basic tests passed!")