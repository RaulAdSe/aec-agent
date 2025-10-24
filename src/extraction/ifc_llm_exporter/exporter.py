"""
Main IFC LLM Exporter

This module provides the main exporter class and CLI interface for converting
IFC files to LLM-optimized JSON format with comprehensive property extraction,
coordinate system handling, and spatial relationship mapping.
"""

import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime

try:
    import ifcopenshell
except ImportError:
    ifcopenshell = None

from .crs import CoordinateSystemHandler
from .spatial import SpatialHierarchyExtractor
from .props import PropertyExtractor
from .geom import GeometryProcessor
from .writers import LLMJSONWriter, OutputConfig, ChunkingConfig
from .schema import validate_project_json, validate_elements_batch, sanitize_element_for_schema

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for IFC extraction."""
    # Element filtering
    include_classes: Optional[Set[str]] = None  # e.g., {'IfcWall', 'IfcDoor'}
    exclude_classes: Optional[Set[str]] = None  # e.g., {'IfcOpeningElement'}
    
    # Processing options
    enable_geometry: bool = True
    enable_properties: bool = True
    enable_materials: bool = True
    enable_relationships: bool = True
    enable_mesh_export: bool = False
    
    # Coordinate system overrides
    override_epsg: Optional[int] = None
    
    # Performance options
    max_elements: Optional[int] = None
    skip_large_elements: bool = False
    large_element_threshold_mb: float = 10.0


@dataclass
class ExtractionStats:
    """Statistics from extraction process."""
    total_elements_found: int = 0
    elements_processed: int = 0
    elements_skipped: int = 0
    elements_failed: int = 0
    processing_time_seconds: float = 0.0
    file_size_mb: float = 0.0


class IFCLLMExporter:
    """
    Main IFC to LLM-JSON exporter.
    
    Orchestrates the complete extraction pipeline from IFC files to 
    LLM-optimized chunked JSON output.
    """
    
    def __init__(self, ifc_file_path: Path, output_dir: Path, 
                 extraction_config: ExtractionConfig = None,
                 chunking_config: ChunkingConfig = None):
        """
        Initialize the exporter.
        
        Args:
            ifc_file_path: Path to IFC file
            output_dir: Output directory for generated files
            extraction_config: Extraction configuration
            chunking_config: Chunking configuration
        """
        if not ifcopenshell:
            raise ImportError("ifcopenshell is required for IFC extraction")
        
        self.ifc_file_path = ifc_file_path
        self.output_dir = output_dir
        self.extraction_config = extraction_config or ExtractionConfig()
        self.chunking_config = chunking_config or ChunkingConfig()
        
        # Initialize components
        self.ifc_file = None
        self.crs_handler = None
        self.spatial_extractor = None
        self.property_extractor = None
        self.geometry_processor = None
        self.writer = None
        
        # Statistics
        self.stats = ExtractionStats()
        
        logger.info(f"Initialized IFC LLM Exporter: {ifc_file_path} -> {output_dir}")
    
    def export(self) -> Dict[str, Any]:
        """
        Run the complete export process.
        
        Returns:
            Export summary with statistics and output information
        """
        start_time = time.time()
        
        try:
            # Load IFC file
            self._load_ifc_file()
            
            # Initialize processors
            self._initialize_processors()
            
            # Extract spatial hierarchy and indices
            spatial_data = self._extract_spatial_data()
            
            # Extract all elements
            elements = self._extract_elements()
            
            # Write output files
            output_summary = self._write_output(spatial_data, elements)
            
            # Calculate final statistics
            self.stats.processing_time_seconds = time.time() - start_time
            
            # Generate export summary
            export_summary = {
                'success': True,
                'input_file': str(self.ifc_file_path),
                'output_directory': str(self.output_dir),
                'statistics': self.stats.__dict__,
                'output_files': output_summary,
                'extraction_config': self.extraction_config.__dict__,
                'chunking_config': self.chunking_config.__dict__
            }
            
            logger.info(f"Export completed successfully in {self.stats.processing_time_seconds:.2f}s")
            logger.info(f"Processed {self.stats.elements_processed}/{self.stats.total_elements_found} elements")
            
            return export_summary
            
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            
            export_summary = {
                'success': False,
                'error': str(e),
                'input_file': str(self.ifc_file_path),
                'output_directory': str(self.output_dir),
                'statistics': self.stats.__dict__,
                'processing_time_seconds': time.time() - start_time
            }
            
            return export_summary
    
    def _load_ifc_file(self):
        """Load and validate IFC file."""
        logger.info(f"Loading IFC file: {self.ifc_file_path}")
        
        if not self.ifc_file_path.exists():
            raise FileNotFoundError(f"IFC file not found: {self.ifc_file_path}")
        
        # Get file size
        file_size_bytes = self.ifc_file_path.stat().st_size
        self.stats.file_size_mb = file_size_bytes / (1024 * 1024)
        
        logger.info(f"IFC file size: {self.stats.file_size_mb:.2f} MB")
        
        # Load file
        self.ifc_file = ifcopenshell.open(str(self.ifc_file_path))
        
        logger.info(f"Loaded IFC file: {self.ifc_file.schema}")
        logger.info(f"Total entities: {len(list(self.ifc_file))}")
    
    def _initialize_processors(self):
        """Initialize all processing components."""
        logger.info("Initializing processors...")
        
        # Coordinate system handler
        self.crs_handler = CoordinateSystemHandler(self.ifc_file)
        
        # Override EPSG if specified
        if self.extraction_config.override_epsg:
            if self.crs_handler.coordinate_system.world:
                self.crs_handler.coordinate_system.world.epsg = self.extraction_config.override_epsg
                logger.info(f"Overrode EPSG code: {self.extraction_config.override_epsg}")
        
        # Spatial hierarchy extractor
        self.spatial_extractor = SpatialHierarchyExtractor(self.ifc_file)
        
        # Property extractor
        if self.extraction_config.enable_properties:
            self.property_extractor = PropertyExtractor(
                self.ifc_file, 
                self.crs_handler.units
            )
        
        # Geometry processor
        if self.extraction_config.enable_geometry:
            mesh_output_dir = self.output_dir / "meshes" if self.extraction_config.enable_mesh_export else None
            self.geometry_processor = GeometryProcessor(
                self.ifc_file,
                self.crs_handler,
                enable_mesh_export=self.extraction_config.enable_mesh_export,
                mesh_output_dir=mesh_output_dir
            )
        
        # Output writer
        output_config = OutputConfig(output_dir=self.output_dir)
        self.writer = LLMJSONWriter(output_config, self.chunking_config)
        
        logger.info("Processors initialized successfully")
    
    def _extract_spatial_data(self) -> Dict[str, Any]:
        """Extract spatial hierarchy and build indices."""
        logger.info("Extracting spatial hierarchy...")
        
        # Extract spatial tree
        spatial_tree = self.spatial_extractor.extract_spatial_hierarchy()
        
        # Convert to dictionary format
        spatial_data = self.spatial_extractor.to_dict()
        
        logger.info(f"Extracted spatial hierarchy with {len(spatial_data.get('spatial_tree', []))} top-level nodes")
        
        return spatial_data
    
    def _extract_elements(self) -> List[Dict[str, Any]]:
        """Extract all elements with their properties and relationships."""
        logger.info("Extracting elements...")
        
        # Get relevant element types
        target_classes = self._get_target_element_classes()
        
        all_elements = []
        element_guids = []
        
        # Collect elements by class
        for ifc_class in target_classes:
            try:
                elements = self.ifc_file.by_type(ifc_class)
                logger.debug(f"Found {len(elements)} {ifc_class} elements")
                
                for element in elements:
                    if self._should_process_element(element):
                        all_elements.append(element)
                        element_guids.append(element.GlobalId)
                        
            except Exception as e:
                logger.warning(f"Error collecting {ifc_class} elements: {e}")
                continue
        
        self.stats.total_elements_found = len(all_elements)
        logger.info(f"Found {self.stats.total_elements_found} elements to process")
        
        # Apply max elements limit
        if self.extraction_config.max_elements:
            all_elements = all_elements[:self.extraction_config.max_elements]
            element_guids = element_guids[:self.extraction_config.max_elements]
            logger.info(f"Limited to {len(all_elements)} elements due to max_elements setting")
        
        # Extract relationships for all elements
        if self.extraction_config.enable_relationships:
            logger.info("Extracting element relationships...")
            element_relationships = self.spatial_extractor.extract_element_relationships(element_guids)
        else:
            element_relationships = {}
        
        # Process each element
        processed_elements = []
        
        for i, element in enumerate(all_elements):
            if i % 100 == 0:
                logger.info(f"Processing element {i+1}/{len(all_elements)}")
            
            try:
                element_data = self._process_single_element(element, element_relationships)
                if element_data:
                    processed_elements.append(element_data)
                    self.stats.elements_processed += 1
                else:
                    self.stats.elements_skipped += 1
                    
            except Exception as e:
                logger.warning(f"Error processing element {element.id()}: {e}")
                self.stats.elements_failed += 1
                continue
        
        # Build indices
        logger.info("Building element indices...")
        indices = self.spatial_extractor.build_indices(processed_elements)
        
        # Update spatial data with indices
        spatial_data = self.spatial_extractor.to_dict()
        
        logger.info(f"Successfully processed {len(processed_elements)} elements")
        
        return processed_elements
    
    def _get_target_element_classes(self) -> List[str]:
        """Get list of IFC classes to extract."""
        # Default comprehensive class list
        default_classes = [
            # Architecture
            'IfcWall', 'IfcWallStandardCase', 'IfcSlab', 'IfcDoor', 'IfcWindow',
            'IfcStair', 'IfcRoof', 'IfcCovering', 'IfcCurtainWall', 'IfcRailing',
            
            # Structure
            'IfcBeam', 'IfcColumn', 'IfcFooting', 'IfcPile', 'IfcPlate',
            
            # MEP
            'IfcFlowTerminal', 'IfcFlowSegment', 'IfcFlowFitting', 'IfcFlowController',
            'IfcEnergyConversionDevice', 'IfcFlowMovingDevice', 'IfcFlowStorageDevice',
            'IfcDistributionElement', 'IfcCableCarrierSegment', 'IfcCableSegment',
            
            # Spaces and Zones
            'IfcSpace', 'IfcZone',
            
            # Other
            'IfcBuildingElementProxy', 'IfcFurnishingElement', 'IfcEquipmentElement'
        ]
        
        # Apply include/exclude filters
        if self.extraction_config.include_classes:
            target_classes = list(self.extraction_config.include_classes)
        else:
            target_classes = default_classes
        
        if self.extraction_config.exclude_classes:
            target_classes = [cls for cls in target_classes 
                            if cls not in self.extraction_config.exclude_classes]
        
        # Filter to only classes that exist in the file
        available_classes = []
        for cls in target_classes:
            try:
                elements = self.ifc_file.by_type(cls)
                if elements:
                    available_classes.append(cls)
            except:
                continue
        
        logger.info(f"Target element classes: {available_classes}")
        return available_classes
    
    def _should_process_element(self, element) -> bool:
        """Check if an element should be processed."""
        # Skip elements without GUID
        if not hasattr(element, 'GlobalId') or not element.GlobalId:
            return False
        
        # Skip large elements if configured
        if self.extraction_config.skip_large_elements:
            # This is a placeholder - actual size check would require geometry analysis
            pass
        
        return True
    
    def _process_single_element(self, element, element_relationships: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single element and extract all its data."""
        try:
            element_data = {
                'guid': element.GlobalId,
                'ifc_class': element.is_a(),
                'name': getattr(element, 'Name', None) or f"{element.is_a()}_{element.id()}",
                'type': self._get_element_type_name(element)
            }
            
            # Spatial containment
            element_data['spatial'] = self.spatial_extractor.find_element_spatial_container(element)
            
            # Coordinates and geometry
            if self.extraction_config.enable_geometry and self.geometry_processor:
                coordinate_info, geometry_summary = self.geometry_processor.process_element_geometry(element)
                geom_data = self.geometry_processor.to_dict(coordinate_info, geometry_summary)
                element_data.update(geom_data)
            else:
                # Minimal coordinate info
                element_data['coordinates'] = {
                    'transform_local_to_model': [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                    'transform_model_to_world': [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                    'reference_point_model_m': [0, 0, 0],
                    'bbox_model_m': [0, 0, 0, 0, 0, 0]
                }
                element_data['geometry_summary'] = {'shape': 'unknown'}
            
            # Properties and materials
            if self.extraction_config.enable_properties and self.property_extractor:
                property_sets = self.property_extractor.extract_element_properties(element)
                
                if self.extraction_config.enable_materials:
                    materials = self.property_extractor.extract_material_properties(element)
                else:
                    materials = []
                
                props_data = self.property_extractor.to_dict(property_sets, materials)
                element_data.update(props_data)
            else:
                element_data['properties'] = {}
                element_data['materials'] = []
            
            # Relationships
            if self.extraction_config.enable_relationships:
                guid = element.GlobalId
                if guid in element_relationships:
                    rel = element_relationships[guid]
                    element_data['relations'] = {
                        'type_defined_by': rel.type_element,
                        'connects_to': rel.connected_elements,
                        'voids': rel.voids_elements,
                        'voided_by': rel.voided_by,
                        'adjacent_spaces': rel.adjacent_spaces
                    }
                else:
                    element_data['relations'] = {}
            else:
                element_data['relations'] = {}
            
            # Source information
            element_data['source'] = {
                'ifc_label': getattr(element, 'Tag', None) or getattr(element, 'Name', None),
                'ifc_id': element.id(),
                'file_name': self.ifc_file_path.name,
                'extraction_date': datetime.now().isoformat()
            }
            
            # Sanitize for schema compliance
            element_data = sanitize_element_for_schema(element_data)
            
            return element_data
            
        except Exception as e:
            logger.warning(f"Error processing element {element.id()}: {e}")
            return None
    
    def _get_element_type_name(self, element) -> Optional[str]:
        """Get the type name for an element."""
        try:
            if hasattr(element, 'IsDefinedBy'):
                for rel in element.IsDefinedBy:
                    if rel.is_a('IfcRelDefinesByType'):
                        type_element = rel.RelatingType
                        if type_element:
                            return getattr(type_element, 'Name', None) or f"{type_element.is_a()}"
            return None
        except:
            return None
    
    def _write_output(self, spatial_data: Dict[str, Any], elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Write all output files."""
        logger.info("Writing output files...")
        
        # Prepare project metadata
        project_metadata = self.crs_handler.to_dict()
        
        # Write project file
        project_file = self.writer.write_project_data(
            project_metadata, 
            spatial_data, 
            spatial_data.get('indices', {})
        )
        
        # Validate elements if configured
        if self.writer.output_config.validate_json:
            logger.info("Validating elements against schema...")
            elements = validate_elements_batch(elements, stop_on_error=False)
            logger.info(f"Validation complete: {len(elements)} valid elements")
        
        # Write elements in chunks
        chunks_info = self.writer.write_elements_chunked(elements)
        
        # Write chunk manifest
        manifest_file = self.writer.write_chunk_manifest()
        
        # Get output summary
        output_summary = self.writer.get_output_summary()
        output_summary['project_file'] = str(project_file)
        output_summary['manifest_file'] = str(manifest_file)
        
        logger.info(f"Output written to: {self.output_dir}")
        logger.info(f"Generated {len(chunks_info)} chunk files")
        
        return output_summary


def export_to_llm_json(ifc_file_path: Path, output_dir: Path,
                      extraction_config: ExtractionConfig = None,
                      chunking_config: ChunkingConfig = None) -> Dict[str, Any]:
    """
    High-level function to export IFC file to LLM-optimized JSON.
    
    Args:
        ifc_file_path: Path to IFC file
        output_dir: Output directory
        extraction_config: Extraction configuration
        chunking_config: Chunking configuration
        
    Returns:
        Export summary
    """
    exporter = IFCLLMExporter(
        ifc_file_path=ifc_file_path,
        output_dir=output_dir,
        extraction_config=extraction_config,
        chunking_config=chunking_config
    )
    
    return exporter.export()


def main():
    """CLI interface for IFC LLM exporter."""
    parser = argparse.ArgumentParser(
        description="Export IFC files to LLM-optimized JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic export
  python -m ifc_llm_exporter.exporter input.ifc output/

  # Export specific element types
  python -m ifc_llm_exporter.exporter input.ifc output/ --classes IfcWall,IfcDoor,IfcSpace

  # Export with mesh files
  python -m ifc_llm_exporter.exporter input.ifc output/ --enable-meshes

  # Custom chunking
  python -m ifc_llm_exporter.exporter input.ifc output/ --max-chunk-elements 500

  # Override coordinate system
  python -m ifc_llm_exporter.exporter input.ifc output/ --epsg 25831
        """
    )
    
    # Required arguments
    parser.add_argument('input_file', type=Path, help='Input IFC file')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    
    # Element filtering
    parser.add_argument('--classes', type=str, help='Comma-separated list of IFC classes to include')
    parser.add_argument('--exclude-classes', type=str, help='Comma-separated list of IFC classes to exclude')
    parser.add_argument('--max-elements', type=int, help='Maximum number of elements to process')
    
    # Processing options
    parser.add_argument('--no-geometry', action='store_true', help='Skip geometry processing')
    parser.add_argument('--no-properties', action='store_true', help='Skip property extraction')
    parser.add_argument('--no-materials', action='store_true', help='Skip material extraction')
    parser.add_argument('--no-relationships', action='store_true', help='Skip relationship extraction')
    parser.add_argument('--enable-meshes', action='store_true', help='Export mesh files')
    
    # Coordinate system
    parser.add_argument('--epsg', type=int, help='Override EPSG coordinate system code')
    
    # Chunking options
    parser.add_argument('--max-chunk-elements', type=int, default=1000, help='Max elements per chunk')
    parser.add_argument('--max-chunk-size-kb', type=int, default=5000, help='Max chunk size in KB')
    parser.add_argument('--no-chunk-by-domain', action='store_true', help='Disable domain-based chunking')
    parser.add_argument('--no-chunk-by-storey', action='store_true', help='Disable storey-based chunking')
    
    # Output options
    parser.add_argument('--no-validation', action='store_true', help='Skip JSON schema validation')
    parser.add_argument('--pretty-print', action='store_true', help='Pretty print JSON output')
    
    # Other options
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--log-file', type=Path, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(args.log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Build extraction config
        extraction_config = ExtractionConfig(
            enable_geometry=not args.no_geometry,
            enable_properties=not args.no_properties,
            enable_materials=not args.no_materials,
            enable_relationships=not args.no_relationships,
            enable_mesh_export=args.enable_meshes,
            override_epsg=args.epsg,
            max_elements=args.max_elements
        )
        
        # Parse class filters
        if args.classes:
            extraction_config.include_classes = set(args.classes.split(','))
        
        if args.exclude_classes:
            extraction_config.exclude_classes = set(args.exclude_classes.split(','))
        
        # Build chunking config
        chunking_config = ChunkingConfig(
            max_elements_per_chunk=args.max_chunk_elements,
            max_chunk_size_kb=args.max_chunk_size_kb,
            chunk_by_domain=not args.no_chunk_by_domain,
            chunk_by_storey=not args.no_chunk_by_storey
        )
        
        # Export
        logger.info(f"Starting IFC LLM export: {args.input_file} -> {args.output_dir}")
        
        summary = export_to_llm_json(
            ifc_file_path=args.input_file,
            output_dir=args.output_dir,
            extraction_config=extraction_config,
            chunking_config=chunking_config
        )
        
        # Print summary
        if summary['success']:
            print("✅ Export completed successfully!")
            print(f"Input: {summary['input_file']}")
            print(f"Output: {summary['output_directory']}")
            print(f"Elements processed: {summary['statistics']['elements_processed']}")
            print(f"Processing time: {summary['statistics']['processing_time_seconds']:.2f}s")
            print(f"Output chunks: {summary['output_files']['total_chunks']}")
            print(f"Total size: {summary['output_files']['total_size_mb']} MB")
        else:
            print("❌ Export failed!")
            print(f"Error: {summary['error']}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\\n❌ Export cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())