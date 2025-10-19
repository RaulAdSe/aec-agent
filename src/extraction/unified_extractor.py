"""
Unified extraction pipeline for AEC Compliance Agent.

This module provides a single interface for extracting building data from both
DWG/DXF and Revit files, with automatic file type detection and appropriate
extraction method selection.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime

try:
    from .dwg_extractor import DWGExtractor, extract_and_save as extract_dwg
    from .revit_extractor import RevitExtractor, extract_and_save_revit
    from ..schemas import Project
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from extraction.dwg_extractor import DWGExtractor, extract_and_save as extract_dwg
    from extraction.revit_extractor import RevitExtractor, extract_and_save_revit
    from schemas import Project

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedExtractor:
    """
    Unified extractor that can handle both DWG/DXF and Revit files.
    
    This class automatically detects file types and uses the appropriate
    extraction method for each file format.
    """
    
    def __init__(self):
        self.dwg_extractor = DWGExtractor()
        self.revit_extractor = RevitExtractor()
        
    def extract_from_file(self, file_path: Union[str, Path], 
                         project_name: Optional[str] = None,
                         level_name: str = "Planta Baja",
                         output_path: Optional[Union[str, Path]] = None) -> Project:
        """
        Extract building data from a CAD file (DWG, DXF, or RVT).
        
        Args:
            file_path: Path to the CAD file
            project_name: Optional project name override
            level_name: Name of the building level
            output_path: Optional path to save extracted JSON
            
        Returns:
            Project object with extracted data
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file type
        file_type = self._detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type} for {file_path.name}")
        
        # Extract based on file type
        if file_type in ['dwg', 'dxf']:
            project = self._extract_dwg_file(file_path, project_name, level_name)
        elif file_type == 'rvt':
            project = self._extract_revit_file(file_path, project_name, level_name)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Save to JSON if output path provided
        if output_path:
            output_path = Path(output_path)
            self._save_project_json(project, output_path)
        
        return project
    
    def extract_from_directory(self, directory_path: Union[str, Path],
                              output_dir: Optional[Union[str, Path]] = None,
                              level_name: str = "Planta Baja") -> Dict[str, Project]:
        """
        Extract building data from all CAD files in a directory.
        
        Args:
            directory_path: Path to directory containing CAD files
            output_dir: Optional directory to save extracted JSON files
            level_name: Name of the building level
            
        Returns:
            Dictionary mapping file paths to Project objects
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all CAD files
        cad_files = self._find_cad_files(directory_path)
        logger.info(f"Found {len(cad_files)} CAD files in {directory_path}")
        
        # Extract from each file
        projects = {}
        successful_extractions = 0
        
        for file_path in cad_files:
            try:
                logger.info(f"Processing: {file_path.name}")
                
                # Generate output path if output directory specified
                output_path = None
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{file_path.stem}_extracted.json"
                
                # Extract data
                project = self.extract_from_file(
                    file_path, 
                    level_name=level_name,
                    output_path=output_path
                )
                
                projects[str(file_path)] = project
                successful_extractions += 1
                
                logger.info(f"✅ Successfully extracted: {file_path.name}")
                logger.info(f"   - Rooms: {len(project.rooms)}")
                logger.info(f"   - Doors: {len(project.doors)}")
                logger.info(f"   - Walls: {len(project.walls)}")
                logger.info(f"   - Fire Equipment: {len(project.fire_equipment)}")
                logger.info(f"   - Sectors: {len(project.sectors)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to extract {file_path.name}: {e}")
                continue
        
        logger.info(f"Extraction complete: {successful_extractions}/{len(cad_files)} files processed successfully")
        return projects
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect the type of CAD file based on extension."""
        extension = file_path.suffix.lower()
        
        if extension in ['.dwg', '.dxf']:
            return extension[1:]  # Remove the dot
        elif extension == '.rvt':
            return 'rvt'
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    def _extract_dwg_file(self, file_path: Path, project_name: Optional[str], level_name: str) -> Project:
        """Extract data from DWG/DXF file."""
        logger.info(f"Extracting from DWG/DXF: {file_path.name}")
        
        # Use the DWG extractor
        project = self.dwg_extractor.extract_from_file(
            file_path=file_path,
            project_name=project_name,
            level_name=level_name
        )
        
        return project
    
    def _extract_revit_file(self, file_path: Path, project_name: Optional[str], level_name: str) -> Project:
        """Extract data from Revit file."""
        logger.info(f"Extracting from Revit: {file_path.name}")
        
        # For Revit files, we need to use the Revit API
        # This will return mock data if not running inside Revit
        project = self.revit_extractor.extract_from_document(
            doc=None,  # Will use active document or create mock
            project_name=project_name,
            level_name=level_name
        )
        
        return project
    
    def _find_cad_files(self, directory_path: Path) -> list:
        """Find all CAD files in a directory."""
        supported_extensions = ['.dwg', '.dxf', '.rvt']
        cad_files = []
        
        for extension in supported_extensions:
            # Find files with this extension (case insensitive)
            files = list(directory_path.glob(f"*{extension}"))
            files.extend(directory_path.glob(f"*{extension.upper()}"))
            cad_files.extend(files)
        
        return sorted(cad_files)
    
    def _save_project_json(self, project: Project, output_path: Path):
        """Save project to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                project.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        
        logger.info(f"Saved extracted data to: {output_path}")
    
    def analyze_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a CAD file without extracting full data.
        
        Args:
            file_path: Path to the CAD file
            
        Returns:
            Dictionary with file analysis information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = self._detect_file_type(file_path)
        
        analysis = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "modified_date": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            "supported": file_type in ['dwg', 'dxf', 'rvt']
        }
        
        if file_type in ['dwg', 'dxf']:
            # For DWG/DXF files, we can do a quick analysis
            try:
                import ezdxf
                doc = ezdxf.readfile(str(file_path))
                msp = doc.modelspace()
                
                # Count entities
                entity_counts = {}
                for entity in msp:
                    entity_type = entity.dxftype()
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                
                analysis.update({
                    "entity_counts": entity_counts,
                    "total_entities": sum(entity_counts.values()),
                    "layers": list(doc.layers.keys()),
                    "blocks": list(doc.blocks.keys())
                })
                
            except Exception as e:
                analysis["analysis_error"] = str(e)
        
        elif file_type == 'rvt':
            # For Revit files, we can only provide basic file info
            analysis.update({
                "note": "Revit file analysis requires Revit API access",
                "extraction_available": False
            })
        
        return analysis


def extract_from_file(file_path: Union[str, Path], 
                     project_name: Optional[str] = None,
                     level_name: str = "Planta Baja",
                     output_path: Optional[Union[str, Path]] = None) -> Project:
    """
    Convenience function to extract data from a single CAD file.
    
    Args:
        file_path: Path to the CAD file
        project_name: Optional project name override
        level_name: Name of the building level
        output_path: Optional path to save extracted JSON
        
    Returns:
        Project object with extracted data
    """
    extractor = UnifiedExtractor()
    return extractor.extract_from_file(file_path, project_name, level_name, output_path)


def extract_from_directory(directory_path: Union[str, Path],
                          output_dir: Optional[Union[str, Path]] = None,
                          level_name: str = "Planta Baja") -> Dict[str, Project]:
    """
    Convenience function to extract data from all CAD files in a directory.
    
    Args:
        directory_path: Path to directory containing CAD files
        output_dir: Optional directory to save extracted JSON files
        level_name: Name of the building level
        
    Returns:
        Dictionary mapping file paths to Project objects
    """
    extractor = UnifiedExtractor()
    return extractor.extract_from_directory(directory_path, output_dir, level_name)


def analyze_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to analyze a CAD file.
    
    Args:
        file_path: Path to the CAD file
        
    Returns:
        Dictionary with file analysis information
    """
    extractor = UnifiedExtractor()
    return extractor.analyze_file(file_path)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified CAD file extraction for AEC Compliance Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from single file
  python -m src.extraction.unified_extractor --file "building.dwg" --output "building.json"
  
  # Extract from directory
  python -m src.extraction.unified_extractor --directory "data/blueprints/" --output-dir "data/extracted/"
  
  # Analyze file without extraction
  python -m src.extraction.unified_extractor --analyze "building.dwg"
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=Path,
        help='Single CAD file to process'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=Path,
        help='Directory containing CAD files'
    )
    input_group.add_argument(
        '--analyze', '-a',
        type=Path,
        help='Analyze CAD file without extraction'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (for single file)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory (for directory processing)'
    )
    
    # Optional parameters
    parser.add_argument(
        '--project', '-p',
        type=str,
        help='Project name override'
    )
    parser.add_argument(
        '--level', '-l',
        type=str,
        default="Planta Baja",
        help='Building level name'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.analyze:
            # Analyze file
            analysis = analyze_file(args.analyze)
            print(json.dumps(analysis, indent=2))
            
        elif args.file:
            # Process single file
            project = extract_from_file(
                args.file,
                project_name=args.project,
                level_name=args.level,
                output_path=args.output
            )
            
            print(f"✅ Extraction completed successfully!")
            print(f"   - Rooms: {len(project.rooms)}")
            print(f"   - Doors: {len(project.doors)}")
            print(f"   - Walls: {len(project.walls)}")
            print(f"   - Fire Equipment: {len(project.fire_equipment)}")
            print(f"   - Sectors: {len(project.sectors)}")
            
        elif args.directory:
            # Process directory
            projects = extract_from_directory(
                args.directory,
                output_dir=args.output_dir,
                level_name=args.level
            )
            
            print(f"✅ Directory processing completed!")
            print(f"   - Files processed: {len(projects)}")
            
            for file_path, project in projects.items():
                print(f"   - {Path(file_path).name}: {len(project.rooms)} rooms, {len(project.doors)} doors")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)
