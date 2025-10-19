#!/usr/bin/env python3
"""
CAD File Extraction Script for AEC Compliance Agent.

This script processes DWG/DXF files from the blueprints directory and extracts
building data using the DWGExtractor class. The extracted data is saved as JSON
files in the data/extracted/ directory.

Usage:
    python scripts/extract_cad_files.py --input data/blueprints/cad/ --output data/extracted/
    python scripts/extract_cad_files.py --file "data/blueprints/cad/I01.4 PCI - EXTINCIÓN AUTOMÁTICA.dwg"
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from extraction.dwg_extractor import DWGExtractor


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('extraction.log')
        ]
    )


def get_project_name_from_filename(filename: str) -> str:
    """
    Extract a meaningful project name from the CAD filename.
    
    Args:
        filename: Name of the CAD file
        
    Returns:
        Cleaned project name
    """
    # Remove file extension
    name = Path(filename).stem
    
    # Clean up common CAD file naming patterns
    name = name.replace('I01.4 PCI - ', '').replace('I01.6 PCI - ', '')
    name = name.replace('EXTINCIÓN AUTOMÁTICA', 'Fire Suppression System')
    name = name.replace('SECTORIZACIÓN', 'Fire Sectorization')
    
    return name if name else "Building Project"


def process_single_file(file_path: Path, output_dir: Path, project_name: Optional[str] = None) -> bool:
    """
    Process a single CAD file and extract building data.
    
    Args:
        file_path: Path to the CAD file
        output_dir: Directory to save extracted JSON
        project_name: Optional project name override
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input file
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if file_path.suffix.lower() not in ['.dwg', '.dxf']:
            logger.warning(f"Skipping non-CAD file: {file_path}")
            return False
        
        # Generate project name if not provided
        if not project_name:
            project_name = get_project_name_from_filename(file_path.name)
        
        # Create output filename
        output_filename = f"{file_path.stem}_extracted.json"
        output_path = output_dir / output_filename
        
        logger.info(f"Processing file: {file_path}")
        logger.info(f"Project name: {project_name}")
        logger.info(f"Output path: {output_path}")
        
        # Extract data using DWGExtractor
        extractor = DWGExtractor()
        project = extractor.extract_from_file(
            file_path=file_path,
            project_name=project_name,
            level_name="Planta Baja"  # Ground floor
        )
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save extracted data to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                project.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
                default=str
            )
        
        # Log extraction summary
        logger.info(f"Extraction completed successfully!")
        logger.info(f"  - Rooms: {len(project.rooms)}")
        logger.info(f"  - Doors: {len(project.doors)}")
        logger.info(f"  - Walls: {len(project.walls)}")
        logger.info(f"  - Fire Equipment: {len(project.fire_equipment)}")
        logger.info(f"  - Sectors: {len(project.sectors)}")
        logger.info(f"  - Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        logger.exception("Full traceback:")
        return False


def process_directory(input_dir: Path, output_dir: Path) -> int:
    """
    Process all CAD files in a directory.
    
    Args:
        input_dir: Directory containing CAD files
        output_dir: Directory to save extracted JSON files
        
    Returns:
        Number of files successfully processed
    """
    logger = logging.getLogger(__name__)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 0
    
    # Find all CAD files
    cad_extensions = ['.dwg', '.dxf']
    cad_files = []
    
    for ext in cad_extensions:
        cad_files.extend(input_dir.glob(f"*{ext}"))
        cad_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not cad_files:
        logger.warning(f"No CAD files found in: {input_dir}")
        return 0
    
    logger.info(f"Found {len(cad_files)} CAD files to process")
    
    # Process each file
    success_count = 0
    for file_path in cad_files:
        if process_single_file(file_path, output_dir):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(cad_files)} files")
    return success_count


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract building data from CAD files (DWG/DXF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CAD files in directory
  python scripts/extract_cad_files.py --input data/blueprints/cad/ --output data/extracted/
  
  # Process specific file
  python scripts/extract_cad_files.py --file "data/blueprints/cad/I01.4 PCI - EXTINCIÓN AUTOMÁTICA.dwg" --output data/extracted/
  
  # Process with custom project name
  python scripts/extract_cad_files.py --file "building.dwg" --output data/extracted/ --project "My Building"
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input', '-i',
        type=Path,
        help='Input directory containing CAD files'
    )
    input_group.add_argument(
        '--file', '-f',
        type=Path,
        help='Single CAD file to process'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/extracted'),
        help='Output directory for extracted JSON files (default: data/extracted)'
    )
    
    # Optional parameters
    parser.add_argument(
        '--project', '-p',
        type=str,
        help='Project name override (for single file processing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting CAD file extraction")
    logger.info(f"Arguments: {vars(args)}")
    
    # Process files
    success_count = 0
    
    if args.file:
        # Process single file
        if process_single_file(args.file, args.output, args.project):
            success_count = 1
    else:
        # Process directory
        success_count = process_directory(args.input, args.output)
    
    # Final summary
    if success_count > 0:
        logger.info(f"Extraction completed successfully! Processed {success_count} file(s)")
        print(f"\nExtraction completed successfully!")
        print(f"Processed {success_count} file(s)")
        print(f"Output directory: {args.output.absolute()}")
    else:
        logger.error("No files were processed successfully")
        print("\nNo files were processed successfully. Check the logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()