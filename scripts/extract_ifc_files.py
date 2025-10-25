#!/usr/bin/env python3
"""
IFC File Extraction CLI Script

This script provides a command-line interface for extracting building elements
from IFC files and converting them to structured JSON format.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from extraction.ifc_extractor import extract_from_ifc, save_to_json
from extraction.unified_extractor import analyze_file


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def extract_single_file(input_file: Path, output_file: Path = None, verbose: bool = False):
    """Extract data from a single IFC file."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    if not input_file.suffix.lower() == '.ifc':
        logger.error(f"Input file must be an IFC file (.ifc), got: {input_file.suffix}")
        return False
    
    try:
        logger.info(f"Extracting from IFC file: {input_file}")
        
        # Extract project data
        project = extract_from_ifc(input_file)
        
        # Determine output path
        if output_file is None:
            output_file = input_file.with_suffix('.json')
        
        # Save to JSON
        save_to_json(project, output_file)
        
        # Print summary
        print(f"✅ IFC extraction completed successfully!")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   - Levels: {len(project.levels)}")
        print(f"   - Rooms: {len(project.get_all_rooms())}")
        print(f"   - Doors: {len(project.get_all_doors())}")
        print(f"   - Walls: {len(project.get_all_walls())}")
        print(f"   - Total Area: {project.metadata.total_area:.1f} m²")
        
        return True
        
    except Exception as e:
        logger.error(f"Error extracting from {input_file}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def extract_directory(input_dir: Path, output_dir: Path = None, verbose: bool = False):
    """Extract data from all IFC files in a directory."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return False
    
    # Find all IFC files
    ifc_files = list(input_dir.glob('*.ifc')) + list(input_dir.glob('*.IFC'))
    
    if not ifc_files:
        logger.warning(f"No IFC files found in directory: {input_dir}")
        return True
    
    logger.info(f"Found {len(ifc_files)} IFC files in {input_dir}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = input_dir / 'extracted'
    
    output_dir.mkdir(exist_ok=True)
    
    # Process each file
    successful = 0
    failed = 0
    
    for ifc_file in ifc_files:
        output_file = output_dir / f"{ifc_file.stem}_extracted.json"
        
        logger.info(f"Processing: {ifc_file.name}")
        
        if extract_single_file(ifc_file, output_file, verbose):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\\n✅ Directory processing completed!")
    print(f"   Processed: {successful + failed} files")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Output directory: {output_dir}")
    
    return failed == 0


def analyze_ifc_file(input_file: Path, verbose: bool = False):
    """Analyze an IFC file without extracting data."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    try:
        logger.info(f"Analyzing IFC file: {input_file}")
        
        # Analyze file
        analysis = analyze_file(input_file)
        
        # Print analysis in pretty format
        print(json.dumps(analysis, indent=2))
        
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing {input_file}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Extract building elements from IFC files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from single file
  python extract_ifc_files.py -f building.ifc

  # Extract with custom output
  python extract_ifc_files.py -f building.ifc -o building_data.json

  # Extract from directory
  python extract_ifc_files.py -d /path/to/ifc/files/

  # Extract to specific output directory
  python extract_ifc_files.py -d /path/to/ifc/files/ -o /path/to/output/

  # Analyze file without extraction
  python extract_ifc_files.py -a building.ifc

  # Verbose output
  python extract_ifc_files.py -f building.ifc -v
        """
    )
    
    # Main action (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '-f', '--file',
        type=Path,
        help='Extract from single IFC file'
    )
    action_group.add_argument(
        '-d', '--directory',
        type=Path,
        help='Extract from all IFC files in directory'
    )
    action_group.add_argument(
        '-a', '--analyze',
        type=Path,
        help='Analyze IFC file without extraction'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file/directory path'
    )
    
    # Other options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output and debugging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute based on action
    success = False
    
    try:
        if args.file:
            success = extract_single_file(args.file, args.output, args.verbose)
        elif args.directory:
            success = extract_directory(args.directory, args.output, args.verbose)
        elif args.analyze:
            success = analyze_ifc_file(args.analyze, args.verbose)
    
    except KeyboardInterrupt:
        print("\\n❌ Operation cancelled by user")
        return 1
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())