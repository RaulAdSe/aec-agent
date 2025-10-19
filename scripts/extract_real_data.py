#!/usr/bin/env python3
"""
Real Data Extraction Script for AEC Compliance Agent.

This script demonstrates how to extract building data from real CAD files
(DWG/DXF and Revit) using the unified extraction pipeline.

Usage:
    python scripts/extract_real_data.py --input data/blueprints/ --output data/extracted/
    python scripts/extract_real_data.py --file "data/blueprints/cad/building.dwg" --output "building.json"
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from extraction.unified_extractor import UnifiedExtractor, analyze_file
from extraction.dwg_extractor import DWGExtractor
from extraction.revit_extractor import RevitExtractor
from schemas import Project


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


def analyze_cad_files(input_path: Path) -> None:
    """
    Analyze CAD files to understand their structure and content.
    
    Args:
        input_path: Path to file or directory to analyze
    """
    logger = logging.getLogger(__name__)
    
    if input_path.is_file():
        # Analyze single file
        logger.info(f"Analyzing file: {input_path}")
        try:
            analysis = analyze_file(input_path)
            print(f"\n📊 File Analysis: {input_path.name}")
            print("=" * 50)
            print(f"File Type: {analysis['file_type']}")
            print(f"File Size: {analysis['file_size_mb']:.2f} MB")
            print(f"Supported: {analysis['supported']}")
            
            if 'entity_counts' in analysis:
                print(f"Total Entities: {analysis['total_entities']}")
                print("\nEntity Types:")
                for entity_type, count in analysis['entity_counts'].items():
                    print(f"  {entity_type}: {count}")
                
                if analysis['layers']:
                    print(f"\nLayers ({len(analysis['layers'])}):")
                    for layer in analysis['layers'][:10]:  # Show first 10
                        print(f"  - {layer}")
                    if len(analysis['layers']) > 10:
                        print(f"  ... and {len(analysis['layers']) - 10} more")
                
                if analysis['blocks']:
                    print(f"\nBlocks ({len(analysis['blocks'])}):")
                    for block in analysis['blocks'][:10]:  # Show first 10
                        print(f"  - {block}")
                    if len(analysis['blocks']) > 10:
                        print(f"  ... and {len(analysis['blocks']) - 10} more")
            
            if 'analysis_error' in analysis:
                print(f"\n⚠️  Analysis Error: {analysis['analysis_error']}")
                
        except Exception as e:
            logger.error(f"Error analyzing {input_path}: {e}")
    
    elif input_path.is_dir():
        # Analyze all files in directory
        logger.info(f"Analyzing directory: {input_path}")
        
        # Find all CAD files
        cad_extensions = ['.dwg', '.dxf', '.rvt']
        cad_files = []
        
        for ext in cad_extensions:
            cad_files.extend(input_path.glob(f"*{ext}"))
            cad_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not cad_files:
            print(f"No CAD files found in {input_path}")
            return
        
        print(f"\n📊 Directory Analysis: {input_path}")
        print("=" * 50)
        print(f"Found {len(cad_files)} CAD files")
        
        for file_path in sorted(cad_files):
            try:
                analysis = analyze_file(file_path)
                print(f"\n📄 {file_path.name}")
                print(f"   Type: {analysis['file_type']}")
                print(f"   Size: {analysis['file_size_mb']:.2f} MB")
                print(f"   Supported: {analysis['supported']}")
                
                if 'total_entities' in analysis:
                    print(f"   Entities: {analysis['total_entities']}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")


def extract_single_file(file_path: Path, output_path: Optional[Path] = None, 
                       project_name: Optional[str] = None) -> bool:
    """
    Extract data from a single CAD file.
    
    Args:
        file_path: Path to CAD file
        output_path: Optional output JSON path
        project_name: Optional project name
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Extracting from: {file_path}")
        
        # Use unified extractor
        extractor = UnifiedExtractor()
        project = extractor.extract_from_file(
            file_path=file_path,
            project_name=project_name,
            level_name="Planta Baja",
            output_path=output_path
        )
        
        # Print extraction summary
        print(f"\n✅ Extraction completed: {file_path.name}")
        print("=" * 50)
        print(f"Project: {project.metadata.project_name}")
        print(f"Level: {project.metadata.level_name}")
        print(f"Extraction Date: {project.metadata.extraction_date}")
        print(f"\nBuilding Elements:")
        print(f"  🏠 Rooms: {len(project.rooms)}")
        print(f"  🚪 Doors: {len(project.doors)}")
        print(f"  🧱 Walls: {len(project.walls)}")
        print(f"  🔥 Fire Equipment: {len(project.fire_equipment)}")
        print(f"  🏢 Sectors: {len(project.sectors)}")
        
        # Show some details
        if project.rooms:
            print(f"\nRoom Details:")
            for room in project.rooms[:5]:  # Show first 5 rooms
                area_str = f" ({room.area:.1f} m²)" if room.area else ""
                print(f"  - {room.name}: {room.use_type or 'general'}{area_str}")
            if len(project.rooms) > 5:
                print(f"  ... and {len(project.rooms) - 5} more rooms")
        
        if project.doors:
            print(f"\nDoor Details:")
            for door in project.doors[:5]:  # Show first 5 doors
                egress_mark = " [EGRESS]" if door.is_egress else ""
                print(f"  - {door.id}: {door.width:.2f}m wide{egress_mark}")
            if len(project.doors) > 5:
                print(f"  ... and {len(project.doors) - 5} more doors")
        
        if project.fire_equipment:
            print(f"\nFire Equipment:")
            equipment_types = {}
            for eq in project.fire_equipment:
                equipment_types[eq.equipment_type] = equipment_types.get(eq.equipment_type, 0) + 1
            
            for eq_type, count in equipment_types.items():
                print(f"  - {eq_type}: {count}")
        
        if output_path:
            print(f"\n💾 Data saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {file_path}: {e}")
        print(f"❌ Extraction failed: {e}")
        return False


def extract_directory(input_dir: Path, output_dir: Path) -> int:
    """
    Extract data from all CAD files in a directory.
    
    Args:
        input_dir: Input directory containing CAD files
        output_dir: Output directory for JSON files
        
    Returns:
        Number of files successfully processed
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Processing directory: {input_dir}")
        
        # Use unified extractor
        extractor = UnifiedExtractor()
        projects = extractor.extract_from_directory(
            directory_path=input_dir,
            output_dir=output_dir,
            level_name="Planta Baja"
        )
        
        # Print summary
        print(f"\n✅ Directory processing completed!")
        print("=" * 50)
        print(f"Files processed: {len(projects)}")
        
        total_rooms = 0
        total_doors = 0
        total_walls = 0
        total_equipment = 0
        
        for file_path, project in projects.items():
            file_name = Path(file_path).name
            total_rooms += len(project.rooms)
            total_doors += len(project.doors)
            total_walls += len(project.walls)
            total_equipment += len(project.fire_equipment)
            
            print(f"\n📄 {file_name}")
            print(f"   Rooms: {len(project.rooms)}")
            print(f"   Doors: {len(project.doors)}")
            print(f"   Walls: {len(project.walls)}")
            print(f"   Fire Equipment: {len(project.fire_equipment)}")
        
        print(f"\n📊 Total Summary:")
        print(f"   Total Rooms: {total_rooms}")
        print(f"   Total Doors: {total_doors}")
        print(f"   Total Walls: {total_walls}")
        print(f"   Total Fire Equipment: {total_equipment}")
        
        return len(projects)
        
    except Exception as e:
        logger.error(f"Error processing directory {input_dir}: {e}")
        print(f"❌ Directory processing failed: {e}")
        return 0


def demonstrate_extraction_capabilities() -> None:
    """Demonstrate the extraction capabilities with example data."""
    print("\n🚀 AEC Compliance Agent - Extraction Capabilities")
    print("=" * 60)
    
    print("\n📋 Supported File Types:")
    print("  • DWG files (AutoCAD)")
    print("  • DXF files (AutoCAD Exchange)")
    print("  • RVT files (Revit) - requires Revit API")
    
    print("\n🏗️  Extracted Building Elements:")
    print("  • Rooms (boundaries, areas, use types)")
    print("  • Doors (width, type, fire rating, egress status)")
    print("  • Walls (length, thickness, fire rating, exterior/interior)")
    print("  • Fire Equipment (extinguishers, hydrants, alarms, etc.)")
    print("  • Fire Sectors (compartments, fire resistance)")
    
    print("\n🔧 Extraction Features:")
    print("  • Automatic file type detection")
    print("  • Intelligent entity recognition")
    print("  • Fire safety equipment classification")
    print("  • Building code compliance data extraction")
    print("  • JSON output with validation")
    
    print("\n📊 Output Format:")
    print("  • Structured JSON with Pydantic validation")
    print("  • Project metadata and building elements")
    print("  • Ready for compliance analysis")
    print("  • Compatible with RAG and agent systems")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract building data from real CAD files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze files to understand structure
  python scripts/extract_real_data.py --analyze data/blueprints/cad/building.dwg
  python scripts/extract_real_data.py --analyze data/blueprints/
  
  # Extract from single file
  python scripts/extract_real_data.py --file data/blueprints/cad/building.dwg --output building.json
  
  # Extract from directory
  python scripts/extract_real_data.py --input data/blueprints/ --output data/extracted/
  
  # Show capabilities
  python scripts/extract_real_data.py --demo
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
        '--input', '-i',
        type=Path,
        help='Input directory containing CAD files'
    )
    input_group.add_argument(
        '--analyze', '-a',
        type=Path,
        help='Analyze CAD file(s) without extraction'
    )
    input_group.add_argument(
        '--demo',
        action='store_true',
        help='Show extraction capabilities'
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        if args.demo:
            # Show capabilities
            demonstrate_extraction_capabilities()
            
        elif args.analyze:
            # Analyze files
            analyze_cad_files(args.analyze)
            
        elif args.file:
            # Process single file
            if not args.file.exists():
                print(f"❌ File not found: {args.file}")
                sys.exit(1)
            
            # Generate output path if not provided
            if not args.output:
                output_dir = Path("data/extracted")
                output_dir.mkdir(parents=True, exist_ok=True)
                args.output = output_dir / f"{args.file.stem}_extracted.json"
            
            success = extract_single_file(args.file, args.output, args.project)
            if not success:
                sys.exit(1)
            
        elif args.input:
            # Process directory
            if not args.input.exists():
                print(f"❌ Directory not found: {args.input}")
                sys.exit(1)
            
            # Generate output directory if not provided
            if not args.output_dir:
                args.output_dir = Path("data/extracted")
            
            success_count = extract_directory(args.input, args.output_dir)
            if success_count == 0:
                sys.exit(1)
        
        print(f"\n🎉 Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()