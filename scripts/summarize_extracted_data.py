#!/usr/bin/env python3
"""
Summarize Extracted CAD Data for AEC Compliance Agent.

This script provides a comprehensive summary of all extracted data from CAD files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from schemas import Project


def load_extracted_data(data_dir: Path) -> Dict[str, Project]:
    """Load all extracted data from JSON files."""
    projects = {}
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return projects
    
    json_files = list(data_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to Project object
            project = Project(**data)
            projects[json_file.stem] = project
            
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
    
    return projects


def summarize_project(project: Project) -> Dict[str, Any]:
    """Create a summary of a project."""
    summary = {
        "project_name": project.metadata.project_name,
        "source_file": project.metadata.source_file,
        "extraction_date": project.metadata.extraction_date,
        "building_use": project.metadata.building_use,
        "rooms": {
            "count": len(project.rooms),
            "types": {},
            "total_area": 0
        },
        "doors": {
            "count": len(project.doors),
            "types": {},
            "egress_count": 0,
            "fire_rated_count": 0
        },
        "walls": {
            "count": len(project.walls),
            "exterior_count": 0,
            "fire_rated_count": 0
        },
        "fire_equipment": {
            "count": len(project.fire_equipment),
            "types": {}
        },
        "sectors": {
            "count": len(project.sectors),
            "fire_resistance_types": {}
        }
    }
    
    # Analyze rooms
    for room in project.rooms:
        use_type = room.use_type or "general"
        summary["rooms"]["types"][use_type] = summary["rooms"]["types"].get(use_type, 0) + 1
        if room.area:
            summary["rooms"]["total_area"] += room.area
    
    # Analyze doors
    for door in project.doors:
        door_type = door.door_type or "single"
        summary["doors"]["types"][door_type] = summary["doors"]["types"].get(door_type, 0) + 1
        if door.is_egress:
            summary["doors"]["egress_count"] += 1
        if door.fire_rating:
            summary["doors"]["fire_rated_count"] += 1
    
    # Analyze walls
    for wall in project.walls:
        if wall.is_exterior:
            summary["walls"]["exterior_count"] += 1
        if wall.fire_rating:
            summary["walls"]["fire_rated_count"] += 1
    
    # Analyze fire equipment
    for eq in project.fire_equipment:
        eq_type = eq.equipment_type
        summary["fire_equipment"]["types"][eq_type] = summary["fire_equipment"]["types"].get(eq_type, 0) + 1
    
    # Analyze sectors
    for sector in project.sectors:
        fire_resistance = sector.fire_resistance or "Unknown"
        summary["sectors"]["fire_resistance_types"][fire_resistance] = summary["sectors"]["fire_resistance_types"].get(fire_resistance, 0) + 1
    
    return summary


def print_project_summary(project_name: str, summary: Dict[str, Any]):
    """Print a detailed summary of a project."""
    print(f"\n📊 Project: {summary['project_name']}")
    print("=" * 60)
    print(f"Source File: {Path(summary['source_file']).name}")
    print(f"Extraction Date: {summary['extraction_date']}")
    print(f"Building Use: {summary['building_use']}")
    
    # Rooms summary
    print(f"\n🏠 Rooms ({summary['rooms']['count']} total)")
    if summary['rooms']['count'] > 0:
        print(f"   Total Area: {summary['rooms']['total_area']:.1f} m²")
        print("   Room Types:")
        for room_type, count in summary['rooms']['types'].items():
            print(f"     - {room_type}: {count}")
    else:
        print("   No rooms extracted")
    
    # Doors summary
    print(f"\n🚪 Doors ({summary['doors']['count']} total)")
    if summary['doors']['count'] > 0:
        print(f"   Egress Doors: {summary['doors']['egress_count']}")
        print(f"   Fire-Rated Doors: {summary['doors']['fire_rated_count']}")
        print("   Door Types:")
        for door_type, count in summary['doors']['types'].items():
            print(f"     - {door_type}: {count}")
    else:
        print("   No doors extracted")
    
    # Walls summary
    print(f"\n🧱 Walls ({summary['walls']['count']} total)")
    if summary['walls']['count'] > 0:
        print(f"   Exterior Walls: {summary['walls']['exterior_count']}")
        print(f"   Fire-Rated Walls: {summary['walls']['fire_rated_count']}")
    else:
        print("   No walls extracted")
    
    # Fire equipment summary
    print(f"\n🔥 Fire Equipment ({summary['fire_equipment']['count']} total)")
    if summary['fire_equipment']['count'] > 0:
        print("   Equipment Types:")
        for eq_type, count in summary['fire_equipment']['types'].items():
            print(f"     - {eq_type}: {count}")
    else:
        print("   No fire equipment extracted")
    
    # Sectors summary
    print(f"\n🏢 Fire Sectors ({summary['sectors']['count']} total)")
    if summary['sectors']['count'] > 0:
        print("   Fire Resistance Types:")
        for fire_resistance, count in summary['sectors']['fire_resistance_types'].items():
            print(f"     - {fire_resistance}: {count}")
    else:
        print("   No fire sectors extracted")


def print_overall_summary(projects: Dict[str, Project]):
    """Print overall summary of all projects."""
    print("\n🎯 OVERALL SUMMARY")
    print("=" * 60)
    
    total_rooms = sum(len(project.rooms) for project in projects.values())
    total_doors = sum(len(project.doors) for project in projects.values())
    total_walls = sum(len(project.walls) for project in projects.values())
    total_equipment = sum(len(project.fire_equipment) for project in projects.values())
    total_sectors = sum(len(project.sectors) for project in projects.values())
    
    print(f"📁 Projects Processed: {len(projects)}")
    print(f"🏠 Total Rooms: {total_rooms}")
    print(f"🚪 Total Doors: {total_doors}")
    print(f"🧱 Total Walls: {total_walls}")
    print(f"🔥 Total Fire Equipment: {total_equipment}")
    print(f"🏢 Total Fire Sectors: {total_sectors}")
    
    # Equipment type summary
    all_equipment_types = {}
    for project in projects.values():
        for eq in project.fire_equipment:
            eq_type = eq.equipment_type
            all_equipment_types[eq_type] = all_equipment_types.get(eq_type, 0) + 1
    
    if all_equipment_types:
        print(f"\n🔥 Fire Equipment Summary:")
        for eq_type, count in sorted(all_equipment_types.items()):
            print(f"   - {eq_type}: {count}")
    
    # Room type summary
    all_room_types = {}
    for project in projects.values():
        for room in project.rooms:
            room_type = room.use_type or "general"
            all_room_types[room_type] = all_room_types.get(room_type, 0) + 1
    
    if all_room_types:
        print(f"\n🏠 Room Types Summary:")
        for room_type, count in sorted(all_room_types.items()):
            print(f"   - {room_type}: {count}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize extracted CAD data")
    parser.add_argument('--data-dir', '-d', type=Path, default=Path("data/extracted"), 
                       help='Directory containing extracted JSON files')
    parser.add_argument('--project', '-p', type=str, help='Specific project to summarize')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("📊 AEC Compliance Agent - Extracted Data Summary")
    print("=" * 60)
    
    # Load all extracted data
    projects = load_extracted_data(args.data_dir)
    
    if not projects:
        print("❌ No extracted data found")
        sys.exit(1)
    
    print(f"📁 Found {len(projects)} extracted projects")
    
    # Summarize specific project or all projects
    if args.project:
        if args.project in projects:
            summary = summarize_project(projects[args.project])
            print_project_summary(args.project, summary)
        else:
            print(f"❌ Project '{args.project}' not found")
            print(f"Available projects: {', '.join(projects.keys())}")
            sys.exit(1)
    else:
        # Summarize all projects
        for project_name, project in projects.items():
            summary = summarize_project(project)
            print_project_summary(project_name, summary)
        
        # Print overall summary
        print_overall_summary(projects)
    
    print(f"\n✅ Summary completed successfully!")


if __name__ == '__main__':
    main()
