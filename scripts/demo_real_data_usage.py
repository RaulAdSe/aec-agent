#!/usr/bin/env python3
"""
Demo: Real CAD Data Usage for AEC Compliance Agent.

This script demonstrates how to use the extracted real data from CAD files
for building code compliance analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from schemas import Project
from extraction.unified_extractor import UnifiedExtractor


def load_project_data(json_file: Path) -> Project:
    """Load project data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Project(**data)


def analyze_fire_safety_compliance(project: Project) -> Dict[str, Any]:
    """Analyze fire safety compliance for a project."""
    analysis = {
        "project_name": project.metadata.project_name,
        "compliance_issues": [],
        "recommendations": [],
        "fire_equipment_coverage": {},
        "egress_analysis": {},
        "fire_resistance_analysis": {}
    }
    
    # Analyze fire equipment coverage
    total_area = sum(room.area for room in project.rooms if room.area)
    if total_area > 0:
        extinguisher_count = len([eq for eq in project.fire_equipment if eq.equipment_type == "extinguisher"])
        hydrant_count = len([eq for eq in project.fire_equipment if eq.equipment_type == "hydrant"])
        
        analysis["fire_equipment_coverage"] = {
            "total_area": total_area,
            "extinguisher_count": extinguisher_count,
            "hydrant_count": hydrant_count,
            "extinguisher_coverage": extinguisher_count * 15.0,  # 15m radius per extinguisher
            "hydrant_coverage": hydrant_count * 25.0  # 25m radius per hydrant
        }
        
        # Check if coverage is adequate
        if extinguisher_count < (total_area / 200):  # 1 extinguisher per 200 m²
            analysis["compliance_issues"].append(
                f"Insufficient fire extinguishers: {extinguisher_count} for {total_area:.1f} m²"
            )
            analysis["recommendations"].append(
                f"Add {int(total_area / 200) - extinguisher_count} more fire extinguishers"
            )
    
    # Analyze egress doors
    egress_doors = [door for door in project.doors if door.is_egress]
    analysis["egress_analysis"] = {
        "egress_door_count": len(egress_doors),
        "total_door_count": len(project.doors),
        "egress_door_widths": [door.width for door in egress_doors]
    }
    
    # Check egress door widths
    for door in egress_doors:
        if door.width < 0.8:  # Minimum 80cm for egress
            analysis["compliance_issues"].append(
                f"Egress door {door.id} too narrow: {door.width:.2f}m (minimum 0.8m)"
            )
    
    # Analyze fire resistance
    fire_rated_walls = [wall for wall in project.walls if wall.fire_rating]
    fire_rated_doors = [door for door in project.doors if door.fire_rating]
    
    analysis["fire_resistance_analysis"] = {
        "fire_rated_walls": len(fire_rated_walls),
        "fire_rated_doors": len(fire_rated_doors),
        "wall_fire_ratings": [wall.fire_rating for wall in fire_rated_walls],
        "door_fire_ratings": [door.fire_rating for door in fire_rated_doors]
    }
    
    return analysis


def analyze_room_compliance(project: Project) -> Dict[str, Any]:
    """Analyze room compliance for a project."""
    analysis = {
        "project_name": project.metadata.project_name,
        "room_analysis": {},
        "compliance_issues": [],
        "recommendations": []
    }
    
    # Analyze each room
    for room in project.rooms:
        room_analysis = {
            "name": room.name,
            "use_type": room.use_type,
            "area": room.area,
            "has_emergency_lighting": room.has_emergency_lighting,
            "has_fire_detection": room.has_fire_detection,
            "issues": []
        }
        
        # Check area requirements
        if room.area:
            if room.use_type == "office" and room.area < 6.0:  # Minimum 6 m² per person
                room_analysis["issues"].append("Office area below minimum requirement")
            elif room.use_type == "meeting_room" and room.area < 10.0:
                room_analysis["issues"].append("Meeting room area below minimum requirement")
        
        # Check emergency lighting
        if room.use_type in ["corridor", "stairs", "lobby"] and not room.has_emergency_lighting:
            room_analysis["issues"].append("Emergency lighting required for circulation areas")
        
        # Check fire detection
        if room.use_type in ["storage", "technical"] and not room.has_fire_detection:
            room_analysis["issues"].append("Fire detection required for high-risk areas")
        
        analysis["room_analysis"][room.id] = room_analysis
        
        if room_analysis["issues"]:
            analysis["compliance_issues"].extend([f"{room.name}: {issue}" for issue in room_analysis["issues"]])
    
    return analysis


def generate_compliance_report(project: Project) -> str:
    """Generate a comprehensive compliance report."""
    fire_analysis = analyze_fire_safety_compliance(project)
    room_analysis = analyze_room_compliance(project)
    
    report = f"""
🏗️ BUILDING COMPLIANCE REPORT
{'=' * 60}

Project: {project.metadata.project_name}
Source File: {Path(project.metadata.source_file).name}
Analysis Date: {project.metadata.extraction_date}

📊 BUILDING OVERVIEW
{'=' * 30}
• Total Rooms: {len(project.rooms)}
• Total Doors: {len(project.doors)}
• Total Walls: {len(project.walls)}
• Fire Equipment: {len(project.fire_equipment)}
• Fire Sectors: {len(project.sectors)}

🔥 FIRE SAFETY ANALYSIS
{'=' * 30}
"""
    
    # Fire equipment analysis
    if fire_analysis["fire_equipment_coverage"]:
        coverage = fire_analysis["fire_equipment_coverage"]
        report += f"• Total Area: {coverage['total_area']:.1f} m²\n"
        report += f"• Fire Extinguishers: {coverage['extinguisher_count']}\n"
        report += f"• Fire Hydrants: {coverage['hydrant_count']}\n"
        report += f"• Extinguisher Coverage: {coverage['extinguisher_coverage']:.1f} m²\n"
        report += f"• Hydrant Coverage: {coverage['hydrant_coverage']:.1f} m²\n"
    
    # Egress analysis
    egress = fire_analysis["egress_analysis"]
    report += f"• Egress Doors: {egress['egress_door_count']}/{egress['total_door_count']}\n"
    if egress["egress_door_widths"]:
        min_width = min(egress["egress_door_widths"])
        report += f"• Minimum Egress Width: {min_width:.2f}m\n"
    
    # Fire resistance analysis
    fire_res = fire_analysis["fire_resistance_analysis"]
    report += f"• Fire-Rated Walls: {fire_res['fire_rated_walls']}\n"
    report += f"• Fire-Rated Doors: {fire_res['fire_rated_doors']}\n"
    
    # Compliance issues
    if fire_analysis["compliance_issues"]:
        report += f"\n❌ FIRE SAFETY ISSUES:\n"
        for issue in fire_analysis["compliance_issues"]:
            report += f"  • {issue}\n"
    
    if fire_analysis["recommendations"]:
        report += f"\n💡 FIRE SAFETY RECOMMENDATIONS:\n"
        for rec in fire_analysis["recommendations"]:
            report += f"  • {rec}\n"
    
    # Room analysis
    report += f"\n🏠 ROOM ANALYSIS\n{'=' * 30}\n"
    
    for room_id, room_data in room_analysis["room_analysis"].items():
        area_str = f"{room_data['area']:.1f}" if room_data['area'] else "Unknown"
        report += f"• {room_data['name']} ({room_data['use_type']}): {area_str} m²\n"
        if room_data['issues']:
            for issue in room_data['issues']:
                report += f"  - {issue}\n"
    
    # Overall compliance status
    total_issues = len(fire_analysis["compliance_issues"]) + len(room_analysis["compliance_issues"])
    
    report += f"\n📋 COMPLIANCE STATUS\n{'=' * 30}\n"
    if total_issues == 0:
        report += "✅ COMPLIANT: No major compliance issues found\n"
    else:
        report += f"⚠️  NON-COMPLIANT: {total_issues} issues found\n"
    
    return report


def main():
    """Main function to demonstrate real data usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo real CAD data usage")
    parser.add_argument('--data-dir', '-d', type=Path, default=Path("data/extracted"),
                       help='Directory containing extracted JSON files')
    parser.add_argument('--project', '-p', type=str, help='Specific project to analyze')
    parser.add_argument('--output', '-o', type=Path, help='Output report file')
    
    args = parser.parse_args()
    
    print("🏗️ AEC Compliance Agent - Real Data Usage Demo")
    print("=" * 60)
    
    # Find JSON files
    if not args.data_dir.exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    json_files = list(args.data_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No extracted data found")
        sys.exit(1)
    
    # Load and analyze projects
    projects = {}
    for json_file in json_files:
        try:
            project = load_project_data(json_file)
            projects[json_file.stem] = project
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
    
    if not projects:
        print("❌ No valid projects found")
        sys.exit(1)
    
    print(f"📁 Found {len(projects)} projects to analyze")
    
    # Analyze specific project or all projects
    if args.project:
        if args.project in projects:
            project = projects[args.project]
            report = generate_compliance_report(project)
            print(report)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"💾 Report saved to: {args.output}")
        else:
            print(f"❌ Project '{args.project}' not found")
            print(f"Available projects: {', '.join(projects.keys())}")
            sys.exit(1)
    else:
        # Analyze all projects
        for project_name, project in projects.items():
            print(f"\n🔍 Analyzing: {project_name}")
            report = generate_compliance_report(project)
            print(report)
            
            if args.output:
                output_file = args.output.parent / f"{project_name}_report.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"💾 Report saved to: {output_file}")
    
    print(f"\n✅ Analysis completed successfully!")


if __name__ == '__main__':
    main()
