#!/usr/bin/env python3
"""
Script to extract real data from DWG files and test agent tools.

This script extracts data from the existing DWG files in the data directory
and then tests the agent tools against the extracted data.
"""

import sys
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.extraction.dwg_extractor import extract_from_dwg, save_to_json
from src.agent.tools import (
    load_project_data, 
    get_room_info, 
    get_door_info, 
    list_all_doors,
    check_door_width_compliance,
    calculate_egress_distance,
    get_project_summary
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def extract_dwg_files():
    """Extract data from all DWG files in the data directory."""
    data_dir = project_root / "data"
    blueprints_dir = data_dir / "blueprints" / "cad"
    
    if not blueprints_dir.exists():
        print(f"❌ Blueprints directory not found: {blueprints_dir}")
        return []
    
    # Find all DWG files
    dwg_files = list(blueprints_dir.glob("*.dwg"))
    
    if not dwg_files:
        print(f"❌ No DWG files found in {blueprints_dir}")
        return []
    
    print(f"🔍 Found {len(dwg_files)} DWG files:")
    for dwg_file in dwg_files:
        print(f"   - {dwg_file.name}")
    
    extracted_files = []
    
    for dwg_file in dwg_files:
        print(f"\n📦 Extracting from {dwg_file.name}...")
        
        try:
            # Extract data
            project = extract_from_dwg(dwg_file)
            
            # Create output path
            output_dir = data_dir / "extracted"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"{dwg_file.stem}_extracted.json"
            
            # Save extracted data
            save_to_json(project, output_file)
            
            print(f"✅ Successfully extracted:")
            print(f"   - Rooms: {len(project.get_all_rooms())}")
            print(f"   - Doors: {len(project.get_all_doors())}")
            print(f"   - Walls: {len(project.get_all_walls())}")
            print(f"   - Output: {output_file}")
            
            extracted_files.append(output_file)
            
        except Exception as e:
            print(f"❌ Error extracting {dwg_file.name}: {e}")
            continue
    
    return extracted_files


def test_agent_tools(extracted_file: Path):
    """Test agent tools with extracted data."""
    print(f"\n🧪 Testing agent tools with {extracted_file.name}...")
    print("=" * 60)
    
    try:
        # Load project data
        project = load_project_data(extracted_file)
        print(f"✅ Loaded project: {project.metadata.project_name}")
        
        # Test 1: Project summary
        print("\n1. Project Summary:")
        summary = get_project_summary()
        print(f"   - Total rooms: {summary['total_rooms']}")
        print(f"   - Total doors: {summary['total_doors']}")
        print(f"   - Total area: {summary['total_area_sqm']:.1f} sqm")
        print(f"   - Room uses: {summary['room_uses']}")
        
        # Test 2: Get room info
        rooms = project.get_all_rooms()
        if rooms:
            room_id = rooms[0].id
            print(f"\n2. Room Info (Room {room_id}):")
            room_info = get_room_info(room_id)
            if "error" not in room_info:
                print(f"   - Name: {room_info['name']}")
                print(f"   - Area: {room_info['area_sqm']} sqm")
                print(f"   - Use: {room_info['use']}")
                print(f"   - Occupancy capacity: {room_info['calculated_occupancy_capacity']}")
            else:
                print(f"   ❌ Error: {room_info['error']}")
        
        # Test 3: Get door info
        doors = project.get_all_doors()
        if doors:
            door_id = doors[0].id
            print(f"\n3. Door Info (Door {door_id}):")
            door_info = get_door_info(door_id)
            if "error" not in door_info:
                print(f"   - Width: {door_info['width_mm']} mm")
                print(f"   - Clear width: {door_info['clear_width_mm']} mm")
                print(f"   - Type: {door_info['door_type']}")
                print(f"   - Emergency exit: {door_info['is_emergency_exit']}")
            else:
                print(f"   ❌ Error: {door_info['error']}")
        
        # Test 4: List all doors
        print(f"\n4. All Doors ({len(doors)} total):")
        door_list = list_all_doors()
        if door_list and "error" not in door_list[0]:
            for door in door_list[:3]:  # Show first 3
                print(f"   - {door['id']}: {door['width_mm']}mm, {door['door_type']}")
        else:
            print(f"   ❌ Error: {door_list[0].get('error', 'Unknown error')}")
        
        # Test 5: Check door compliance
        if doors:
            door_id = doors[0].id
            print(f"\n5. Door Compliance Check (Door {door_id}):")
            compliance = check_door_width_compliance(door_id)
            if "error" not in compliance:
                print(f"   - Clear width: {compliance['clear_width_mm']} mm")
                print(f"   - Required: {compliance['required_width_mm']} mm")
                print(f"   - Status: {compliance['compliance_status']}")
                print(f"   - Message: {compliance['message']}")
            else:
                print(f"   ❌ Error: {compliance['error']}")
        
        # Test 6: Calculate egress distance
        if rooms:
            room_id = rooms[0].id
            print(f"\n6. Egress Distance (Room {room_id}):")
            egress = calculate_egress_distance(room_id)
            if "error" not in egress:
                print(f"   - Distance: {egress['egress_distance_m']:.1f} m")
                print(f"   - Max allowed: {egress['max_allowed_distance_m']} m")
                print(f"   - Status: {egress['compliance_status']}")
                print(f"   - Accessible: {egress['is_accessible']}")
            else:
                print(f"   ⚠️  Warning: {egress['error']}")
        
        print(f"\n✅ Agent tools testing completed for {extracted_file.name}")
        
    except Exception as e:
        print(f"❌ Error testing agent tools: {e}")


def main():
    """Main function."""
    setup_logging()
    
    print("🏗️  AEC Compliance Agent - Real Data Extraction & Testing")
    print("=" * 60)
    
    # Step 1: Extract data from DWG files
    print("\n📦 Step 1: Extracting data from DWG files...")
    extracted_files = extract_dwg_files()
    
    if not extracted_files:
        print("❌ No files were successfully extracted. Exiting.")
        return 1
    
    # Step 2: Test agent tools with extracted data
    print(f"\n🧪 Step 2: Testing agent tools with {len(extracted_files)} extracted files...")
    
    for extracted_file in extracted_files:
        test_agent_tools(extracted_file)
    
    print(f"\n🎉 All tests completed!")
    print(f"   - Extracted files: {len(extracted_files)}")
    print(f"   - Agent tools tested successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())
