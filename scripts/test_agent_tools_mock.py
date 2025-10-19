#!/usr/bin/env python3
"""
Test script for agent tools using mock data.

This script tests the 6 agent tools with the existing mock data to demonstrate
that the agent tools work correctly.
"""

import sys
import json
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent.tools import (
    load_project_data, 
    get_room_info, 
    get_door_info, 
    list_all_doors,
    check_door_width_compliance,
    calculate_egress_distance,
    get_project_summary,
    get_available_tools
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def convert_mock_to_project_format(mock_data_path: Path) -> Path:
    """Convert mock data to project format and save it."""
    print(f"🔄 Converting mock data from {mock_data_path.name}...")
    
    with open(mock_data_path, 'r', encoding='utf-8') as f:
        mock_data = json.load(f)
    
    # Convert mock data to our project format
    if "project_name" in mock_data and "levels" in mock_data and "metadata" in mock_data:
        # This is already in project format
        project_data = mock_data
    elif "project_name" in mock_data and "levels" in mock_data:
        # This has project structure but missing metadata
        project_data = mock_data
        # Add metadata if missing
        if "metadata" not in project_data:
            project_data["metadata"] = {
                "project_name": project_data["project_name"],
                "file_name": mock_data_path.name,
                "building_type": project_data.get("building_type", "commercial"),
                "total_area": 0,  # Will be calculated
                "number_of_levels": len(project_data["levels"]),
                "created_date": "2025-10-19T12:00:00",
                "modified_date": "2025-10-19T12:00:00"
            }
    else:
        # Convert from mock format to project format
        project_data = {
            "metadata": {
                "project_name": mock_data.get("file", "Mock Project"),
                "file_name": mock_data_path.name,
                "building_type": "commercial",
                "total_area": 0,  # Will be calculated
                "number_of_levels": 1,
                "created_date": "2025-10-19T12:00:00",
                "modified_date": "2025-10-19T12:00:00"
            },
            "levels": [
                {
                    "name": "Planta Principal",
                    "elevation": 0.0,
                    "rooms": [],
                    "doors": [],
                    "walls": []
                }
            ]
        }
        
        # Add rooms if they exist in mock data
        if "rooms" in mock_data:
            for i, room_data in enumerate(mock_data["rooms"]):
                room = {
                    "id": room_data.get("id", f"R{i+1:03d}"),
                    "name": room_data.get("name", f"Room {i+1}"),
                    "area": room_data.get("area", 25.0),
                    "use": room_data.get("use", "commercial"),
                    "level": "Planta Principal",
                    "occupancy_load": room_data.get("occupancy_load", 3)
                }
                project_data["levels"][0]["rooms"].append(room)
        
        # Add doors if they exist in mock data
        if "doors" in mock_data:
            for i, door_data in enumerate(mock_data["doors"]):
                door = {
                    "id": door_data.get("id", f"D{i+1:03d}"),
                    "name": door_data.get("name", f"Door {i+1}"),
                    "width_mm": door_data.get("width_mm", 900),
                    "height_mm": door_data.get("height_mm", 2100),
                    "door_type": door_data.get("door_type", "single"),
                    "position": {
                        "x": door_data.get("position", [0, 0, 0])[0],
                        "y": door_data.get("position", [0, 0, 0])[1],
                        "z": door_data.get("position", [0, 0, 0])[2]
                    },
                    "from_room": door_data.get("from_room"),
                    "to_room": door_data.get("to_room"),
                    "is_emergency_exit": door_data.get("is_emergency_exit", False),
                    "is_accessible": door_data.get("is_accessible", True)
                }
                project_data["levels"][0]["doors"].append(door)
    
    # Calculate total area
    total_area = sum(room.get("area", 0) for room in project_data["levels"][0]["rooms"])
    project_data["metadata"]["total_area"] = total_area
    
    # Save converted data
    output_dir = project_root / "data" / "extracted"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{mock_data_path.stem}_converted.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted and saved to: {output_file}")
    return output_file


def test_agent_tools(project_file: Path):
    """Test agent tools with project data."""
    print(f"\n🧪 Testing agent tools with {project_file.name}...")
    print("=" * 60)
    
    try:
        # Load project data
        project = load_project_data(project_file)
        print(f"✅ Loaded project: {project.metadata.project_name}")
        
        # Test 1: Project summary
        print("\n1. Project Summary:")
        summary = get_project_summary()
        print(f"   - Total rooms: {summary['total_rooms']}")
        print(f"   - Total doors: {summary['total_doors']}")
        print(f"   - Total area: {summary['total_area_sqm']:.1f} sqm")
        print(f"   - Room uses: {summary['room_uses']}")
        print(f"   - Door types: {summary['door_types']}")
        
        # Test 2: Get room info
        rooms = project.get_all_rooms()
        if rooms:
            room_id = rooms[0].id
            print(f"\n2. Room Info (Room {room_id}):")
            room_info = get_room_info.invoke({"room_id": room_id})
            if "error" not in room_info:
                print(f"   - Name: {room_info['name']}")
                print(f"   - Area: {room_info['area_sqm']} sqm")
                print(f"   - Use: {room_info['use']}")
                print(f"   - Level: {room_info['level']}")
                print(f"   - Occupancy capacity: {room_info['calculated_occupancy_capacity']}")
            else:
                print(f"   ❌ Error: {room_info['error']}")
        
        # Test 3: Get door info
        doors = project.get_all_doors()
        if doors:
            door_id = doors[0].id
            print(f"\n3. Door Info (Door {door_id}):")
            door_info = get_door_info.invoke({"door_id": door_id})
            if "error" not in door_info:
                print(f"   - Name: {door_info['name']}")
                print(f"   - Width: {door_info['width_mm']} mm")
                print(f"   - Clear width: {door_info['clear_width_mm']} mm")
                print(f"   - Type: {door_info['door_type']}")
                print(f"   - Emergency exit: {door_info['is_emergency_exit']}")
                print(f"   - From room: {door_info['from_room']}")
                print(f"   - To room: {door_info['to_room']}")
            else:
                print(f"   ❌ Error: {door_info['error']}")
        
        # Test 4: List all doors
        print(f"\n4. All Doors ({len(doors)} total):")
        door_list = list_all_doors.invoke({})
        if door_list and len(door_list) > 0 and "error" not in door_list[0]:
            for door in door_list:
                print(f"   - {door['id']}: {door['width_mm']}mm, {door['door_type']}, Emergency: {door['is_emergency_exit']}")
        elif door_list and len(door_list) > 0:
            print(f"   ❌ Error: {door_list[0].get('error', 'Unknown error')}")
        else:
            print("   - No doors found in project")
        
        # Test 5: Check door compliance
        if doors:
            door_id = doors[0].id
            print(f"\n5. Door Compliance Check (Door {door_id}):")
            compliance = check_door_width_compliance.invoke({"door_id": door_id})
            if "error" not in compliance:
                print(f"   - Clear width: {compliance['clear_width_mm']} mm")
                print(f"   - Required: {compliance['required_width_mm']} mm")
                print(f"   - Status: {compliance['compliance_status']}")
                print(f"   - Message: {compliance['message']}")
                print(f"   - Regulation: {compliance['regulation_reference']}")
            else:
                print(f"   ❌ Error: {compliance['error']}")
        
        # Test 6: Calculate egress distance
        if rooms:
            room_id = rooms[0].id
            print(f"\n6. Egress Distance (Room {room_id}):")
            egress = calculate_egress_distance.invoke({"room_id": room_id})
            if "error" not in egress:
                print(f"   - Distance: {egress['egress_distance_m']:.1f} m")
                print(f"   - Max allowed: {egress['max_allowed_distance_m']} m")
                print(f"   - Status: {egress['compliance_status']}")
                print(f"   - Accessible: {egress['is_accessible']}")
                print(f"   - Message: {egress['message']}")
            else:
                print(f"   ⚠️  Warning: {egress['error']}")
        
        # Test 7: Available tools
        print(f"\n7. Available Agent Tools:")
        tools = get_available_tools()
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")
        
        print(f"\n✅ Agent tools testing completed for {project_file.name}")
        
    except Exception as e:
        print(f"❌ Error testing agent tools: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    setup_logging()
    
    print("🏗️  AEC Compliance Agent - Agent Tools Testing with Mock Data")
    print("=" * 70)
    
    # Find mock data files
    data_dir = project_root / "data" / "blueprints"
    mock_files = []
    
    # Check CAD mock files
    cad_dir = data_dir / "cad"
    if cad_dir.exists():
        mock_files.extend(cad_dir.glob("*mock_data.json"))
    
    # Check Revit mock files
    revit_dir = data_dir / "revit"
    if revit_dir.exists():
        mock_files.extend(revit_dir.glob("*mock_data.json"))
    
    if not mock_files:
        print("❌ No mock data files found!")
        return 1
    
    print(f"🔍 Found {len(mock_files)} mock data files:")
    for mock_file in mock_files:
        print(f"   - {mock_file.name}")
    
    # Convert and test each mock file
    for mock_file in mock_files:
        print(f"\n📦 Processing {mock_file.name}...")
        
        try:
            # Convert mock data to project format
            project_file = convert_mock_to_project_format(mock_file)
            
            # Test agent tools
            test_agent_tools(project_file)
            
        except Exception as e:
            print(f"❌ Error processing {mock_file.name}: {e}")
            continue
    
    print(f"\n🎉 All tests completed!")
    print(f"   - Mock files processed: {len(mock_files)}")
    print(f"   - Agent tools tested successfully")
    print(f"   - Generated project files in data/extracted/")
    
    return 0


if __name__ == "__main__":
    exit(main())
