#!/usr/bin/env python3
"""
Test script for agent tools.

This script tests the 6 agent tools with the mock data to ensure they work correctly.
"""

import json
from pathlib import Path
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


def test_agent_tools():
    """Test all agent tools with mock data."""
    print("🧪 Testing Agent Tools")
    print("=" * 50)
    
    # Test 1: Load project data
    print("\n1. Testing load_project_data()...")
    try:
        mock_data_path = Path("data/blueprints/cad/I01.4_mock_data.json")
        project = load_project_data(mock_data_path)
        print(f"✅ Successfully loaded project: {project.metadata.project_name}")
        print(f"   - Building type: {project.metadata.building_type}")
        print(f"   - Levels: {len(project.levels)}")
        print(f"   - Total rooms: {len(project.get_all_rooms())}")
        print(f"   - Total doors: {len(project.get_all_doors())}")
    except Exception as e:
        print(f"❌ Error loading project data: {e}")
        return
    
    # Test 2: Get project summary
    print("\n2. Testing get_project_summary()...")
    try:
        summary = get_project_summary()
        print(f"✅ Project summary:")
        print(f"   - Total area: {summary['total_area_sqm']:.1f} sqm")
        print(f"   - Room uses: {summary['room_uses']}")
        print(f"   - Door types: {summary['door_types']}")
    except Exception as e:
        print(f"❌ Error getting project summary: {e}")
    
    # Test 3: Get room info
    print("\n3. Testing get_room_info()...")
    try:
        room_info = get_room_info("R001")
        print(f"✅ Room R001 info:")
        print(f"   - Name: {room_info['name']}")
        print(f"   - Area: {room_info['area_sqm']} sqm")
        print(f"   - Use: {room_info['use']}")
        print(f"   - Level: {room_info['level']}")
    except Exception as e:
        print(f"❌ Error getting room info: {e}")
    
    # Test 4: Get door info
    print("\n4. Testing get_door_info()...")
    try:
        door_info = get_door_info("D001")
        print(f"✅ Door D001 info:")
        print(f"   - Width: {door_info['width_mm']} mm")
        print(f"   - Clear width: {door_info['clear_width_mm']} mm")
        print(f"   - Type: {door_info['door_type']}")
        print(f"   - Emergency exit: {door_info['is_emergency_exit']}")
    except Exception as e:
        print(f"❌ Error getting door info: {e}")
    
    # Test 5: List all doors
    print("\n5. Testing list_all_doors()...")
    try:
        doors = list_all_doors()
        print(f"✅ Found {len(doors)} doors:")
        for door in doors[:3]:  # Show first 3 doors
            print(f"   - {door['id']}: {door['width_mm']}mm, {door['door_type']}")
    except Exception as e:
        print(f"❌ Error listing doors: {e}")
    
    # Test 6: Check door compliance
    print("\n6. Testing check_door_width_compliance()...")
    try:
        compliance = check_door_width_compliance("D001")
        print(f"✅ Door D001 compliance:")
        print(f"   - Clear width: {compliance['clear_width_mm']} mm")
        print(f"   - Required: {compliance['required_width_mm']} mm")
        print(f"   - Status: {compliance['compliance_status']}")
        print(f"   - Message: {compliance['message']}")
    except Exception as e:
        print(f"❌ Error checking door compliance: {e}")
    
    # Test 7: Calculate egress distance
    print("\n7. Testing calculate_egress_distance()...")
    try:
        egress = calculate_egress_distance("R001")
        if "error" in egress:
            print(f"⚠️  Egress calculation warning: {egress['error']}")
        else:
            print(f"✅ Room R001 egress:")
            print(f"   - Distance: {egress['egress_distance_m']:.1f} m")
            print(f"   - Max allowed: {egress['max_allowed_distance_m']} m")
            print(f"   - Status: {egress['compliance_status']}")
            print(f"   - Accessible: {egress['is_accessible']}")
    except Exception as e:
        print(f"❌ Error calculating egress distance: {e}")
    
    # Test 8: Get available tools
    print("\n8. Testing get_available_tools()...")
    try:
        tools = get_available_tools()
        print(f"✅ Available tools ({len(tools)}):")
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")
    except Exception as e:
        print(f"❌ Error getting available tools: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Agent tools testing completed!")


if __name__ == "__main__":
    test_agent_tools()
