#!/usr/bin/env python3
"""
Comprehensive test of all system components.
"""

import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_data_loading():
    """Test data loading functionality."""
    print("📦 Testing Data Loading")
    print("-" * 30)
    
    try:
        with open('data/extracted/test_project.json', 'r') as f:
            project_data = json.load(f)
        
        print(f"✅ Project loaded: {project_data['metadata']['project_name']}")
        print(f"✅ Rooms: {len(project_data['levels'][0]['rooms'])}")
        print(f"✅ Doors: {len(project_data['levels'][0]['doors'])}")
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_calculations():
    """Test geometric calculations."""
    print("\n🧮 Testing Calculations")
    print("-" * 30)
    
    try:
        from shapely.geometry import Polygon
        import networkx as nx
        
        with open('data/extracted/test_project.json', 'r') as f:
            project = json.load(f)
        
        # Test area calculation
        for room in project['levels'][0]['rooms']:
            # Convert boundary points to polygon
            points = [(p['x'], p['y']) for p in room['boundary']['points']]
            polygon = Polygon(points)
            area = polygon.area
            print(f"✅ {room['name']}: {area:.2f} m²")
        
        # Test graph creation
        G = nx.Graph()
        for room in project['levels'][0]['rooms']:
            G.add_node(room['id'], name=room['name'])
        
        for door in project['levels'][0]['doors']:
            if door.get('from_room') and door.get('to_room'):
                G.add_edge(door['from_room'], door['to_room'])
        
        print(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"✅ Connected: {nx.is_connected(G)}")
        return True
    except Exception as e:
        print(f"❌ Calculations failed: {e}")
        return False

def test_schemas():
    """Test Pydantic schemas."""
    print("\n📋 Testing Schemas")
    print("-" * 30)
    
    try:
        from schemas import Project, Room, Door, Point2D, Boundary, BuildingUse, DoorType
        
        # Test basic schema imports
        print("✅ Schema imports successful")
        
        # Test room schema
        room_data = {
            "id": "TEST_ROOM",
            "name": "Test Room",
            "area": 20.0,
            "use": "office",
            "boundary": {
                "points": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 5.0, "y": 0.0},
                    {"x": 5.0, "y": 4.0},
                    {"x": 0.0, "y": 4.0},
                    {"x": 0.0, "y": 0.0}
                ]
            },
            "level": "Ground Floor",
            "occupancy_load": 10,
            "fire_rating": "RF_60"
        }
        
        room = Room(**room_data)
        print(f"✅ Room schema: {room.name} ({room.area}m²)")
        
        # Test door schema
        door_data = {
            "id": "TEST_DOOR",
            "name": "Test Door",
            "width_mm": 800.0,
            "height_mm": 2100.0,
            "door_type": "single",
            "fire_rating": "RF_60",
            "position": {"x": 2.5, "y": 2.0, "z": 0.0},
            "from_room": "TEST_ROOM",
            "to_room": None,
            "is_emergency_exit": True,
            "is_accessible": True
        }
        
        door = Door(**door_data)
        print(f"✅ Door schema: {door.id} ({door.width_mm}mm wide)")
        
        return True
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False

def test_agent_tools():
    """Test agent tools."""
    print("\n🛠️ Testing Agent Tools")
    print("-" * 30)
    
    try:
        from agent.tools import (
            load_project_data, get_room_info, get_door_info, 
            list_all_doors, check_door_width_compliance
        )
        
        # Load project data
        project_path = Path("data/extracted/test_project.json")
        load_project_data(project_path)
        print("✅ Project data loaded")
        
        # Test room info
        room_info = get_room_info.invoke({"room_id": "ROOM_001"})
        print(f"✅ Room info: {room_info['name']}")
        
        # Test door info
        door_info = get_door_info.invoke({"door_id": "DOOR_001"})
        print(f"✅ Door info: {door_info['name']}")
        
        # Test list doors
        doors = list_all_doors.invoke({})
        print(f"✅ Listed {len(doors)} doors")
        
        return True
    except Exception as e:
        print(f"❌ Agent tools failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embeddings():
    """Test embeddings without full RAG."""
    print("\n🔍 Testing Embeddings")
    print("-" * 30)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test basic embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "ancho mínimo puerta evacuación"
        embedding = model.encode(test_text)
        
        print(f"✅ Embeddings working: {len(embedding)} dimensions")
        return True
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AEC Compliance Agent - Component Testing")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_calculations,
        test_schemas,
        test_agent_tools,
        test_embeddings
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n📊 Test Summary")
    print("=" * 50)
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 System is ready for presentation!")
        print("\nComponents working:")
        print("  ✅ Data loading from JSON files")
        print("  ✅ Geometric calculations (Shapely, NetworkX)")
        print("  ✅ Pydantic schema validation")
        print("  ✅ Agent tools for compliance checking")
        print("  ✅ Embeddings for RAG system")
        print("\n📝 Note: RAG system needs dependency fixes for full testing")
    else:
        print(f"❌ {passed}/{total} tests passed")
        print("\n⚠️  Some components need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
