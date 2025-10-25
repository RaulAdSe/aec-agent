#!/usr/bin/env python3
"""
Test script to verify notebooks can run without issues.
"""

import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_loading():
    """Test if we can load the example data."""
    print("Testing data loading...")
    
    # Load example data
    example_file = Path("data/extracted/bauhaus_example.json")
    if not example_file.exists():
        print(f"❌ Example file not found: {example_file}")
        return False
    
    with open(example_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded project: {data['metadata']['project_name']}")
    print(f"   - Rooms: {len(data['rooms'])}")
    print(f"   - Doors: {len(data['doors'])}")
    print(f"   - Fire Equipment: {len(data['fire_equipment'])}")
    
    # Load mock data
    mock_files = [
        Path("data/blueprints/cad/I01.4_mock_data.json"),
        Path("data/blueprints/cad/I01.6_mock_data.json")
    ]
    
    for mock_file in mock_files:
        if mock_file.exists():
            with open(mock_file, 'r') as f:
                mock_data = json.load(f)
            print(f"✅ Loaded mock data: {mock_file.name}")
        else:
            print(f"⚠️  Mock file not found: {mock_file}")
    
    return True

def test_schemas():
    """Test if schemas work correctly."""
    print("\nTesting schemas...")
    
    try:
        from src.schemas import Project, Room, Door, FireEquipment
        
        # Create a test room
        room = Room(
            id="TEST01",
            name="Test Room",
            level="Ground",
            boundary=[[0, 0], [5, 0], [5, 4], [0, 4]]
        )
        print(f"✅ Created room: {room.name}")
        
        # Create a test door
        door = Door(
            id="DTEST01",
            position=[2.5, 0],
            width=0.9
        )
        print(f"✅ Created door with width: {door.width}m")
        
        return True
    except Exception as e:
        print(f"❌ Schema error: {e}")
        return False

def test_geometry():
    """Test geometry calculations."""
    print("\nTesting geometry calculations...")
    
    try:
        from src.calculations.geometry import (
            calculate_room_area,
            get_room_centroid,
            distance_between_points
        )
        from src.schemas import Room
        
        # Create test room
        room = Room(
            id="GEOM01",
            name="Geometry Test",
            level="Ground",
            boundary=[[0, 0], [10, 0], [10, 8], [0, 8]]
        )
        
        area = calculate_room_area(room)
        centroid = get_room_centroid(room)
        distance = distance_between_points([0, 0], [3, 4])
        
        print(f"✅ Room area: {area:.1f} m²")
        print(f"✅ Room centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
        print(f"✅ Distance calculation: {distance:.1f} m")
        
        return True
    except Exception as e:
        print(f"❌ Geometry error: {e}")
        return False

def test_graph():
    """Test graph calculations."""
    print("\nTesting graph analysis...")
    
    try:
        from src.calculations.graph import CirculationGraph
        from src.schemas import Project
        
        # Load example data
        with open("data/extracted/bauhaus_example.json", 'r') as f:
            data = json.load(f)
        
        project = Project(**data)
        graph = CirculationGraph(project)
        graph.build_graph()
        
        stats = graph.get_graph_statistics()
        print(f"✅ Graph created:")
        print(f"   - Nodes (rooms): {stats['num_rooms']}")
        print(f"   - Edges (doors): {stats['num_connections']}")
        print(f"   - Connected: {stats.get('is_connected', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Graph error: {e}")
        return False

def test_visualization():
    """Test if matplotlib works."""
    print("\nTesting visualization...")
    
    try:
        # Create a simple plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1, 2], [0, 1, 0], 'b-')
        ax.set_title("Visualization Test")
        
        # Save to file
        output_path = Path("outputs/visualizations/test_plot.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
        print(f"✅ Created test plot: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("NOTEBOOK DEPENDENCY TESTS")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Schemas", test_schemas),
        ("Geometry", test_geometry),
        ("Graph", test_graph),
        ("Visualization", test_visualization)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    total_passed = sum(1 for _, s in results if s)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Notebooks should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)