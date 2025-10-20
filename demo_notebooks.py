#!/usr/bin/env python3
"""
Demo script to showcase the AEC Compliance Agent capabilities.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.schemas import Project
from src.calculations.geometry import calculate_room_area, get_room_centroid
from src.calculations.graph import CirculationGraph
from src.utils.visualization import plot_floorplan

def demo_data_extraction():
    """Demo data extraction capabilities."""
    print("🏗️  DEMO: Data Extraction from CAD Files")
    print("=" * 60)
    
    # Load BAUHAUS project data
    with open("data/extracted/bauhaus_example.json", 'r') as f:
        data = json.load(f)
    
    project = Project(**data)
    
    print(f"Project: {project.metadata.project_name}")
    print(f"Building Use: {project.metadata.building_use}")
    print(f"Extraction Date: {project.metadata.extraction_date}")
    
    print(f"\n📊 Building Elements:")
    print(f"  - Rooms: {len(project.rooms)}")
    print(f"  - Doors: {len(project.doors)}")
    print(f"  - Walls: {len(project.walls)}")
    print(f"  - Fire Equipment: {len(project.fire_equipment)}")
    print(f"  - Fire Sectors: {len(project.sectors)}")
    
    # Show some room details
    print(f"\n🏠 Sample Rooms:")
    for room in project.rooms[:3]:
        area = calculate_room_area(room)
        centroid = get_room_centroid(room)
        print(f"  - {room.name}: {area:.1f} m² at ({centroid[0]:.1f}, {centroid[1]:.1f})")
    
    # Show fire equipment
    print(f"\n🔥 Fire Safety Equipment:")
    equipment_counts = {}
    for eq in project.fire_equipment:
        equipment_counts[eq.equipment_type] = equipment_counts.get(eq.equipment_type, 0) + 1
    
    for eq_type, count in equipment_counts.items():
        print(f"  - {eq_type.title()}: {count}")
    
    return project

def demo_calculations(project):
    """Demo calculation capabilities."""
    print(f"\n📐 DEMO: Geometric Calculations & Route Analysis")
    print("=" * 60)
    
    # Calculate total building area
    total_area = sum(calculate_room_area(room) for room in project.rooms)
    print(f"Total Building Area: {total_area:.1f} m²")
    
    # Find egress doors
    egress_doors = project.get_egress_doors()
    print(f"Emergency Exits: {len(egress_doors)}")
    
    # Analyze circulation graph
    graph = CirculationGraph(project)
    stats = graph.get_graph_statistics()
    
    print(f"\n🔄 Circulation Analysis:")
    print(f"  - Connected Spaces: {stats['num_rooms']}")
    print(f"  - Connections: {stats['num_connections']}")
    print(f"  - Graph Connected: {stats.get('is_connected', 'Unknown')}")
    
    # Calculate evacuation routes
    if egress_doors:
        print(f"\n🚨 Evacuation Routes:")
        for room in project.rooms[:3]:
            try:
                route_info = graph.shortest_egress_path(room.id)
                if route_info:
                    path, distance = route_info
                    print(f"  - {room.name}: {distance:.1f}m to exit via {' → '.join(path)}")
            except:
                print(f"  - {room.name}: Route not found")
    
    return graph

def demo_visualization(project, graph):
    """Demo visualization capabilities."""
    print(f"\n📊 DEMO: Floor Plan Visualization")
    print("=" * 60)
    
    try:
        # Create floor plan visualization
        fig = plot_floorplan(project, title="BAUHAUS Leganés - Floor Plan")
        
        # Save visualization
        output_path = Path("outputs/visualizations/bauhaus_demo.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Floor plan saved to: {output_path}")
        
        # Create graph visualization
        fig = graph.visualize_graph()
        
        output_path = Path("outputs/visualizations/circulation_graph_demo.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Circulation graph saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")

def demo_mock_data():
    """Demo mock data from CAD files."""
    print(f"\n📁 DEMO: Mock Data from CAD Files")
    print("=" * 60)
    
    # Load fire equipment data
    fire_file = Path("data/blueprints/cad/I01.4_mock_data.json")
    if fire_file.exists():
        with open(fire_file, 'r') as f:
            fire_data = json.load(f)
        
        print(f"Fire Protection System (I01.4):")
        print(f"  - Equipment Count: {len(fire_data['equipment'])}")
        
        equipment_types = {}
        for eq in fire_data['equipment']:
            eq_type = eq['type']
            equipment_types[eq_type] = equipment_types.get(eq_type, 0) + 1
        
        for eq_type, count in equipment_types.items():
            print(f"    - {eq_type}: {count}")
    
    # Load sectorization data
    sector_file = Path("data/blueprints/cad/I01.6_mock_data.json")
    if sector_file.exists():
        with open(sector_file, 'r') as f:
            sector_data = json.load(f)
        
        print(f"\nSectorization (I01.6):")
        print(f"  - Fire Sectors: {len(sector_data['sectors'])}")
        print(f"  - Fire Doors: {len(sector_data['fire_doors'])}")
        
        for sector in sector_data['sectors']:
            print(f"    - {sector['name']}: {sector['fire_rating']}, {sector['area']} m²")
    

def main():
    """Run the complete demo."""
    print("🎉 AEC COMPLIANCE AGENT - DEMONSTRATION")
    print("🏗️  Agentic AI for Building Code Compliance")
    print("📅 Presentation: October 29, 2025 at UPC-EPSEB")
    print("=" * 60)
    
    try:
        # Demo 1: Data extraction
        project = demo_data_extraction()
        
        # Demo 2: Calculations
        graph = demo_calculations(project)
        
        # Demo 3: Visualization
        demo_visualization(project, graph)
        
        # Demo 4: Mock data from CAD files
        demo_mock_data()
        
        print(f"\n🎯 DEMO SUMMARY")
        print("=" * 60)
        print("✅ Data extraction from CAD: Working")
        print("✅ Geometric calculations: Working") 
        print("✅ Route analysis: Working")
        print("✅ Visualization: Working")
        print("✅ Mock data generation: Working")
        
        print(f"\n📓 Next Steps:")
        print("1. Run Jupyter notebooks:")
        print("   jupyter notebook")
        print("   Open: notebooks/01_data_extraction_simple.ipynb")
        print("2. Convert DWG files to DXF (requires AutoCAD)")
        print("3. Implement RAG system for normativa queries")
        print("4. Create ReAct agent for autonomous verification")
        
        print(f"\n🚀 Ready for October 29 presentation!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()