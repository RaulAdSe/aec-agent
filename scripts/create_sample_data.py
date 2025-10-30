#!/usr/bin/env python3
"""
Create sample building data for the notebook
"""

import json
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path().absolute()
sys.path.insert(0, str(current_dir))

def create_sample_building_data():
    """Create sample building data for demonstration"""
    
    print("🏗️ CREATING SAMPLE BUILDING DATA")
    print("=" * 60)
    
    # Create sample building data
    building_data = {
        'architecture': [
            {
                'global_id': 'wall_001',
                'ifc_class': 'IfcWall',
                'name': 'External Wall - North',
                'properties': {
                    'Pset_WallCommon': {
                        'IsExternal': True,
                        'LoadBearing': True,
                        'ThermalTransmittance': 0.25
                    }
                },
                'coordinates': {
                    'reference_point_world_m': [0.0, 0.0, 0.0]
                },
                'geometry_summary': {
                    'length_m': 20.0,
                    'width_m': 0.3,
                    'height_m': 3.0,
                    'volume_m3': 18.0
                }
            },
            {
                'global_id': 'wall_002',
                'ifc_class': 'IfcWall',
                'name': 'External Wall - South',
                'properties': {
                    'Pset_WallCommon': {
                        'IsExternal': True,
                        'LoadBearing': True,
                        'ThermalTransmittance': 0.25
                    }
                },
                'coordinates': {
                    'reference_point_world_m': [0.0, 15.0, 0.0]
                },
                'geometry_summary': {
                    'length_m': 20.0,
                    'width_m': 0.3,
                    'height_m': 3.0,
                    'volume_m3': 18.0
                }
            },
            {
                'global_id': 'door_001',
                'ifc_class': 'IfcDoor',
                'name': 'Main Entrance',
                'properties': {
                    'Pset_DoorCommon': {
                        'IsExternal': True,
                        'FireRating': '30min'
                    }
                },
                'coordinates': {
                    'reference_point_world_m': [10.0, 0.0, 0.0]
                },
                'geometry_summary': {
                    'width_m': 1.2,
                    'height_m': 2.1,
                    'thickness_m': 0.05
                }
            },
            {
                'global_id': 'space_001',
                'ifc_class': 'IfcSpace',
                'name': 'Main Hall',
                'properties': {
                    'Pset_SpaceCommon': {
                        'OccupancyLoad': 50,
                        'Use': 'Assembly'
                    }
                },
                'coordinates': {
                    'reference_point_world_m': [10.0, 7.5, 1.5]
                },
                'geometry_summary': {
                    'area_m2': 300.0,
                    'volume_m3': 900.0
                }
            }
        ],
        'mep': [
            {
                'global_id': 'hvac_001',
                'ifc_class': 'IfcAirTerminal',
                'name': 'Supply Air Terminal',
                'properties': {
                    'Pset_AirTerminalCommon': {
                        'AirFlowRate': 2.5,
                        'NoiseLevel': 35
                    }
                },
                'coordinates': {
                    'reference_point_world_m': [5.0, 5.0, 2.5]
                }
            }
        ],
        'structural': [
            {
                'global_id': 'beam_001',
                'ifc_class': 'IfcBeam',
                'name': 'Main Beam',
                'properties': {
                    'Pset_BeamCommon': {
                        'LoadBearing': True,
                        'Material': 'Steel'
                    }
                },
                'coordinates': {
                    'reference_point_world_m': [10.0, 7.5, 3.0]
                },
                'geometry_summary': {
                    'length_m': 20.0,
                    'width_m': 0.3,
                    'height_m': 0.5
                }
            }
        ]
    }
    
    # Calculate total elements
    total_elements = sum(len(elements) for elements in building_data.values())
    
    print(f"✅ Created sample building data!")
    print(f"\n📈 BUILDING DATA OVERVIEW:")
    print(f"   • Total domains: {len(building_data)}")
    
    for domain, elements in building_data.items():
        count = len(elements)
        print(f"   • {domain.upper()}: {count:,} elements")
    
    print(f"   • Total building elements: {total_elements:,}")
    
    # Show sample element structure
    if building_data['architecture']:
        sample_element = building_data['architecture'][0]
        
        print(f"\n🔍 SAMPLE ELEMENT STRUCTURE (ARCHITECTURE):")
        print(f"   • Element ID: {sample_element.get('global_id', 'N/A')}")
        print(f"   • IFC Class: {sample_element.get('ifc_class', 'N/A')}")
        print(f"   • Name: {sample_element.get('name', 'N/A')}")
        print(f"   • Properties: {len(sample_element.get('properties', {}))} property sets")
        print(f"   • Coordinates: {'✅' if sample_element.get('coordinates') else '❌'}")
        print(f"   • Geometry: {'✅' if sample_element.get('geometry_summary') else '❌'}")
    
    print(f"\n🎯 Ready to analyze {total_elements:,} building elements!")
    
    return building_data

if __name__ == "__main__":
    building_data = create_sample_building_data()
    
    # Save the formatted data for the notebook
    output_path = Path('data/extracted/vilamalla_building.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(building_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Building data saved to: {output_path}")
    print("✅ Data ready for notebook analysis!")
