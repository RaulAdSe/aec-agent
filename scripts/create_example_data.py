#!/usr/bin/env python3
"""
Example Data Generator for AEC Compliance Agent.

This script creates comprehensive example building data for the BAUHAUS Leganés project.
It generates realistic rooms, doors, walls, fire equipment, and sectors following
Spanish fire safety regulations and building codes.

Usage:
    python scripts/create_example_data.py --output data/extracted/bauhaus_example.json
    python scripts/create_example_data.py --project "Custom Project" --levels 2
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from schemas import (
    Project, ProjectMetadata, Room, Door, Wall, FireEquipment, 
    Sector, Level, EvacuationRoute
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_project_metadata(project_name: str = "BAUHAUS Leganés") -> ProjectMetadata:
    """
    Create project metadata for the example building.
    
    Args:
        project_name: Name of the building project
        
    Returns:
        ProjectMetadata object
    """
    return ProjectMetadata(
        project_name=project_name,
        project_id="BAUHAUS-2024-001",
        level_name="Planta Baja",
        level_number=0,
        extraction_date=datetime.now(),
        source_file="example_data_generator.py",
        building_use="commercial"
    )


def create_example_rooms(level_name: str = "Planta Baja") -> List[Room]:
    """
    Create example rooms for a commercial building.
    
    Args:
        level_name: Name of the building level
        
    Returns:
        List of Room objects
    """
    rooms = []
    
    # Main lobby/entrance
    rooms.append(Room(
        id="R001",
        name="Lobby Principal",
        level=level_name,
        boundary=[[0, 0], [15, 0], [15, 8], [0, 8]],
        use_type="lobby",
        area=120.0,
        occupancy_load=60,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=3.5
    ))
    
    # Sales area
    rooms.append(Room(
        id="R002",
        name="Área de Ventas",
        level=level_name,
        boundary=[[15, 0], [35, 0], [35, 20], [15, 20]],
        use_type="retail",
        area=400.0,
        occupancy_load=200,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=4.0
    ))
    
    # Storage/warehouse
    rooms.append(Room(
        id="R003",
        name="Almacén",
        level=level_name,
        boundary=[[35, 0], [50, 0], [50, 15], [35, 15]],
        use_type="storage",
        area=225.0,
        occupancy_load=15,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=6.0
    ))
    
    # Office area
    rooms.append(Room(
        id="R004",
        name="Oficina Administración",
        level=level_name,
        boundary=[[15, 20], [25, 20], [25, 25], [15, 25]],
        use_type="office",
        area=50.0,
        occupancy_load=10,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=2.8
    ))
    
    # Meeting room
    rooms.append(Room(
        id="R005",
        name="Sala de Reuniones",
        level=level_name,
        boundary=[[25, 20], [35, 20], [35, 25], [25, 25]],
        use_type="meeting_room",
        area=50.0,
        occupancy_load=12,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=2.8
    ))
    
    # Main corridor
    rooms.append(Room(
        id="R006",
        name="Pasillo Principal",
        level=level_name,
        boundary=[[0, 8], [15, 8], [15, 12], [0, 12]],
        use_type="corridor",
        area=60.0,
        occupancy_load=30,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=2.8
    ))
    
    # Staircase
    rooms.append(Room(
        id="R007",
        name="Escalera Principal",
        level=level_name,
        boundary=[[0, 12], [6, 12], [6, 18], [0, 18]],
        use_type="stairs",
        area=36.0,
        occupancy_load=50,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=3.0
    ))
    
    # Bathrooms
    rooms.append(Room(
        id="R008",
        name="Aseos",
        level=level_name,
        boundary=[[6, 12], [12, 12], [12, 18], [6, 18]],
        use_type="bathroom",
        area=36.0,
        occupancy_load=8,
        has_emergency_lighting=True,
        has_fire_detection=False,
        ceiling_height=2.5
    ))
    
    # Technical room
    rooms.append(Room(
        id="R009",
        name="Cuarto Técnico",
        level=level_name,
        boundary=[[35, 15], [45, 15], [45, 20], [35, 20]],
        use_type="technical",
        area=50.0,
        occupancy_load=2,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=2.5
    ))
    
    # Loading dock
    rooms.append(Room(
        id="R010",
        name="Muelle de Carga",
        level=level_name,
        boundary=[[45, 0], [55, 0], [55, 15], [45, 15]],
        use_type="loading",
        area=150.0,
        occupancy_load=10,
        has_emergency_lighting=True,
        has_fire_detection=True,
        ceiling_height=5.0
    ))
    
    return rooms


def create_example_doors() -> List[Door]:
    """
    Create example doors for the building.
    
    Returns:
        List of Door objects
    """
    doors = []
    
    # Main entrance (double doors)
    doors.append(Door(
        id="D001",
        position=[7.5, 0],
        width=1.8,
        height=2.5,
        door_type="double",
        is_egress=True,
        fire_rating="EI-60",
        opening_direction="outward",
        connected_rooms=["R001", "exterior"]
    ))
    
    # Lobby to sales area
    doors.append(Door(
        id="D002",
        position=[15, 4],
        width=1.2,
        door_type="double",
        is_egress=False,
        opening_direction="inward",
        connected_rooms=["R001", "R002"]
    ))
    
    # Sales to storage
    doors.append(Door(
        id="D003",
        position=[35, 7.5],
        width=0.9,
        door_type="single",
        is_egress=False,
        fire_rating="EI-60",
        connected_rooms=["R002", "R003"]
    ))
    
    # Emergency exit from sales area
    doors.append(Door(
        id="D004",
        position=[35, 20],
        width=1.2,
        door_type="double",
        is_egress=True,
        fire_rating="EI-60",
        opening_direction="outward",
        connected_rooms=["R002", "exterior"]
    ))
    
    # Office entrance
    doors.append(Door(
        id="D005",
        position=[15, 22.5],
        width=0.9,
        door_type="single",
        is_egress=False,
        connected_rooms=["R002", "R004"]
    ))
    
    # Meeting room entrance
    doors.append(Door(
        id="D006",
        position=[25, 22.5],
        width=0.9,
        door_type="single",
        is_egress=False,
        connected_rooms=["R004", "R005"]
    ))
    
    # Corridor access
    doors.append(Door(
        id="D007",
        position=[7.5, 8],
        width=1.0,
        door_type="single",
        is_egress=False,
        connected_rooms=["R001", "R006"]
    ))
    
    # Staircase access
    doors.append(Door(
        id="D008",
        position=[3, 12],
        width=1.0,
        door_type="single",
        is_egress=True,
        fire_rating="EI-30",
        connected_rooms=["R006", "R007"]
    ))
    
    # Bathroom access
    doors.append(Door(
        id="D009",
        position=[9, 12],
        width=0.8,
        door_type="single",
        is_egress=False,
        connected_rooms=["R006", "R008"]
    ))
    
    # Technical room access
    doors.append(Door(
        id="D010",
        position=[40, 15],
        width=0.8,
        door_type="single",
        is_egress=False,
        fire_rating="EI-60",
        connected_rooms=["R003", "R009"]
    ))
    
    # Loading dock access
    doors.append(Door(
        id="D011",
        position=[50, 7.5],
        width=2.5,
        door_type="sliding",
        is_egress=True,
        connected_rooms=["R010", "exterior"]
    ))
    
    return doors


def create_example_walls() -> List[Wall]:
    """
    Create example walls for the building structure.
    
    Returns:
        List of Wall objects
    """
    walls = []
    
    # Exterior walls
    walls.extend([
        Wall(id="W001", start=[0, 0], end=[55, 0], length=55.0, thickness=0.30, 
             fire_rating="EI-120", material="brick", is_exterior=True),
        Wall(id="W002", start=[55, 0], end=[55, 25], length=25.0, thickness=0.30, 
             fire_rating="EI-120", material="brick", is_exterior=True),
        Wall(id="W003", start=[55, 25], end=[0, 25], length=55.0, thickness=0.30, 
             fire_rating="EI-120", material="brick", is_exterior=True),
        Wall(id="W004", start=[0, 25], end=[0, 0], length=25.0, thickness=0.30, 
             fire_rating="EI-120", material="brick", is_exterior=True),
    ])
    
    # Interior separating walls (fire-rated)
    walls.extend([
        Wall(id="W005", start=[15, 0], end=[15, 25], length=25.0, thickness=0.20, 
             fire_rating="EI-90", material="concrete_block", is_exterior=False),
        Wall(id="W006", start=[35, 0], end=[35, 25], length=25.0, thickness=0.20, 
             fire_rating="EI-90", material="concrete_block", is_exterior=False),
        Wall(id="W007", start=[0, 8], end=[15, 8], length=15.0, thickness=0.15, 
             fire_rating="EI-60", material="drywall", is_exterior=False),
        Wall(id="W008", start=[15, 20], end=[35, 20], length=20.0, thickness=0.15, 
             fire_rating="EI-60", material="drywall", is_exterior=False),
    ])
    
    # Storage area walls
    walls.extend([
        Wall(id="W009", start=[35, 15], end=[50, 15], length=15.0, thickness=0.20, 
             fire_rating="EI-90", material="concrete_block", is_exterior=False),
        Wall(id="W010", start=[45, 0], end=[45, 15], length=15.0, thickness=0.15, 
             fire_rating="EI-60", material="drywall", is_exterior=False),
    ])
    
    # Office partitions
    walls.extend([
        Wall(id="W011", start=[25, 20], end=[25, 25], length=5.0, thickness=0.10, 
             fire_rating="EI-30", material="drywall", is_exterior=False),
        Wall(id="W012", start=[0, 12], end=[15, 12], length=15.0, thickness=0.15, 
             fire_rating="EI-60", material="concrete_block", is_exterior=False),
    ])
    
    # Bathroom and technical room walls
    walls.extend([
        Wall(id="W013", start=[6, 12], end=[6, 18], length=6.0, thickness=0.15, 
             fire_rating="EI-60", material="concrete_block", is_exterior=False),
        Wall(id="W014", start=[12, 12], end=[12, 18], length=6.0, thickness=0.15, 
             fire_rating="EI-60", material="concrete_block", is_exterior=False),
        Wall(id="W015", start=[0, 18], end=[12, 18], length=12.0, thickness=0.15, 
             fire_rating="EI-60", material="concrete_block", is_exterior=False),
    ])
    
    return walls


def create_example_fire_equipment(level_name: str = "Planta Baja") -> List[FireEquipment]:
    """
    Create example fire safety equipment following Spanish regulations.
    
    Args:
        level_name: Name of the building level
        
    Returns:
        List of FireEquipment objects
    """
    equipment = []
    
    # Fire extinguishers (every 15m maximum distance)
    extinguishers = [
        {"id": "FE001", "pos": [7.5, 4], "area": "lobby"},
        {"id": "FE002", "pos": [25, 10], "area": "sales"},
        {"id": "FE003", "pos": [42.5, 7.5], "area": "storage"},
        {"id": "FE004", "pos": [20, 22.5], "area": "office"},
        {"id": "FE005", "pos": [7.5, 15], "area": "corridor"},
        {"id": "FE006", "pos": [50, 7.5], "area": "loading"},
    ]
    
    for ext in extinguishers:
        equipment.append(FireEquipment(
            id=ext["id"],
            equipment_type="extinguisher",
            position=ext["pos"],
            coverage_radius=15.0,
            floor_level=level_name,
            status="active",
            last_inspection=datetime(2024, 1, 15)
        ))
    
    # Fire hose reels (BIEs)
    hose_reels = [
        {"id": "FE007", "pos": [1, 10], "area": "main_corridor"},
        {"id": "FE008", "pos": [30, 2], "area": "sales_center"},
        {"id": "FE009", "pos": [40, 18], "area": "storage_area"},
    ]
    
    for hose in hose_reels:
        equipment.append(FireEquipment(
            id=hose["id"],
            equipment_type="hydrant",
            position=hose["pos"],
            coverage_radius=25.0,
            floor_level=level_name,
            status="active",
            last_inspection=datetime(2024, 2, 1)
        ))
    
    # Smoke detectors
    detectors = [
        {"id": "FE010", "pos": [7.5, 4], "area": "lobby"},
        {"id": "FE011", "pos": [25, 10], "area": "sales"},
        {"id": "FE012", "pos": [42.5, 7.5], "area": "storage"},
        {"id": "FE013", "pos": [20, 22.5], "area": "office"},
        {"id": "FE014", "pos": [30, 22.5], "area": "meeting"},
        {"id": "FE015", "pos": [3, 15], "area": "stairs"},
        {"id": "FE016", "pos": [40, 17.5], "area": "technical"},
    ]
    
    for detector in detectors:
        equipment.append(FireEquipment(
            id=detector["id"],
            equipment_type="alarm",
            position=detector["pos"],
            coverage_radius=10.0,
            floor_level=level_name,
            status="active",
            last_inspection=datetime(2024, 3, 1)
        ))
    
    # Emergency lighting
    emergency_lights = [
        {"id": "FE017", "pos": [7.5, 1], "area": "main_exit"},
        {"id": "FE018", "pos": [35, 19], "area": "emergency_exit"},
        {"id": "FE019", "pos": [3, 13], "area": "stairs"},
        {"id": "FE020", "pos": [7.5, 9], "area": "corridor"},
        {"id": "FE021", "pos": [50, 1], "area": "loading_exit"},
    ]
    
    for light in emergency_lights:
        equipment.append(FireEquipment(
            id=light["id"],
            equipment_type="emergency_light",
            position=light["pos"],
            coverage_radius=5.0,
            floor_level=level_name,
            status="active",
            last_inspection=datetime(2024, 1, 30)
        ))
    
    # Exit signs
    exit_signs = [
        {"id": "FE022", "pos": [7.5, 0.5], "area": "main_entrance"},
        {"id": "FE023", "pos": [35, 19.5], "area": "sales_exit"},
        {"id": "FE024", "pos": [2, 12.5], "area": "stair_access"},
        {"id": "FE025", "pos": [50, 0.5], "area": "loading_exit"},
    ]
    
    for sign in exit_signs:
        equipment.append(FireEquipment(
            id=sign["id"],
            equipment_type="exit_sign",
            position=sign["pos"],
            coverage_radius=20.0,
            floor_level=level_name,
            status="active",
            last_inspection=datetime(2024, 2, 15)
        ))
    
    return equipment


def create_example_sectors() -> List[Sector]:
    """
    Create fire compartments/sectors following Spanish CTE regulations.
    
    Returns:
        List of Sector objects
    """
    sectors = []
    
    # Main commercial sector (sales + lobby)
    sectors.append(Sector(
        id="S001",
        name="Sector Comercial Principal",
        boundary=[[0, 0], [35, 0], [35, 20], [15, 20], [15, 8], [0, 8]],
        area=520.0,
        fire_resistance="EI-90",
        rooms=["R001", "R002"],
        max_evacuation_distance=25.0,
        compartment_type="general"
    ))
    
    # Storage/warehouse sector
    sectors.append(Sector(
        id="S002",
        name="Sector Almacén",
        boundary=[[35, 0], [55, 0], [55, 25], [35, 25]],
        area=500.0,
        fire_resistance="EI-120",
        rooms=["R003", "R009", "R010"],
        max_evacuation_distance=35.0,
        compartment_type="risk_special"
    ))
    
    # Office sector
    sectors.append(Sector(
        id="S003",
        name="Sector Oficinas",
        boundary=[[15, 20], [35, 20], [35, 25], [15, 25]],
        area=100.0,
        fire_resistance="EI-60",
        rooms=["R004", "R005"],
        max_evacuation_distance=15.0,
        compartment_type="general"
    ))
    
    # Circulation and services sector
    sectors.append(Sector(
        id="S004",
        name="Sector Circulación",
        boundary=[[0, 8], [15, 8], [15, 25], [0, 25]],
        area=132.0,
        fire_resistance="EI-60",
        rooms=["R006", "R007", "R008"],
        max_evacuation_distance=20.0,
        compartment_type="protected"
    ))
    
    return sectors


def create_example_evacuation_routes() -> List[EvacuationRoute]:
    """
    Create example evacuation routes from rooms to exits.
    
    Returns:
        List of EvacuationRoute objects
    """
    routes = []
    
    # From sales area to main exit
    routes.append(EvacuationRoute(
        id="ER001",
        from_room="R002",
        to_exit="D001",
        path_nodes=[[25, 10], [15, 10], [15, 4], [7.5, 4], [7.5, 0]],
        distance=22.5,
        is_accessible=True,
        has_emergency_lighting=True,
        width_minimum=1.2
    ))
    
    # From sales area to emergency exit
    routes.append(EvacuationRoute(
        id="ER002",
        from_room="R002",
        to_exit="D004",
        path_nodes=[[25, 10], [35, 10], [35, 20]],
        distance=20.0,
        is_accessible=True,
        has_emergency_lighting=True,
        width_minimum=1.2
    ))
    
    # From office to main exit via corridor
    routes.append(EvacuationRoute(
        id="ER003",
        from_room="R004",
        to_exit="D001",
        path_nodes=[[20, 22.5], [15, 22.5], [15, 10], [7.5, 10], [7.5, 0]],
        distance=30.0,
        is_accessible=True,
        has_emergency_lighting=True,
        width_minimum=1.0
    ))
    
    # From storage to loading dock exit
    routes.append(EvacuationRoute(
        id="ER004",
        from_room="R003",
        to_exit="D011",
        path_nodes=[[42.5, 7.5], [50, 7.5]],
        distance=7.5,
        is_accessible=True,
        has_emergency_lighting=True,
        width_minimum=2.5
    ))
    
    return routes


def create_comprehensive_example(project_name: str = "BAUHAUS Leganés", 
                                num_levels: int = 1) -> Project:
    """
    Create a comprehensive example project with all building data.
    
    Args:
        project_name: Name of the building project
        num_levels: Number of building levels to create
        
    Returns:
        Complete Project object
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating example project: {project_name}")
    
    # Create metadata
    metadata = create_project_metadata(project_name)
    
    # Create levels
    levels = []
    all_rooms = []
    all_doors = []
    all_walls = []
    all_fire_equipment = []
    all_sectors = []
    
    for level_num in range(num_levels):
        level_name = f"Planta {level_num}" if level_num > 0 else "Planta Baja"
        
        # Create building elements for this level
        rooms = create_example_rooms(level_name)
        doors = create_example_doors()
        walls = create_example_walls()
        fire_equipment = create_example_fire_equipment(level_name)
        sectors = create_example_sectors()
        
        # Adjust IDs for multiple levels
        if level_num > 0:
            for room in rooms:
                room.id = f"{room.id}_{level_num}"
            for door in doors:
                door.id = f"{door.id}_{level_num}"
            for wall in walls:
                wall.id = f"{wall.id}_{level_num}"
            for equipment in fire_equipment:
                equipment.id = f"{equipment.id}_{level_num}"
            for sector in sectors:
                sector.id = f"{sector.id}_{level_num}"
        
        # Create level object
        level = Level(
            name=level_name,
            number=level_num,
            height=level_num * 3.0,  # 3m per floor
            rooms=rooms,
            doors=doors,
            walls=walls,
            fire_equipment=fire_equipment,
            sectors=sectors
        )
        
        levels.append(level)
        all_rooms.extend(rooms)
        all_doors.extend(doors)
        all_walls.extend(walls)
        all_fire_equipment.extend(fire_equipment)
        all_sectors.extend(sectors)
    
    # Create evacuation routes (only for ground floor)
    evacuation_routes = create_example_evacuation_routes()
    
    # Create complete project
    project = Project(
        metadata=metadata,
        levels=levels,
        rooms=all_rooms,
        doors=all_doors,
        walls=all_walls,
        fire_equipment=all_fire_equipment,
        sectors=all_sectors,
        evacuation_routes=evacuation_routes
    )
    
    logger.info(f"Created example project with:")
    logger.info(f"  - Levels: {len(levels)}")
    logger.info(f"  - Rooms: {len(all_rooms)}")
    logger.info(f"  - Doors: {len(all_doors)}")
    logger.info(f"  - Walls: {len(all_walls)}")
    logger.info(f"  - Fire Equipment: {len(all_fire_equipment)}")
    logger.info(f"  - Sectors: {len(all_sectors)}")
    logger.info(f"  - Evacuation Routes: {len(evacuation_routes)}")
    
    return project


def save_project_to_json(project: Project, output_path: Path) -> None:
    """
    Save project data to JSON file.
    
    Args:
        project: Project object to save
        output_path: Path where to save the JSON file
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            project.model_dump(),
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )
    
    logger.info(f"Example project saved to: {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive example building data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate basic example data
  python scripts/create_example_data.py
  
  # Custom project name and output location
  python scripts/create_example_data.py --project "My Building" --output custom_example.json
  
  # Multi-level building
  python scripts/create_example_data.py --levels 3 --output multilevel_example.json
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('data/extracted/bauhaus_example.json'),
        help='Output JSON file path (default: data/extracted/bauhaus_example.json)'
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        default="BAUHAUS Leganés",
        help='Project name (default: BAUHAUS Leganés)'
    )
    
    parser.add_argument(
        '--levels', '-l',
        type=int,
        default=1,
        help='Number of building levels to create (default: 1)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting example data generation")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create comprehensive example project
        project = create_comprehensive_example(
            project_name=args.project,
            num_levels=args.levels
        )
        
        # Save to JSON file
        save_project_to_json(project, args.output)
        
        # Success message
        print(f"\nExample data generated successfully!")
        print(f"Project: {args.project}")
        print(f"Levels: {args.levels}")
        print(f"Output file: {args.output.absolute()}")
        print(f"\nData summary:")
        print(f"  - Rooms: {len(project.rooms)}")
        print(f"  - Doors: {len(project.doors)}")
        print(f"  - Walls: {len(project.walls)}")
        print(f"  - Fire Equipment: {len(project.fire_equipment)}")
        print(f"  - Fire Sectors: {len(project.sectors)}")
        print(f"  - Evacuation Routes: {len(project.evacuation_routes)}")
        
    except Exception as e:
        logger.error(f"Error generating example data: {str(e)}")
        logger.exception("Full traceback:")
        print(f"\nError generating example data: {str(e)}")
        print("Check the logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()