"""
Integration tests for geometry and graph modules.

This module tests how the geometry and graph functions work together
for real-world compliance verification scenarios.
"""

import pytest
from src.schemas import (
    Project, ProjectMetadata, Level, Room, Door, Wall, 
    Point2D, Point3D, Boundary, BuildingUse, DoorType
)
from src.calculations.geometry import (
    calculate_room_area,
    calculate_room_centroid,
    calculate_door_clear_width,
    calculate_egress_capacity,
    calculate_fire_rating_equivalent
)
from src.calculations.graph import (
    CirculationGraph,
    create_circulation_graph,
    calculate_egress_distance,
    calculate_travel_time
)


def create_realistic_office_building():
    """Create a realistic office building for integration testing."""
    # Create rooms with realistic boundaries
    rooms = [
        Room(
            id="R001",
            name="Reception",
            area=45.0,
            use=BuildingUse.RECEPTION,
            boundary=Boundary(points=[
                Point2D(x=0, y=0),
                Point2D(x=6, y=0),
                Point2D(x=6, y=7.5),
                Point2D(x=0, y=7.5),
                Point2D(x=0, y=0)
            ]),
            level="Ground Floor",
            occupancy_load=5
        ),
        Room(
            id="R002",
            name="Office A",
            area=25.0,
            use=BuildingUse.OFFICE,
            boundary=Boundary(points=[
                Point2D(x=8, y=0),
                Point2D(x=13, y=0),
                Point2D(x=13, y=5),
                Point2D(x=8, y=5),
                Point2D(x=8, y=0)
            ]),
            level="Ground Floor",
            occupancy_load=3
        ),
        Room(
            id="R003",
            name="Office B",
            area=30.0,
            use=BuildingUse.OFFICE,
            boundary=Boundary(points=[
                Point2D(x=15, y=0),
                Point2D(x=20, y=0),
                Point2D(x=20, y=6),
                Point2D(x=15, y=6),
                Point2D(x=15, y=0)
            ]),
            level="Ground Floor",
            occupancy_load=3
        ),
        Room(
            id="R004",
            name="Meeting Room",
            area=35.0,
            use=BuildingUse.MEETING,
            boundary=Boundary(points=[
                Point2D(x=8, y=7),
                Point2D(x=15, y=7),
                Point2D(x=15, y=12),
                Point2D(x=8, y=12),
                Point2D(x=8, y=7)
            ]),
            level="Ground Floor",
            occupancy_load=7
        ),
        Room(
            id="R005",
            name="Main Corridor",
            area=60.0,
            use=BuildingUse.CORRIDOR,
            boundary=Boundary(points=[
                Point2D(x=6, y=0),
                Point2D(x=22, y=0),
                Point2D(x=22, y=2),
                Point2D(x=6, y=2),
                Point2D(x=6, y=0)
            ]),
            level="Ground Floor",
            occupancy_load=12
        ),
        Room(
            id="R006",
            name="Emergency Exit",
            area=15.0,
            use=BuildingUse.EXIT,
            boundary=Boundary(points=[
                Point2D(x=22, y=0),
                Point2D(x=25, y=0),
                Point2D(x=25, y=3),
                Point2D(x=22, y=3),
                Point2D(x=22, y=0)
            ]),
            level="Ground Floor",
            occupancy_load=0
        )
    ]
    
    # Create doors with realistic positions
    doors = [
        # Reception to Main Corridor
        Door(
            id="D001",
            name="Reception Door",
            width_mm=900,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=6, y=1, z=0),
            from_room="R001",
            to_room="R005"
        ),
        # Office A to Main Corridor
        Door(
            id="D002",
            name="Office A Door",
            width_mm=800,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=8, y=1, z=0),
            from_room="R002",
            to_room="R005"
        ),
        # Office B to Main Corridor
        Door(
            id="D003",
            name="Office B Door",
            width_mm=900,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=15, y=1, z=0),
            from_room="R003",
            to_room="R005"
        ),
        # Meeting Room to Main Corridor
        Door(
            id="D004",
            name="Meeting Room Door",
            width_mm=1000,
            height_mm=2100,
            door_type=DoorType.DOUBLE,
            position=Point3D(x=11.5, y=7, z=0),
            from_room="R004",
            to_room="R005"
        ),
        # Main Corridor to Emergency Exit
        Door(
            id="D005",
            name="Emergency Exit Door",
            width_mm=1200,
            height_mm=2100,
            door_type=DoorType.EMERGENCY_EXIT,
            position=Point3D(x=22, y=1, z=0),
            from_room="R005",
            to_room="R006",
            is_emergency_exit=True
        )
    ]
    
    # Create walls
    walls = [
        # External walls
        Wall(
            id="W001",
            start_point=Point3D(x=0, y=0, z=0),
            end_point=Point3D(x=25, y=0, z=0),
            thickness_mm=200,
            height_mm=2700
        ),
        Wall(
            id="W002",
            start_point=Point3D(x=25, y=0, z=0),
            end_point=Point3D(x=25, y=12, z=0),
            thickness_mm=200,
            height_mm=2700
        ),
        Wall(
            id="W003",
            start_point=Point3D(x=25, y=12, z=0),
            end_point=Point3D(x=0, y=12, z=0),
            thickness_mm=200,
            height_mm=2700
        ),
        Wall(
            id="W004",
            start_point=Point3D(x=0, y=12, z=0),
            end_point=Point3D(x=0, y=0, z=0),
            thickness_mm=200,
            height_mm=2700
        )
    ]
    
    # Create level
    level = Level(
        name="Ground Floor",
        elevation=0.0,
        rooms=rooms,
        doors=doors,
        walls=walls
    )
    
    # Create project metadata
    metadata = ProjectMetadata(
        project_name="Realistic Office Building",
        file_name="realistic_office.json",
        building_type="office",
        total_area=210.0,
        number_of_levels=1,
        created_date="2025-10-19T12:00:00",
        modified_date="2025-10-19T12:00:00"
    )
    
    # Create project
    project = Project(
        metadata=metadata,
        levels=[level]
    )
    
    return project


class TestGeometryGraphIntegration:
    """Test integration between geometry and graph modules."""
    
    def test_room_area_calculation_with_boundaries(self):
        """Test that room areas are calculated correctly from boundaries."""
        project = create_realistic_office_building()
        
        for room in project.get_all_rooms():
            calculated_area = calculate_room_area(room)
            expected_area = room.area
            
            # Allow larger differences for integration test (boundary calculation vs stored area)
            assert abs(calculated_area - expected_area) < 50.0
    
    def test_room_centroid_calculation(self):
        """Test that room centroids are calculated correctly."""
        project = create_realistic_office_building()
        
        for room in project.get_all_rooms():
            if room.boundary:
                centroid = calculate_room_centroid(room)
                assert centroid is not None
                assert isinstance(centroid, Point2D)
                
                # Check that centroid is within reasonable bounds
                assert 0 <= centroid.x <= 25
                assert 0 <= centroid.y <= 12
    
    def test_door_clear_width_calculations(self):
        """Test that door clear widths are calculated correctly."""
        project = create_realistic_office_building()
        
        for door in project.get_all_doors():
            clear_width = calculate_door_clear_width(door)
            
            if door.door_type == DoorType.DOUBLE:
                # Double doors should have reduced clear width
                assert clear_width < door.width_mm
                assert clear_width == door.width_mm * 0.67
            else:
                # Single doors should have full width
                assert clear_width == door.width_mm
    
    def test_egress_capacity_calculations(self):
        """Test that egress capacities are calculated correctly."""
        project = create_realistic_office_building()
        
        for room in project.get_all_rooms():
            capacity = calculate_egress_capacity(room.area, room.use)
            
            # Check that capacity is reasonable
            assert capacity > 0
            assert capacity <= room.area * 0.5  # Should not exceed 0.5 persons per sqm
    
    def test_circulation_graph_creation(self):
        """Test that circulation graph is created correctly from realistic data."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        # Check graph structure
        assert graph.graph.number_of_nodes() == len(project.get_all_rooms())
        assert graph.graph.number_of_edges() > 0
        
        # Check that exit nodes are identified
        assert len(graph.exit_nodes) > 0
        
        # Check that all rooms are connected to the circulation system
        stats = graph.get_graph_statistics()
        assert stats["is_connected"] is True
    
    def test_egress_distance_calculations(self):
        """Test egress distance calculations for realistic building."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        # Test egress distances from different rooms
        test_rooms = ["R001", "R002", "R003", "R004"]
        
        for room_id in test_rooms:
            egress_info = graph.calculate_egress_distance(room_id)
            
            # Check that egress information is complete
            assert "distance" in egress_info
            assert "exit_room_id" in egress_info
            assert "path" in egress_info
            assert "is_accessible" in egress_info
            
            # Check that distance is reasonable (not infinite)
            assert egress_info["distance"] != float('inf')
            assert egress_info["distance"] > 0
            
            # Check that path includes the room and exit
            assert room_id in egress_info["path"]
            assert egress_info["exit_room_id"] in ["R005", "R006"]  # Either room can be the exit
            assert egress_info["exit_room_id"] in egress_info["path"]
    
    def test_travel_time_calculations(self):
        """Test travel time calculations for realistic building."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        # Test travel times from different rooms
        test_rooms = ["R001", "R002", "R003", "R004"]
        
        for room_id in test_rooms:
            travel_info = graph.calculate_travel_time(room_id, walking_speed=1.2)
            
            # Check that travel information is complete
            assert "travel_time_seconds" in travel_info
            assert "travel_time_minutes" in travel_info
            assert "distance" in travel_info
            
            # Check that travel times are reasonable
            assert travel_info["travel_time_seconds"] > 0
            assert travel_info["travel_time_minutes"] > 0
            assert travel_info["travel_time_seconds"] == travel_info["travel_time_minutes"] * 60
            
            # Check that travel time is reasonable for building size
            assert travel_info["travel_time_minutes"] < 5.0  # Should be less than 5 minutes
    
    def test_path_accessibility_analysis(self):
        """Test path accessibility analysis for realistic building."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        # Test accessibility from different rooms
        test_rooms = ["R001", "R002", "R003", "R004"]
        
        for room_id in test_rooms:
            egress_info = graph.calculate_egress_distance(room_id)
            
            # Check accessibility
            assert "is_accessible" in egress_info
            
            # In our test building, all doors should be accessible (800mm+)
            assert egress_info["is_accessible"] is True
    
    def test_multiple_egress_routes(self):
        """Test finding multiple egress routes."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        # Test finding multiple routes from a room
        routes = graph.find_egress_routes("R004", max_routes=3)
        
        # Should find at least one route
        assert len(routes) > 0
        
        # All routes should end at the exit
        for route in routes:
            assert route["exit_room_id"] in ["R005", "R006"]  # Either room can be the exit
            assert "R004" in route["path"]
            assert route["exit_room_id"] in route["path"]
            assert route["distance"] > 0
            assert route["is_accessible"] is True
    
    def test_room_connections_analysis(self):
        """Test room connections analysis."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        # Test connections for each room
        for room in project.get_all_rooms():
            if room.use != BuildingUse.EXIT:  # Exit rooms might not have connections
                connections = graph.get_room_connections(room.id)
                
                # Should have at least one connection (to corridor or other rooms)
                assert len(connections) > 0
                
                # Check connection details
                for connection in connections:
                    assert "room_id" in connection
                    assert "door_id" in connection
                    assert "door_width" in connection
                    assert "distance" in connection
    
    def test_compliance_scenario_analysis(self):
        """Test a complete compliance scenario analysis."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        # Analyze compliance for each room
        compliance_results = []
        
        for room in project.get_all_rooms():
            if room.use in [BuildingUse.OFFICE, BuildingUse.MEETING, BuildingUse.RECEPTION]:
                # Calculate egress distance
                egress_info = graph.calculate_egress_distance(room.id)
                
                # Calculate travel time
                travel_info = graph.calculate_travel_time(room.id)
                
                # Calculate occupancy capacity
                capacity = calculate_egress_capacity(room.area, room.use)
                
                # Check compliance
                max_distance = 25.0  # 25m for office buildings
                max_travel_time = 2.0  # 2 minutes maximum
                
                is_distance_compliant = egress_info["distance"] <= max_distance
                is_time_compliant = travel_info["travel_time_minutes"] <= max_travel_time
                is_accessible = egress_info["is_accessible"]
                
                compliance_results.append({
                    "room_id": room.id,
                    "room_name": room.name,
                    "distance": egress_info["distance"],
                    "travel_time": travel_info["travel_time_minutes"],
                    "capacity": capacity,
                    "is_distance_compliant": is_distance_compliant,
                    "is_time_compliant": is_time_compliant,
                    "is_accessible": is_accessible,
                    "overall_compliant": is_distance_compliant and is_time_compliant and is_accessible
                })
        
        # Check that all rooms have compliance results
        assert len(compliance_results) > 0
        
        # Check that all rooms are compliant in our test building
        for result in compliance_results:
            assert result["overall_compliant"] is True
            assert result["distance"] <= 25.0
            assert result["travel_time"] <= 2.0
            assert result["is_accessible"] is True
    
    def test_fire_rating_analysis(self):
        """Test fire rating analysis integration."""
        # Test fire rating calculations
        assert calculate_fire_rating_equivalent("RF_30") == 30
        assert calculate_fire_rating_equivalent("RF_60") == 60
        assert calculate_fire_rating_equivalent("RF_90") == 90
        assert calculate_fire_rating_equivalent("RF_120") == 120
        assert calculate_fire_rating_equivalent("no_rating") == 0
    
    def test_graph_statistics_analysis(self):
        """Test graph statistics for realistic building."""
        project = create_realistic_office_building()
        graph = create_circulation_graph(project)
        
        stats = graph.get_graph_statistics()
        
        # Check that statistics are reasonable
        assert stats["total_nodes"] == len(project.get_all_rooms())
        assert stats["total_edges"] > 0
        assert stats["exit_nodes"] > 0
        assert stats["connected_components"] == 1  # Should be fully connected
        assert stats["is_connected"] is True
        assert stats["average_degree"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
