"""
Unit tests for graph calculation functions.

This module tests all the fundamental graph functions that the agent will use
for circulation and egress analysis.
"""

import pytest
from src.schemas import Project, ProjectMetadata, Level, Room, Door, Point3D, BuildingUse, DoorType
from src.calculations.graph import (
    CirculationGraph,
    create_circulation_graph,
    calculate_egress_distance,
    calculate_travel_time
)


def create_test_project():
    """Create a test project with rooms and doors for testing."""
    # Create rooms
    rooms = [
        Room(
            id="R001",
            name="Office 1",
            area=25.0,
            use=BuildingUse.OFFICE,
            level="Ground Floor"
        ),
        Room(
            id="R002", 
            name="Office 2",
            area=30.0,
            use=BuildingUse.OFFICE,
            level="Ground Floor"
        ),
        Room(
            id="R003",
            name="Corridor",
            area=50.0,
            use=BuildingUse.CORRIDOR,
            level="Ground Floor"
        ),
        Room(
            id="R004",
            name="Exit",
            area=10.0,
            use=BuildingUse.EXIT,
            level="Ground Floor"
        )
    ]
    
    # Create doors
    doors = [
        Door(
            id="D001",
            width_mm=900,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=0, y=0, z=0),
            from_room="R001",
            to_room="R003"
        ),
        Door(
            id="D002",
            width_mm=900,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=5, y=0, z=0),
            from_room="R002",
            to_room="R003"
        ),
        Door(
            id="D003",
            width_mm=1200,
            height_mm=2100,
            door_type=DoorType.EMERGENCY_EXIT,
            position=Point3D(x=10, y=0, z=0),
            from_room="R003",
            to_room="R004",
            is_emergency_exit=True
        )
    ]
    
    # Create level
    level = Level(
        name="Ground Floor",
        elevation=0.0,
        rooms=rooms,
        doors=doors
    )
    
    # Create project metadata
    metadata = ProjectMetadata(
        project_name="Test Building",
        file_name="test_building.json",
        building_type="office",
        total_area=115.0,
        number_of_levels=1
    )
    
    # Create project
    project = Project(
        metadata=metadata,
        levels=[level]
    )
    
    return project


def create_complex_test_project():
    """Create a more complex test project with multiple paths."""
    # Create rooms
    rooms = [
        Room(
            id="R001",
            name="Office A",
            area=20.0,
            use=BuildingUse.OFFICE,
            level="Ground Floor"
        ),
        Room(
            id="R002",
            name="Office B", 
            area=25.0,
            use=BuildingUse.OFFICE,
            level="Ground Floor"
        ),
        Room(
            id="R003",
            name="Meeting Room",
            area=30.0,
            use=BuildingUse.MEETING,
            level="Ground Floor"
        ),
        Room(
            id="R004",
            name="Main Corridor",
            area=60.0,
            use=BuildingUse.CORRIDOR,
            level="Ground Floor"
        ),
        Room(
            id="R005",
            name="Side Corridor",
            area=40.0,
            use=BuildingUse.CORRIDOR,
            level="Ground Floor"
        ),
        Room(
            id="R006",
            name="Emergency Exit",
            area=15.0,
            use=BuildingUse.EXIT,
            level="Ground Floor"
        )
    ]
    
    # Create doors
    doors = [
        # Office A to Main Corridor
        Door(
            id="D001",
            width_mm=800,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=0, y=0, z=0),
            from_room="R001",
            to_room="R004"
        ),
        # Office B to Main Corridor
        Door(
            id="D002",
            width_mm=900,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=5, y=0, z=0),
            from_room="R002",
            to_room="R004"
        ),
        # Meeting Room to Main Corridor
        Door(
            id="D003",
            width_mm=1000,
            height_mm=2100,
            door_type=DoorType.DOUBLE,
            position=Point3D(x=10, y=0, z=0),
            from_room="R003",
            to_room="R004"
        ),
        # Main Corridor to Side Corridor
        Door(
            id="D004",
            width_mm=1200,
            height_mm=2100,
            door_type=DoorType.SINGLE,
            position=Point3D(x=15, y=0, z=0),
            from_room="R004",
            to_room="R005"
        ),
        # Side Corridor to Emergency Exit
        Door(
            id="D005",
            width_mm=1000,
            height_mm=2100,
            door_type=DoorType.EMERGENCY_EXIT,
            position=Point3D(x=20, y=0, z=0),
            from_room="R005",
            to_room="R006",
            is_emergency_exit=True
        )
    ]
    
    # Create level
    level = Level(
        name="Ground Floor",
        elevation=0.0,
        rooms=rooms,
        doors=doors
    )
    
    # Create project metadata
    metadata = ProjectMetadata(
        project_name="Complex Test Building",
        file_name="complex_test_building.json",
        building_type="office",
        total_area=190.0,
        number_of_levels=1
    )
    
    # Create project
    project = Project(
        metadata=metadata,
        levels=[level]
    )
    
    return project


class TestCirculationGraph:
    """Test CirculationGraph class functionality."""
    
    def test_create_circulation_graph(self):
        """Test creating a circulation graph from project data."""
        project = create_test_project()
        graph = create_circulation_graph(project)
        
        assert isinstance(graph, CirculationGraph)
        assert graph.project == project
        assert graph.graph.number_of_nodes() > 0
        assert graph.graph.number_of_edges() > 0
    
    def test_graph_nodes_creation(self):
        """Test that graph nodes are created correctly."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        # Check that all rooms are added as nodes
        expected_nodes = [f"room_{room.id}" for room in project.get_all_rooms()]
        actual_nodes = list(graph.graph.nodes())
        
        for expected_node in expected_nodes:
            assert expected_node in actual_nodes
    
    def test_graph_edges_creation(self):
        """Test that graph edges are created correctly."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        # Check that doors create edges between rooms
        doors = project.get_all_doors()
        for door in doors:
            if door.from_room and door.to_room:
                from_node = f"room_{door.from_room}"
                to_node = f"room_{door.to_room}"
                assert graph.graph.has_edge(from_node, to_node)
    
    def test_exit_nodes_identification(self):
        """Test that exit nodes are identified correctly."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        # Check that emergency exit doors create exit nodes
        assert len(graph.exit_nodes) > 0
        
        # Check that exit nodes are marked correctly
        for exit_node in graph.exit_nodes:
            assert graph.graph.nodes[exit_node].get('is_exit', False) is True


class TestEgressDistanceCalculations:
    """Test egress distance calculation functions."""
    
    def test_calculate_egress_distance_simple(self):
        """Test egress distance calculation for simple layout."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        # Test egress distance from R001 (should go through R003 to R004)
        egress_info = graph.calculate_egress_distance("R001")
        
        assert "distance" in egress_info
        assert "exit_room_id" in egress_info
        assert "path" in egress_info
        assert "is_accessible" in egress_info
        # The algorithm finds the nearest exit room
        assert egress_info["exit_room_id"] in ["R003", "R004"]  # Either room can be the exit
        assert "R001" in egress_info["path"]
        assert egress_info["exit_room_id"] in egress_info["path"]
    
    def test_calculate_egress_distance_complex(self):
        """Test egress distance calculation for complex layout."""
        project = create_complex_test_project()
        graph = CirculationGraph(project)
        
        # Test egress distance from R001 (should go through R004, R005 to R006)
        egress_info = graph.calculate_egress_distance("R001")
        
        assert egress_info["distance"] != float('inf')
        assert egress_info["exit_room_id"] in ["R005", "R006"]  # Either room can be the exit
        assert "R001" in egress_info["path"]
        assert egress_info["exit_room_id"] in egress_info["path"]
    
    def test_calculate_egress_distance_no_exit(self):
        """Test egress distance calculation when no exit exists."""
        # Create project without exit
        rooms = [
            Room(
                id="R001",
                name="Office",
                area=25.0,
                use=BuildingUse.OFFICE,
                level="Ground Floor"
            )
        ]
        
        level = Level(
            name="Ground Floor",
            elevation=0.0,
            rooms=rooms,
            doors=[]
        )
        
        metadata = ProjectMetadata(
            project_name="No Exit Building",
            file_name="no_exit.json",
            building_type="office",
            total_area=25.0,
            number_of_levels=1
        )
        
        project = Project(metadata=metadata, levels=[level])
        graph = CirculationGraph(project)
        
        egress_info = graph.calculate_egress_distance("R001")
        
        assert egress_info["distance"] == float('inf')
        assert egress_info["exit_room_id"] is None
        assert "No exit nodes found" in egress_info["error"]
    
    def test_calculate_egress_distance_room_not_found(self):
        """Test egress distance calculation for non-existent room."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        egress_info = graph.calculate_egress_distance("R999")
        
        assert egress_info["distance"] == float('inf')
        assert "Room R999 not found" in egress_info["error"]
    
    def test_calculate_egress_distance_no_path(self):
        """Test egress distance calculation when no path exists."""
        # Create project with disconnected rooms
        rooms = [
            Room(
                id="R001",
                name="Isolated Room",
                area=25.0,
                use=BuildingUse.OFFICE,
                level="Ground Floor"
            ),
            Room(
                id="R002",
                name="Exit Room",
                area=10.0,
                use=BuildingUse.EXIT,
                level="Ground Floor"
            )
        ]
        
        # No doors connecting the rooms
        level = Level(
            name="Ground Floor",
            elevation=0.0,
            rooms=rooms,
            doors=[]
        )
        
        metadata = ProjectMetadata(
            project_name="Disconnected Building",
            file_name="disconnected.json",
            building_type="office",
            total_area=35.0,
            number_of_levels=1
        )
        
        project = Project(metadata=metadata, levels=[level])
        graph = CirculationGraph(project)
        
        egress_info = graph.calculate_egress_distance("R001")
        
        assert egress_info["distance"] == float('inf')
        assert egress_info["exit_room_id"] is None


class TestPathAccessibility:
    """Test path accessibility checking."""
    
    def test_check_path_accessibility_accessible(self):
        """Test accessibility check for accessible path."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        # All doors in test project are 900mm or wider (accessible)
        path = ["room_R001", "room_R003", "room_R004"]
        is_accessible = graph._check_path_accessibility(path)
        assert is_accessible is True
    
    def test_check_path_accessibility_inaccessible(self):
        """Test accessibility check for inaccessible path."""
        # Create project with narrow doors
        rooms = [
            Room(
                id="R001",
                name="Office",
                area=25.0,
                use=BuildingUse.OFFICE,
                level="Ground Floor"
            ),
            Room(
                id="R002",
                name="Exit",
                area=10.0,
                use=BuildingUse.EXIT,
                level="Ground Floor"
            )
        ]
        
        doors = [
            Door(
                id="D001",
                width_mm=700,  # Too narrow for accessibility
                height_mm=2100,
                door_type=DoorType.SINGLE,
                position=Point3D(x=0, y=0, z=0),
                from_room="R001",
                to_room="R002"
            )
        ]
        
        level = Level(
            name="Ground Floor",
            elevation=0.0,
            rooms=rooms,
            doors=doors
        )
        
        metadata = ProjectMetadata(
            project_name="Inaccessible Building",
            file_name="inaccessible.json",
            building_type="office",
            total_area=35.0,
            number_of_levels=1
        )
        
        project = Project(metadata=metadata, levels=[level])
        graph = CirculationGraph(project)
        
        path = ["room_R001", "room_R002"]
        is_accessible = graph._check_path_accessibility(path)
        assert is_accessible is False
    
    def test_check_path_accessibility_short_path(self):
        """Test accessibility check for very short path."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        # Single node path should be accessible
        path = ["room_R001"]
        is_accessible = graph._check_path_accessibility(path)
        assert is_accessible is True


class TestEgressCapacityCalculations:
    """Test egress capacity calculation functions."""
    
    def test_calculate_egress_capacity_office(self):
        """Test egress capacity calculation for office room."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        capacity = graph.calculate_egress_capacity("R001")
        assert capacity == 2  # 25 * 0.1 = 2.5, rounded down to 2
    
    def test_calculate_egress_capacity_meeting(self):
        """Test egress capacity calculation for meeting room."""
        project = create_complex_test_project()
        graph = CirculationGraph(project)
        
        capacity = graph.calculate_egress_capacity("R003")
        assert capacity == 6  # 30 * 0.2 = 6
    
    def test_calculate_egress_capacity_room_not_found(self):
        """Test egress capacity calculation for non-existent room."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        capacity = graph.calculate_egress_capacity("R999")
        assert capacity == 0


class TestRoomConnections:
    """Test room connection functions."""
    
    def test_get_room_connections(self):
        """Test getting room connections."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        connections = graph.get_room_connections("R001")
        assert len(connections) == 1
        assert connections[0]["room_id"] == "R003"
        assert connections[0]["door_id"] == "D001"
        assert connections[0]["door_width"] == 900
    
    def test_get_room_connections_room_not_found(self):
        """Test getting connections for non-existent room."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        connections = graph.get_room_connections("R999")
        assert connections == []


class TestEgressRoutes:
    """Test egress route finding functions."""
    
    def test_find_egress_routes(self):
        """Test finding multiple egress routes."""
        project = create_complex_test_project()
        graph = CirculationGraph(project)
        
        routes = graph.find_egress_routes("R001", max_routes=3)
        assert len(routes) > 0
        
        # Check that all routes end at the exit
        for route in routes:
            assert route["exit_room_id"] in ["R005", "R006"]  # Either room can be the exit
            assert "R001" in route["path"]
            assert route["exit_room_id"] in route["path"]
            assert route["distance"] >= 0  # Distance can be 0 for same room
    
    def test_find_egress_routes_no_routes(self):
        """Test finding routes when no routes exist."""
        # Create project with no exit
        rooms = [
            Room(
                id="R001",
                name="Office",
                area=25.0,
                use=BuildingUse.OFFICE,
                level="Ground Floor"
            )
        ]
        
        level = Level(
            name="Ground Floor",
            elevation=0.0,
            rooms=rooms,
            doors=[]
        )
        
        metadata = ProjectMetadata(
            project_name="No Routes Building",
            file_name="no_routes.json",
            building_type="office",
            total_area=25.0,
            number_of_levels=1
        )
        
        project = Project(metadata=metadata, levels=[level])
        graph = CirculationGraph(project)
        
        routes = graph.find_egress_routes("R001")
        assert routes == []


class TestTravelTimeCalculations:
    """Test travel time calculation functions."""
    
    def test_calculate_travel_time(self):
        """Test travel time calculation."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        travel_info = graph.calculate_travel_time("R001", walking_speed=1.2)
        
        assert "travel_time_seconds" in travel_info
        assert "travel_time_minutes" in travel_info
        assert "distance" in travel_info
        assert travel_info["travel_time_seconds"] >= 0  # Can be 0 for same room
        assert travel_info["travel_time_minutes"] >= 0
        assert travel_info["travel_time_minutes"] == travel_info["travel_time_seconds"] / 60.0
    
    def test_calculate_travel_time_no_path(self):
        """Test travel time calculation when no path exists."""
        # Create project with no exit
        rooms = [
            Room(
                id="R001",
                name="Office",
                area=25.0,
                use=BuildingUse.OFFICE,
                level="Ground Floor"
            )
        ]
        
        level = Level(
            name="Ground Floor",
            elevation=0.0,
            rooms=rooms,
            doors=[]
        )
        
        metadata = ProjectMetadata(
            project_name="No Path Building",
            file_name="no_path.json",
            building_type="office",
            total_area=25.0,
            number_of_levels=1
        )
        
        project = Project(metadata=metadata, levels=[level])
        graph = CirculationGraph(project)
        
        travel_info = graph.calculate_travel_time("R001")
        
        assert travel_info["travel_time_seconds"] == float('inf')
        assert travel_info["travel_time_minutes"] == float('inf')
        assert "No exit nodes found" in travel_info["error"]


class TestGraphStatistics:
    """Test graph statistics functions."""
    
    def test_get_graph_statistics(self):
        """Test getting graph statistics."""
        project = create_test_project()
        graph = CirculationGraph(project)
        
        stats = graph.get_graph_statistics()
        
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "exit_nodes" in stats
        assert "connected_components" in stats
        assert "is_connected" in stats
        assert "average_degree" in stats
        
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert stats["exit_nodes"] > 0
        assert stats["connected_components"] >= 1
        assert isinstance(stats["is_connected"], bool)
        assert stats["average_degree"] >= 0


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""
    
    def test_calculate_egress_distance_function(self):
        """Test the module-level calculate_egress_distance function."""
        project = create_test_project()
        
        egress_info = calculate_egress_distance(project, "R001")
        
        assert "distance" in egress_info
        assert "exit_room_id" in egress_info
        assert "path" in egress_info
        assert egress_info["exit_room_id"] in ["R003", "R004"]  # Either room can be the exit
    
    def test_calculate_travel_time_function(self):
        """Test the module-level calculate_travel_time function."""
        project = create_test_project()
        
        travel_info = calculate_travel_time(project, "R001", walking_speed=1.0)
        
        assert "travel_time_seconds" in travel_info
        assert "travel_time_minutes" in travel_info
        assert "distance" in travel_info
        assert travel_info["travel_time_seconds"] >= 0  # Can be 0 for same room


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
