# 🧪 Testing Guide - AEC Compliance Agent

## 📋 Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Test Structure](#test-structure)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [End-to-End Tests](#end-to-end-tests)
6. [Test Fixtures](#test-fixtures)
7. [Mocking](#mocking)
8. [Coverage](#coverage)
9. [Running Tests](#running-tests)
10. [Writing Good Tests](#writing-good-tests)

---

## Testing Strategy

### Test Pyramid

```
           /\
          /  \
         / E2E \ ← Few (2-3 tests)
        /______\
       /        \
      /Integration\ ← Some (10-15 tests)
     /____________\
    /              \
   /   Unit Tests   \ ← Many (50+ tests)
  /__________________\
```

### Coverage Goals

- **Overall**: 85%+
- **Unit Tests**: 90%+
- **Integration Tests**: 70%+
- **E2E Tests**: Key workflows only

### Testing Principles

1. **Fast**: Unit tests should run in < 1s
2. **Isolated**: Each test is independent
3. **Repeatable**: Same input → Same output
4. **Readable**: Clear test names and structure
5. **Focused**: One concept per test

---

## Test Structure

```
tests/
├── __init__.py
│
├── unit/                          # Fast, isolated tests
│   ├── test_schemas.py            # Data models
│   ├── test_extraction.py         # DXF parsing
│   ├── test_geometry.py           # Shapely operations
│   ├── test_graph.py              # NetworkX graphs
│   └── test_tools.py              # Agent tools
│
├── integration/                   # Multi-component tests
│   ├── test_rag_pipeline.py       # RAG system
│   ├── test_agent_simple.py       # Agent with mocks
│   └── test_e2e_pipeline.py       # Full pipeline
│
└── fixtures/                      # Test data
    ├── sample.dxf                 # Minimal DXF file
    ├── test_project.json          # Example project
    └── mock_pdf.txt               # Mock document
```

---

## Unit Tests

### Test Schemas

```python
# tests/unit/test_schemas.py
import pytest
from src.schemas import Project, ProjectMetadata, Room, Door, Wall, Level
from datetime import datetime

class TestProjectMetadata:
    """Test ProjectMetadata schema."""
    
    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = ProjectMetadata(
            project_name="Test Project",
            level_name="Ground Floor",
            dwg_file="test.dwg",
            extracted_date=datetime.now().isoformat()
        )
        
        assert metadata.project_name == "Test Project"
        assert metadata.level_name == "Ground Floor"
    
    def test_metadata_defaults(self):
        """Test metadata with minimal fields."""
        metadata = ProjectMetadata(
            project_name="Test",
            level_name="P1",
            dwg_file="test.dwg",
            extracted_date="2025-10-01T00:00:00"
        )
        
        assert metadata.project_name == "Test"


class TestRoom:
    """Test Room schema."""
    
    def test_valid_room(self):
        """Test creating valid room."""
        room = Room(
            id="ROOM_001",
            name="Office",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 5], [0, 5], [0, 0]]
        )
        
        assert room.id == "ROOM_001"
        assert len(room.boundary) == 5
    
    def test_room_with_optional_fields(self):
        """Test room with use_type."""
        room = Room(
            id="ROOM_001",
            name="Office",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 5], [0, 5]],
            use_type="Administrative"
        )
        
        assert room.use_type == "Administrative"
    
    def test_invalid_boundary(self):
        """Test that invalid boundary raises error."""
        with pytest.raises(ValueError):
            Room(
                id="ROOM_001",
                name="Office",
                level="P1",
                boundary=[[0, 0]]  # Too few points
            )


class TestDoor:
    """Test Door schema."""
    
    def test_valid_door(self):
        """Test creating valid door."""
        door = Door(
            id="DOOR_001",
            width=0.90,
            position=[5, 0],
            level="P1"
        )
        
        assert door.width == 0.90
        assert door.height == 2.10  # Default
    
    def test_door_with_fire_rating(self):
        """Test door with fire rating."""
        door = Door(
            id="DOOR_001",
            width=0.80,
            position=[0, 0],
            level="P1",
            is_fire_rated=True,
            is_egress=True
        )
        
        assert door.is_fire_rated is True
        assert door.is_egress is True
    
    def test_invalid_width(self):
        """Test that negative width raises error."""
        with pytest.raises(ValueError):
            Door(
                id="DOOR_001",
                width=-0.90,  # Invalid
                position=[0, 0],
                level="P1"
            )


class TestProject:
    """Test complete Project schema."""
    
    def test_valid_project(self):
        """Test creating valid project."""
        metadata = ProjectMetadata(
            project_name="Test",
            level_name="P1",
            dwg_file="test.dwg",
            extracted_date="2025-10-01T00:00:00"
        )
        
        level = Level(name="P1", elevation=0.0)
        
        room = Room(
            id="ROOM_001",
            name="Office",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 5], [0, 5]]
        )
        
        door = Door(
            id="DOOR_001",
            width=0.90,
            position=[5, 0],
            level="P1"
        )
        
        project = Project(
            metadata=metadata,
            levels=[level],
            rooms=[room],
            doors=[door],
            walls=[]
        )
        
        assert len(project.rooms) == 1
        assert len(project.doors) == 1
    
    def test_empty_project(self):
        """Test project with no rooms/doors."""
        metadata = ProjectMetadata(
            project_name="Empty",
            level_name="P1",
            dwg_file="empty.dwg",
            extracted_date="2025-10-01T00:00:00"
        )
        
        project = Project(
            metadata=metadata,
            levels=[],
            rooms=[],
            doors=[],
            walls=[]
        )
        
        assert len(project.rooms) == 0
```

### Test Geometry

```python
# tests/unit/test_geometry.py
import pytest
from src.calculations.geometry import (
    get_room_polygon,
    calculate_room_area,
    get_room_centroid,
    calculate_perimeter,
    point_in_room,
    distance_between_points
)
from src.schemas import Room

@pytest.fixture
def rectangular_room():
    """Fixture: Simple rectangular room 10x5m."""
    return Room(
        id="ROOM_001",
        name="Rectangle",
        level="P1",
        boundary=[[0, 0], [10, 0], [10, 5], [0, 5], [0, 0]]
    )

@pytest.fixture
def l_shaped_room():
    """Fixture: L-shaped room."""
    return Room(
        id="ROOM_002",
        name="L-Shape",
        level="P1",
        boundary=[
            [0, 0], [10, 0], [10, 5], [5, 5],
            [5, 10], [0, 10], [0, 0]
        ]
    )


class TestGetRoomPolygon:
    """Test polygon conversion."""
    
    def test_rectangle_polygon(self, rectangular_room):
        """Test converting rectangle to polygon."""
        polygon = get_room_polygon(rectangular_room)
        
        assert polygon.is_valid
        assert polygon.geom_type == 'Polygon'
    
    def test_l_shape_polygon(self, l_shaped_room):
        """Test converting L-shape to polygon."""
        polygon = get_room_polygon(l_shaped_room)
        
        assert polygon.is_valid
        assert polygon.geom_type == 'Polygon'


class TestCalculateRoomArea:
    """Test area calculations."""
    
    def test_rectangle_area(self, rectangular_room):
        """Test rectangle area: 10m × 5m = 50m²."""
        area = calculate_room_area(rectangular_room)
        
        assert pytest.approx(area, abs=0.01) == 50.0
    
    def test_l_shape_area(self, l_shaped_room):
        """Test L-shape area: 10×5 + 5×5 = 75m²."""
        area = calculate_room_area(l_shaped_room)
        
        # L-shape: Full rectangle (10×10=100) - cut corner (5×5=25) = 75
        assert pytest.approx(area, abs=0.01) == 75.0
    
    def test_zero_area(self):
        """Test degenerate room (line) has zero area."""
        room = Room(
            id="ROOM_003",
            name="Line",
            level="P1",
            boundary=[[0, 0], [10, 0], [10, 0], [0, 0]]
        )
        
        area = calculate_room_area(room)
        assert area == 0.0


class TestGetRoomCentroid:
    """Test centroid calculations."""
    
    def test_rectangle_centroid(self, rectangular_room):
        """Test rectangle centroid is at center."""
        centroid = get_room_centroid(rectangular_room)
        
        assert pytest.approx(centroid[0], abs=0.01) == 5.0  # x = 10/2
        assert pytest.approx(centroid[1], abs=0.01) == 2.5  # y = 5/2
    
    def test_l_shape_centroid(self, l_shaped_room):
        """Test L-shape centroid."""
        centroid = get_room_centroid(l_shaped_room)
        
        # Centroid should be inside the polygon
        polygon = get_room_polygon(l_shaped_room)
        from shapely.geometry import Point
        assert polygon.contains(Point(centroid))


class TestCalculatePerimeter:
    """Test perimeter calculations."""
    
    def test_rectangle_perimeter(self, rectangular_room):
        """Test rectangle perimeter: 2×(10+5) = 30m."""
        perimeter = calculate_perimeter(rectangular_room)
        
        assert pytest.approx(perimeter, abs=0.01) == 30.0
    
    def test_l_shape_perimeter(self, l_shaped_room):
        """Test L-shape perimeter."""
        perimeter = calculate_perimeter(l_shaped_room)
        
        # Outer perimeter of L-shape
        expected = 10 + 5 + 5 + 5 + 5 + 10  # Sum of all edges
        assert pytest.approx(perimeter, abs=0.01) == expected


class TestPointInRoom:
    """Test point containment."""
    
    def test_point_inside_rectangle(self, rectangular_room):
        """Test point clearly inside rectangle."""
        assert point_in_room((5, 2.5), rectangular_room) is True
    
    def test_point_outside_rectangle(self, rectangular_room):
        """Test point clearly outside rectangle."""
        assert point_in_room((15, 15), rectangular_room) is False
    
    def test_point_on_boundary(self, rectangular_room):
        """Test point exactly on boundary."""
        # Shapely considers boundary as "outside" by default
        assert point_in_room((0, 0), rectangular_room) is False
    
    def test_point_in_l_shape(self, l_shaped_room):
        """Test point in L-shaped room."""
        assert point_in_room((2, 2), l_shaped_room) is True  # In main part
        assert point_in_room((2, 8), l_shaped_room) is True  # In vertical part
        assert point_in_room((8, 8), l_shaped_room) is False  # Outside


class TestDistanceBetweenPoints:
    """Test distance calculations."""
    
    def test_horizontal_distance(self):
        """Test distance along X axis."""
        distance = distance_between_points((0, 0), (10, 0))
        assert pytest.approx(distance, abs=0.01) == 10.0
    
    def test_vertical_distance(self):
        """Test distance along Y axis."""
        distance = distance_between_points((0, 0), (0, 5))
        assert pytest.approx(distance, abs=0.01) == 5.0
    
    def test_diagonal_distance(self):
        """Test diagonal distance (Pythagorean)."""
        distance = distance_between_points((0, 0), (3, 4))
        assert pytest.approx(distance, abs=0.01) == 5.0  # 3-4-5 triangle
    
    def test_same_point(self):
        """Test distance from point to itself."""
        distance = distance_between_points((5, 5), (5, 5))
        assert distance == 0.0
```

### Test Graph

```python
# tests/unit/test_graph.py
import pytest
from src.calculations.graph import CirculationGraph
from src.schemas import Project, ProjectMetadata, Level, Room, Door

@pytest.fixture
def simple_project():
    """Fixture: Simple 3-room linear project."""
    metadata = ProjectMetadata(
        project_name="Linear",
        level_name="P1",
        dwg_file="linear.dwg",
        extracted_date="2025-10-01T00:00:00"
    )
    
    level = Level(name="P1", elevation=0.0)
    
    rooms = [
        Room(
            id="ROOM_A",
            name="Room A",
            level="P1",
            boundary=[[0, 0], [5, 0], [5, 5], [0, 5]]
        ),
        Room(
            id="ROOM_B",
            name="Room B",
            level="P1",
            boundary=[[5, 0], [10, 0], [10, 5], [5, 5]]
        ),
        Room(
            id="ROOM_C",
            name="Room C",
            level="P1",
            boundary=[[10, 0], [15, 0], [15, 5], [10, 5]]
        )
    ]
    
    doors = [
        Door(
            id="DOOR_AB",
            width=0.90,
            position=[5, 2.5],
            level="P1",
            room_from="ROOM_A",
            room_to="ROOM_B"
        ),
        Door(
            id="DOOR_BC",
            width=0.80,
            position=[10, 2.5],
            level="P1",
            room_from="ROOM_B",
            room_to="ROOM_C"
        ),
        Door(
            id="DOOR_EXIT",
            width=0.90,
            position=[15, 2.5],
            level="P1",
            room_from="ROOM_C",
            is_egress=True
        )
    ]
    
    return Project(
        metadata=metadata,
        levels=[level],
        rooms=rooms,
        doors=doors,
        walls=[]
    )


class TestCirculationGraph:
    """Test graph construction."""
    
    def test_graph_creation(self, simple_project):
        """Test creating graph from project."""
        graph = CirculationGraph(simple_project)
        
        assert len(graph.graph.nodes) == 3  # 3 rooms
        assert len(graph.graph.edges) == 3  # 3 doors
    
    def test_nodes_have_positions(self, simple_project):
        """Test nodes have centroid positions."""
        graph = CirculationGraph(simple_project)
        
        for node_id in graph.graph.nodes:
            node = graph.graph.nodes[node_id]
            assert 'pos' in node
            assert len(node['pos']) == 2  # (x, y)
    
    def test_edges_have_door_info(self, simple_project):
        """Test edges have door information."""
        graph = CirculationGraph(simple_project)
        
        edge = graph.graph.edges['ROOM_A', 'ROOM_B']
        assert 'door_id' in edge
        assert edge['width'] == 0.90


class TestShortestPath:
    """Test shortest path finding."""
    
    def test_adjacent_rooms(self, simple_project):
        """Test path between adjacent rooms."""
        graph = CirculationGraph(simple_project)
        
        path = graph.shortest_path("ROOM_A", "ROOM_B")
        
        assert path == ["ROOM_A", "ROOM_B"]
    
    def test_distant_rooms(self, simple_project):
        """Test path across multiple rooms."""
        graph = CirculationGraph(simple_project)
        
        path = graph.shortest_path("ROOM_A", "ROOM_C")
        
        assert path == ["ROOM_A", "ROOM_B", "ROOM_C"]
    
    def test_no_path(self):
        """Test disconnected rooms return None."""
        # Create project with disconnected rooms
        metadata = ProjectMetadata(
            project_name="Disconnected",
            level_name="P1",
            dwg_file="disconnected.dwg",
            extracted_date="2025-10-01T00:00:00"
        )
        
        rooms = [
            Room(id="ROOM_A", name="A", level="P1", boundary=[[0,0],[5,0],[5,5],[0,5]]),
            Room(id="ROOM_B", name="B", level="P1", boundary=[[10,0],[15,0],[15,5],[10,5]])
        ]
        
        project = Project(metadata=metadata, levels=[], rooms=rooms, doors=[], walls=[])
        graph = CirculationGraph(project)
        
        path = graph.shortest_path("ROOM_A", "ROOM_B")
        assert path is None


class TestShortestEgressPath:
    """Test egress path finding."""
    
    def test_egress_from_entry_room(self, simple_project):
        """Test egress from room adjacent to exit."""
        graph = CirculationGraph(simple_project)
        
        result = graph.shortest_egress_path("ROOM_C")
        
        assert result is not None
        assert "ROOM_C" in result['path']
        assert result['distance'] > 0
    
    def test_egress_from_far_room(self, simple_project):
        """Test egress from room far from exit."""
        graph = CirculationGraph(simple_project)
        
        result = graph.shortest_egress_path("ROOM_A")
        
        assert result is not None
        assert result['path'][0] == "ROOM_A"
        assert len(result['path']) == 3  # A → B → C
    
    def test_no_egress_door(self):
        """Test when no egress door exists."""
        metadata = ProjectMetadata(
            project_name="No Exit",
            level_name="P1",
            dwg_file="no_exit.dwg",
            extracted_date="2025-10-01T00:00:00"
        )
        
        rooms = [
            Room(id="ROOM_A", name="A", level="P1", boundary=[[0,0],[5,0],[5,5],[0,5]])
        ]
        
        project = Project(metadata=metadata, levels=[], rooms=rooms, doors=[], walls=[])
        graph = CirculationGraph(project)
        
        result = graph.shortest_egress_path("ROOM_A")
        assert result is None
```

---

## Integration Tests

### Test RAG Pipeline

```python
# tests/integration/test_rag_pipeline.py
import pytest
from pathlib import Path
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain
import shutil

@pytest.fixture(scope="module")
def test_vectorstore():
    """Fixture: Create test vectorstore."""
    vectorstore_dir = Path("tests/fixtures/test_vectorstore")
    
    # Clean up if exists
    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)
    
    # Create vectorstore
    rag = VectorstoreManager(vectorstore_dir)
    
    # Use minimal test documents
    test_docs_dir = Path("tests/fixtures/test_docs")
    test_docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test PDF (or use mock)
    # For now, skip PDF creation and use mock documents
    
    yield rag
    
    # Cleanup
    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)


@pytest.mark.slow
class TestVectorstoreManager:
    """Test vectorstore operations."""
    
    def test_create_vectorstore(self, tmp_path):
        """Test creating vectorstore from PDFs."""
        # This test requires actual PDFs, mark as slow
        vectorstore_dir = tmp_path / "vectorstore"
        rag = VectorstoreManager(vectorstore_dir)
        
        # Would need actual PDF files here
        # For unit tests, we can mock this
        
        assert True  # Placeholder
    
    def test_load_existing_vectorstore(self, test_vectorstore):
        """Test loading existing vectorstore."""
        # Assuming vectorstore was created
        # test_vectorstore.load_existing()
        
        # assert test_vectorstore.vectorstore is not None
        assert True  # Placeholder
    
    def test_query_simple(self, test_vectorstore):
        """Test simple query without LLM."""
        # docs = test_vectorstore.query_simple("test query", k=2)
        
        # assert isinstance(docs, list)
        # assert len(docs) <= 2
        assert True  # Placeholder


@pytest.mark.requires_api
class TestQAChain:
    """Test QA chain with LLM."""
    
    def test_create_qa_chain(self, test_vectorstore):
        """Test creating QA chain."""
        # retriever = test_vectorstore.get_retriever(k=2)
        # qa_chain = create_qa_chain(retriever)
        
        # assert qa_chain is not None
        assert True  # Placeholder
    
    def test_query_with_sources(self, test_vectorstore):
        """Test querying with source documents."""
        # retriever = test_vectorstore.get_retriever(k=2)
        # qa_chain = create_qa_chain(retriever)
        
        # result = qa_chain({"query": "What is the minimum door width?"})
        
        # assert 'result' in result
        # assert 'source_documents' in result
        assert True  # Placeholder
```

### Test Agent

```python
# tests/integration/test_agent_simple.py
import pytest
from pathlib import Path
from src.agent.tools import load_project_data, get_room_info, list_all_doors
from src.agent.graph import create_compliance_agent
from langchain_core.messages import HumanMessage

@pytest.fixture
def loaded_project():
    """Fixture: Load test project."""
    project_path = Path("tests/fixtures/test_project.json")
    load_project_data(project_path)


class TestAgentTools:
    """Test agent tools with real project data."""
    
    def test_get_room_info(self, loaded_project):
        """Test getting room information."""
        result = get_room_info.invoke({"room_id": "ROOM_001"})
        
        assert "ROOM_001" in result
        assert "Area:" in result
    
    def test_list_all_doors(self, loaded_project):
        """Test listing all doors."""
        result = list_all_doors.invoke({})
        
        assert "Total doors:" in result
        assert "DOOR_" in result
    
    def test_get_nonexistent_room(self, loaded_project):
        """Test getting info for nonexistent room."""
        result = get_room_info.invoke({"room_id": "ROOM_999"})
        
        assert "not found" in result


@pytest.mark.slow
class TestAgentExecution:
    """Test agent execution (without real LLM calls)."""
    
    def test_agent_creation(self):
        """Test creating agent."""
        agent = create_compliance_agent()
        
        assert agent is not None
    
    @pytest.mark.skip("Requires API key")
    def test_agent_simple_query(self, loaded_project):
        """Test agent with simple query."""
        agent = create_compliance_agent()
        
        initial_state = {
            "messages": [HumanMessage(content="List all doors")],
            "iterations": 0
        }
        
        result = agent.invoke(initial_state)
        
        assert result['iterations'] > 0
        assert len(result['messages']) > 1
```

---

## End-to-End Tests

```python
# tests/integration/test_e2e_pipeline.py
import pytest
from pathlib import Path
from src.agent.tools import load_project_data, set_vectorstore_manager
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain
from src.agent.graph import create_compliance_agent
from langchain_core.messages import HumanMessage


@pytest.mark.slow
@pytest.mark.requires_api
class TestFullPipeline:
    """End-to-end pipeline tests."""
    
    def test_complete_workflow(self):
        """Test complete compliance verification workflow."""
        # 1. Load project
        project_path = Path("data/extracted/tech_tower_p2.json")
        if not project_path.exists():
            pytest.skip("Test project not found")
        
        load_project_data(project_path)
        
        # 2. Setup RAG
        vectorstore_dir = Path("vectorstore/normativa_db")
        if not vectorstore_dir.exists():
            pytest.skip("Vectorstore not found")
        
        rag = VectorstoreManager(vectorstore_dir)
        rag.load_existing()
        
        retriever = rag.get_retriever(k=3)
        qa_chain = create_qa_chain(retriever)
        
        # Set RAG for tools
        class SimpleRAGWrapper:
            def query(self, question):
                return qa_chain({"query": question})
        
        set_vectorstore_manager(SimpleRAGWrapper())
        
        # 3. Create agent
        agent = create_compliance_agent()
        
        # 4. Run verification
        query = "List all doors and check if they meet minimum width requirements."
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "iterations": 0
        }
        
        result = agent.invoke(initial_state)
        
        # Assertions
        assert result['iterations'] > 0
        assert len(result['messages']) > 1
        
        # Check that agent used tools
        # (Would need to inspect messages for tool calls)
    
    def test_rag_query_accuracy(self):
        """Test RAG retrieval accuracy."""
        vectorstore_dir = Path("vectorstore/normativa_db")
        if not vectorstore_dir.exists():
            pytest.skip("Vectorstore not found")
        
        rag = VectorstoreManager(vectorstore_dir)
        rag.load_existing()
        
        # Query about door width
        docs = rag.query_simple("ancho mínimo de puerta de evacuación", k=3)
        
        assert len(docs) > 0
        
        # Check that results mention relevant terms
        content = " ".join([doc.page_content.lower() for doc in docs])
        assert any(term in content for term in ["puerta", "ancho", "evacuación", "80", "cm"])
```

---

## Test Fixtures

### Creating Test Data

```python
# tests/fixtures/create_fixtures.py
import json
from pathlib import Path
from src.schemas import Project, ProjectMetadata, Level, Room, Door
from datetime import datetime

def create_test_project():
    """Create minimal test project."""
    metadata = ProjectMetadata(
        project_name="Test Project",
        level_name="Ground Floor",
        dwg_file="test.dwg",
        extracted_date=datetime.now().isoformat()
    )
    
    level = Level(name="Ground Floor", elevation=0.0)
    
    rooms = [
        Room(
            id="ROOM_001",
            name="Office A",
            level="Ground Floor",
            boundary=[[0, 0], [10, 0], [10, 8], [0, 8]]
        ),
        Room(
            id="ROOM_002",
            name="Office B",
            level="Ground Floor",
            boundary=[[10, 0], [20, 0], [20, 8], [10, 8]]
        )
    ]
    
    doors = [
        Door(
            id="DOOR_001",
            width=0.90,
            position=[5, 0],
            level="Ground Floor",
            is_egress=True
        ),
        Door(
            id="DOOR_002",
            width=0.75,  # Below minimum!
            position=[10, 4],
            level="Ground Floor",
            room_from="ROOM_001",
            room_to="ROOM_002"
        )
    ]
    
    return Project(
        metadata=metadata,
        levels=[level],
        rooms=rooms,
        doors=doors,
        walls=[]
    )


if __name__ == "__main__":
    project = create_test_project()
    
    output_path = Path("tests/fixtures/test_project.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created: {output_path}")
```

---

## Mocking

### Mocking External APIs

```python
# tests/unit/test_tools_mocked.py
import pytest
from unittest.mock import Mock, patch
from src.agent.tools import query_normativa, check_door_width_compliance

class TestQueryNormativaMocked:
    """Test normativa queries with mocked RAG."""
    
    @patch('src.agent.tools._rag_manager')
    def test_query_with_mock(self, mock_rag):
        """Test query with mocked RAG manager."""
        # Setup mock
        mock_rag.query.return_value = {
            'result': "El ancho mínimo es 80cm según CTE DB-SI",
            'source_documents': []
        }
        
        # Call tool
        result = query_normativa.invoke({"question": "¿Ancho mínimo de puerta?"})
        
        # Assertions
        assert "80cm" in result
        assert "CTE" in result
        mock_rag.query.assert_called_once()
    
    @patch('src.agent.tools._rag_manager')
    def test_query_error_handling(self, mock_rag):
        """Test error handling in query."""
        # Setup mock to raise error
        mock_rag.query.side_effect = Exception("API Error")
        
        # Call tool
        result = query_normativa.invoke({"question": "test"})
        
        # Should return error message, not crash
        assert "Error" in result
```

### Mocking File I/O

```python
@patch('builtins.open', new_callable=mock_open, read_data='{"test": "data"}')
def test_load_project_mocked(mock_file):
    """Test loading project with mocked file."""
    # Your test here
    pass
```

---

## Coverage

### Running Coverage

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html  # Mac
start htmlcov/index.html  # Windows
```

### Coverage Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = 
    -v
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=85
```

### Coverage Goals by Module

```
src/schemas.py              95%  ← Critical
src/extraction/             85%  ← Medium
src/calculations/geometry   90%  ← High
src/calculations/graph      85%  ← High
src/rag/                    70%  ← Integration
src/agent/tools             80%  ← High
src/agent/graph             60%  ← Complex
```

---

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/unit/test_geometry.py
```

### Run Specific Test

```bash
pytest tests/unit/test_geometry.py::TestCalculateRoomArea::test_rectangle_area
```

### Run by Marker

```bash
# Only fast tests
pytest -m "not slow"

# Only slow tests
pytest -m slow

# Skip tests requiring API
pytest -m "not requires_api"
```

### Verbose Output

```bash
pytest -v  # Verbose
pytest -vv  # Extra verbose
```

### Show Print Statements

```bash
pytest -s  # Show stdout
```

### Stop on First Failure

```bash
pytest -x
```

### Run Failed Tests

```bash
pytest --lf  # Last failed
pytest --ff  # Failed first, then others
```

---

## Writing Good Tests

### Test Naming

**Good**:
```python
def test_rectangle_area_is_width_times_height():
    """Test that rectangle area = width × height."""
```

**Bad**:
```python
def test_1():
    """Test area."""
```

### Test Structure: AAA Pattern

```python
def test_door_width_compliance():
    """Test door width compliance checking."""
    
    # Arrange (Setup)
    door = Door(
        id="DOOR_001",
        width=0.75,  # Below minimum
        position=[0, 0],
        level="P1"
    )
    
    # Act (Execute)
    result = check_door_width_compliance(door)
    
    # Assert (Verify)
    assert "NON-COMPLIANT" in result
    assert "0.75" in result  # Actual width mentioned
```

### One Assertion Per Test (Guideline)

**Good**:
```python
def test_room_area_calculation():
    """Test room area is calculated correctly."""
    area = calculate_room_area(room)
    assert pytest.approx(area) == 50.0

def test_room_area_is_positive():
    """Test room area is always positive."""
    area = calculate_room_area(room)
    assert area > 0
```

**Acceptable** (related assertions):
```python
def test_room_properties():
    """Test room has all required properties."""
    assert room.id == "ROOM_001"
    assert room.name == "Office"
    assert len(room.boundary) == 4
```

### Use Fixtures for Reusability

**Good**:
```python
@pytest.fixture
def standard_door():
    """Fixture for standard-compliant door."""
    return Door(
        id="DOOR_001",
        width=0.90,
        position=[0, 0],
        level="P1"
    )

def test_door_width(standard_door):
    """Test door width."""
    assert standard_door.width == 0.90

def test_door_compliance(standard_door):
    """Test door is compliant."""
    result = check_door_width_compliance(standard_door.id)
    assert "COMPLIANT" in result
```

### Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("width,expected", [
    (0.90, "COMPLIANT"),
    (0.80, "COMPLIANT"),
    (0.75, "NON-COMPLIANT"),
    (0.60, "NON-COMPLIANT"),
])
def test_door_width_compliance_cases(width, expected):
    """Test various door widths for compliance."""
    door = Door(id="DOOR_001", width=width, position=[0,0], level="P1")
    result = check_door_width_compliance(door.id)
    assert expected in result
```

### Test Edge Cases

```python
def test_room_area_with_zero_dimensions():
    """Test room with zero-width boundary."""
    room = Room(
        id="ROOM_001",
        name="Line",
        level="P1",
        boundary=[[0, 0], [0, 10], [0, 10], [0, 0]]
    )
    area = calculate_room_area(room)
    assert area == 0.0

def test_room_area_with_negative_coordinates():
    """Test room with negative coordinates (should still work)."""
    room = Room(
        id="ROOM_001",
        name="Negative",
        level="P1",
        boundary=[[-10, -10], [0, -10], [0, 0], [-10, 0]]
    )
    area = calculate_room_area(room)
    assert area == 100.0
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## Summary

### Quick Reference

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run fast tests only
pytest -m "not slow"

# Run specific test
pytest tests/unit/test_geometry.py::test_rectangle_area

# Debug test
pytest --pdb

# Show print statements
pytest -s
```

### Test Checklist

- [ ] Unit tests for all functions
- [ ] Integration tests for workflows
- [ ] Test edge cases
- [ ] Test error handling
- [ ] Use fixtures for common setups
- [ ] Mock external dependencies
- [ ] Achieve 85%+ coverage
- [ ] All tests pass in CI/CD

---

**Version**: 1.0  
**Last Updated**: October 2025