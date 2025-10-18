# 📚 API Reference - AEC Compliance Agent

## 📋 Table of Contents

1. [Overview](#overview)
2. [Data Schemas](#data-schemas)
3. [Extraction Module](#extraction-module)
4. [Calculations Module](#calculations-module)
5. [RAG Module](#rag-module)
6. [Agent Module](#agent-module)
7. [Utilities](#utilities)

---

## Overview

This document provides complete API documentation for all modules in the AEC Compliance Agent.

### Import Conventions

```python
# Schemas
from src.schemas import Project, Room, Door, Wall, Level, ProjectMetadata

# Extraction
from src.extraction.dxf_export import extract_from_dxf, save_to_json
from src.extraction.json_validator import validate_json

# Calculations
from src.calculations.geometry import (
    get_room_polygon, calculate_room_area, get_room_centroid,
    calculate_perimeter, point_in_room, distance_between_points
)
from src.calculations.graph import CirculationGraph

# RAG
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain
from src.rag.embeddings_config import get_embeddings
from src.rag.document_loader import load_pdfs

# Agent
from src.agent.tools import TOOLS, load_project_data, set_vectorstore_manager
from src.agent.graph import create_compliance_agent
from src.agent.state import AgentState

# Utilities
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.visualization import plot_floorplan
```

---

## Data Schemas

### ProjectMetadata

Project-level metadata.

```python
class ProjectMetadata(BaseModel):
    """Project metadata."""
    
    project_name: str
    level_name: str
    dwg_file: str
    extracted_date: str  # ISO format
```

**Example**:
```python
metadata = ProjectMetadata(
    project_name="Tech Tower",
    level_name="Planta 2",
    dwg_file="tech_tower_p2.dwg",
    extracted_date="2025-10-18T10:30:00"
)
```

---

### Level

Building level/floor.

```python
class Level(BaseModel):
    """Building level."""
    
    name: str
    elevation: float = 0.0  # meters above ground
```

**Example**:
```python
level = Level(name="Planta 2", elevation=3.5)
```

---

### Room

Room with boundary polygon.

```python
class Room(BaseModel):
    """Room entity."""
    
    id: str
    name: str
    level: str
    boundary: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    use_type: Optional[str] = None
```

**Fields**:
- `id`: Unique identifier (e.g., "ROOM_001")
- `name`: Human-readable name
- `level`: Name of the level this room is on
- `boundary`: List of [x, y] coordinates forming closed polygon
- `use_type`: Optional room classification (e.g., "Office", "Corridor")

**Example**:
```python
room = Room(
    id="ROOM_001",
    name="Oficina Principal",
    level="Planta 2",
    boundary=[
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 8.0],
        [0.0, 8.0],
        [0.0, 0.0]
    ],
    use_type="Office"
)
```

**Validation**:
- `boundary` must have at least 3 points
- First and last points should be the same (closed polygon)

---

### Door

Door entity.

```python
class Door(BaseModel):
    """Door entity."""
    
    id: str
    width: float  # meters
    height: float = 2.10  # meters
    position: List[float]  # [x, y]
    level: str
    room_from: Optional[str] = None
    room_to: Optional[str] = None
    is_fire_rated: bool = False
    is_egress: bool = False
```

**Fields**:
- `id`: Unique identifier
- `width`: Door width in meters
- `height`: Door height in meters (default 2.10m)
- `position`: [x, y] coordinates
- `level`: Level name
- `room_from`: Room ID on one side
- `room_to`: Room ID on other side
- `is_fire_rated`: Fire-rated door flag
- `is_egress`: Egress/exit door flag

**Example**:
```python
door = Door(
    id="DOOR_001",
    width=0.90,
    height=2.10,
    position=[5.0, 0.0],
    level="Planta 2",
    room_from="ROOM_001",
    room_to="CORRIDOR_01",
    is_fire_rated=False,
    is_egress=True
)
```

**Constraints**:
- `width` must be positive
- `height` must be positive

---

### Wall

Wall entity.

```python
class Wall(BaseModel):
    """Wall entity."""
    
    id: str
    start: List[float]  # [x, y]
    end: List[float]  # [x, y]
    length: float  # meters
    level: str
    fire_rating: Optional[int] = None  # minutes (60, 90, 120, etc.)
```

**Fields**:
- `id`: Unique identifier
- `start`: [x, y] start coordinates
- `end`: [x, y] end coordinates
- `length`: Wall length in meters
- `level`: Level name
- `fire_rating`: Fire resistance in minutes

**Example**:
```python
wall = Wall(
    id="WALL_001",
    start=[0.0, 0.0],
    end=[10.0, 0.0],
    length=10.0,
    level="Planta 2",
    fire_rating=90
)
```

---

### Project

Complete project data.

```python
class Project(BaseModel):
    """Complete project."""
    
    metadata: ProjectMetadata
    levels: List[Level] = []
    rooms: List[Room] = []
    doors: List[Door] = []
    walls: List[Wall] = []
```

**Example**:
```python
project = Project(
    metadata=metadata,
    levels=[level],
    rooms=[room1, room2],
    doors=[door1, door2],
    walls=[wall1, wall2, wall3]
)

# Access data
print(f"Project: {project.metadata.project_name}")
print(f"Rooms: {len(project.rooms)}")
print(f"Doors: {len(project.doors)}")

# Serialize to dict
data = project.model_dump()

# Serialize to JSON
json_str = project.model_dump_json(indent=2)
```

---

## Extraction Module

### extract_from_dxf()

Extract structured data from DXF file.

```python
def extract_from_dxf(
    dxf_path: Path,
    level_name: str = "Planta 2"
) -> Project
```

**Parameters**:
- `dxf_path`: Path to DXF file
- `level_name`: Name of the level being extracted

**Returns**:
- `Project`: Extracted project data

**Raises**:
- `FileNotFoundError`: DXF file not found
- `DXFStructureError`: Invalid DXF file

**Example**:
```python
from pathlib import Path
from src.extraction.dxf_export import extract_from_dxf

dxf_path = Path("data/raw/dwg/building_p2.dxf")
project = extract_from_dxf(dxf_path, level_name="Planta 2")

print(f"Extracted {len(project.rooms)} rooms")
print(f"Extracted {len(project.doors)} doors")
print(f"Extracted {len(project.walls)} walls")
```

**Details**:
- Parses INSERT blocks for doors
- Parses LINE entities for walls
- Parses LWPOLYLINE for room boundaries
- Parses TEXT/MTEXT for labels
- Automatically calculates dimensions

---

### save_to_json()

Save project to JSON file.

```python
def save_to_json(
    project: Project,
    output_path: Path
) -> None
```

**Parameters**:
- `project`: Project object to save
- `output_path`: Path to output JSON file

**Example**:
```python
from src.extraction.dxf_export import save_to_json

output_path = Path("data/extracted/project.json")
save_to_json(project, output_path)
```

---

### validate_json()

Validate JSON against Project schema.

```python
def validate_json(json_path: Path) -> bool
```

**Parameters**:
- `json_path`: Path to JSON file

**Returns**:
- `bool`: True if valid, False otherwise

**Example**:
```python
from src.extraction.json_validator import validate_json

json_path = Path("data/extracted/project.json")
is_valid = validate_json(json_path)

if is_valid:
    print("✅ JSON is valid")
else:
    print("❌ JSON validation failed")
```

---

## Calculations Module

### Geometry Functions

#### get_room_polygon()

Convert Room to Shapely Polygon.

```python
def get_room_polygon(room: Room) -> Polygon
```

**Parameters**:
- `room`: Room object

**Returns**:
- `Polygon`: Shapely polygon

**Example**:
```python
from src.calculations.geometry import get_room_polygon

polygon = get_room_polygon(room)
print(f"Is valid: {polygon.is_valid}")
print(f"Type: {polygon.geom_type}")
```

---

#### calculate_room_area()

Calculate room area.

```python
def calculate_room_area(room: Room) -> float
```

**Parameters**:
- `room`: Room object

**Returns**:
- `float`: Area in square meters

**Example**:
```python
from src.calculations.geometry import calculate_room_area

area = calculate_room_area(room)
print(f"Room area: {area:.2f} m²")
```

**Precision**: ±0.01 m²

---

#### get_room_centroid()

Get room centroid coordinates.

```python
def get_room_centroid(room: Room) -> Tuple[float, float]
```

**Parameters**:
- `room`: Room object

**Returns**:
- `Tuple[float, float]`: (x, y) coordinates of centroid

**Example**:
```python
from src.calculations.geometry import get_room_centroid

centroid = get_room_centroid(room)
print(f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
```

---

#### calculate_perimeter()

Calculate room perimeter.

```python
def calculate_perimeter(room: Room) -> float
```

**Parameters**:
- `room`: Room object

**Returns**:
- `float`: Perimeter in meters

**Example**:
```python
from src.calculations.geometry import calculate_perimeter

perimeter = calculate_perimeter(room)
print(f"Perimeter: {perimeter:.2f} m")
```

---

#### point_in_room()

Check if point is inside room.

```python
def point_in_room(
    point: Tuple[float, float],
    room: Room
) -> bool
```

**Parameters**:
- `point`: (x, y) coordinates
- `room`: Room object

**Returns**:
- `bool`: True if point is inside room

**Example**:
```python
from src.calculations.geometry import point_in_room

is_inside = point_in_room((5.0, 2.5), room)
if is_inside:
    print("Point is inside room")
```

**Note**: Points exactly on boundary return False

---

#### distance_between_points()

Calculate Euclidean distance between two points.

```python
def distance_between_points(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float
```

**Parameters**:
- `p1`: First point (x, y)
- `p2`: Second point (x, y)

**Returns**:
- `float`: Distance in meters

**Example**:
```python
from src.calculations.geometry import distance_between_points

distance = distance_between_points((0, 0), (3, 4))
print(f"Distance: {distance:.2f} m")  # 5.0
```

---

### CirculationGraph

Graph-based circulation analysis.

```python
class CirculationGraph:
    """
    Represents building circulation as a graph.
    
    Nodes = Rooms
    Edges = Doors connecting rooms
    """
    
    def __init__(self, project: Project):
        """Create circulation graph from project."""
```

**Example**:
```python
from src.calculations.graph import CirculationGraph

graph = CirculationGraph(project)
print(f"Nodes: {len(graph.graph.nodes)}")
print(f"Edges: {len(graph.graph.edges)}")
```

---

#### shortest_path()

Find shortest path between two rooms.

```python
def shortest_path(
    self,
    from_room: str,
    to_room: str
) -> Optional[List[str]]
```

**Parameters**:
- `from_room`: Starting room ID
- `to_room`: Destination room ID

**Returns**:
- `List[str]`: List of room IDs in path, or None if no path exists

**Example**:
```python
path = graph.shortest_path("ROOM_001", "ROOM_005")

if path:
    print("Path:", " → ".join(path))
else:
    print("No path found")
```

---

#### shortest_egress_path()

Find shortest evacuation path to any egress door.

```python
def shortest_egress_path(
    self,
    from_room: str
) -> Optional[Dict]
```

**Parameters**:
- `from_room`: Starting room ID

**Returns**:
- `Dict` with keys:
  - `path`: List of room IDs
  - `distance`: Distance in meters
- `None` if no egress path exists

**Example**:
```python
result = graph.shortest_egress_path("ROOM_001")

if result:
    print(f"Path: {' → '.join(result['path'])}")
    print(f"Distance: {result['distance']:.2f} m")
else:
    print("No egress path found")
```

---

#### get_path_visualization_data()

Get data for visualizing a path.

```python
def get_path_visualization_data(
    self,
    path: List[str]
) -> Dict
```

**Parameters**:
- `path`: List of room IDs

**Returns**:
- `Dict` with keys:
  - `positions`: List of (x, y) tuples
  - `names`: List of room names

**Example**:
```python
path = graph.shortest_path("ROOM_001", "ROOM_005")
viz_data = graph.get_path_visualization_data(path)

print("Positions:", viz_data['positions'])
print("Names:", viz_data['names'])
```

---

## RAG Module

### VectorstoreManager

Manages RAG vectorstore creation and loading.

```python
class VectorstoreManager:
    """Manages vectorstore for RAG."""
    
    def __init__(self, persist_directory: Path):
        """Initialize manager."""
```

**Example**:
```python
from pathlib import Path
from src.rag.vectorstore_manager import VectorstoreManager

vectorstore_dir = Path("vectorstore/normativa_db")
rag = VectorstoreManager(vectorstore_dir)
```

---

#### create_from_pdfs()

Create vectorstore from PDF documents.

```python
def create_from_pdfs(
    self,
    pdf_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> None
```

**Parameters**:
- `pdf_dir`: Directory containing PDF files
- `chunk_size`: Size of text chunks (characters)
- `chunk_overlap`: Overlap between chunks (characters)

**Example**:
```python
pdf_dir = Path("data/normativa")
rag.create_from_pdfs(
    pdf_dir,
    chunk_size=1000,
    chunk_overlap=200
)
```

**Time**: ~2-5 minutes for 3 PDFs (~200 pages)

---

#### load_existing()

Load existing vectorstore.

```python
def load_existing(self) -> None
```

**Example**:
```python
rag.load_existing()
print("✅ Vectorstore loaded")
```

**Raises**:
- `ValueError`: If vectorstore doesn't exist

---

#### get_retriever()

Get retriever for similarity searches.

```python
def get_retriever(self, k: int = 3)
```

**Parameters**:
- `k`: Number of documents to retrieve

**Returns**:
- Retriever object

**Example**:
```python
retriever = rag.get_retriever(k=3)
docs = retriever.get_relevant_documents("ancho mínimo de puerta")

for doc in docs:
    print(doc.page_content[:200])
```

---

#### query_simple()

Simple query without LLM (retrieval only).

```python
def query_simple(
    self,
    question: str,
    k: int = 3
) -> List[Document]
```

**Parameters**:
- `question`: Query string
- `k`: Number of results

**Returns**:
- List of Document objects

**Example**:
```python
docs = rag.query_simple("¿Ancho mínimo de puerta?", k=3)

for doc in docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}")
    print("---")
```

---

### create_qa_chain()

Create QA chain with retrieval.

```python
def create_qa_chain(
    retriever,
    model_name: str = "gemini-pro",
    temperature: float = 0.1
) -> RetrievalQA
```

**Parameters**:
- `retriever`: Vectorstore retriever
- `model_name`: Google model name
- `temperature`: LLM temperature (0-1)

**Returns**:
- Configured QA chain

**Example**:
```python
from src.rag.qa_chain import create_qa_chain

retriever = rag.get_retriever(k=3)
qa_chain = create_qa_chain(
    retriever,
    model_name="gemini-pro",
    temperature=0.1
)

# Query
result = qa_chain({"query": "¿Ancho mínimo de puerta de evacuación?"})
print(result['result'])
```

---

### get_embeddings()

Get embeddings model.

```python
def get_embeddings() -> HuggingFaceEmbeddings
```

**Returns**:
- Configured embeddings instance

**Example**:
```python
from src.rag.embeddings_config import get_embeddings

embeddings = get_embeddings()

# Embed a query
vector = embeddings.embed_query("ancho de puerta")
print(f"Vector dimensions: {len(vector)}")  # 384
```

**Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- Dimensions: 384
- Languages: 50+ including Spanish

---

### load_pdfs()

Load PDF documents.

```python
def load_pdfs(pdf_dir: Path) -> List[Document]
```

**Parameters**:
- `pdf_dir`: Directory with PDF files

**Returns**:
- List of Document objects

**Example**:
```python
from src.rag.document_loader import load_pdfs

docs = load_pdfs(Path("data/normativa"))
print(f"Loaded {len(docs)} pages")
```

---

## Agent Module

### Agent Tools

#### load_project_data()

Load project data into memory for agent.

```python
def load_project_data(json_path: Path) -> None
```

**Parameters**:
- `json_path`: Path to project JSON

**Example**:
```python
from src.agent.tools import load_project_data

load_project_data(Path("data/extracted/project.json"))
```

**Note**: Must be called before using agent tools

---

#### set_vectorstore_manager()

Set RAG manager for normativa queries.

```python
def set_vectorstore_manager(rag) -> None
```

**Parameters**:
- `rag`: RAG manager instance or wrapper

**Example**:
```python
from src.agent.tools import set_vectorstore_manager

set_vectorstore_manager(rag)
```

---

#### @tool get_room_info()

Get information about a room.

```python
@tool
def get_room_info(room_id: str) -> str
```

**Parameters**:
- `room_id`: Room ID (e.g., "ROOM_001")

**Returns**:
- String with room information

**Example**:
```python
from src.agent.tools import get_room_info

info = get_room_info.invoke({"room_id": "ROOM_001"})
print(info)
```

**Output**:
```
Room: Oficina A (ROOM_001)
Level: Planta 2
Area: 50.00 m²
Use: Office
Boundary: 5 points
```

---

#### @tool get_door_info()

Get information about a door.

```python
@tool
def get_door_info(door_id: str) -> str
```

**Parameters**:
- `door_id`: Door ID (e.g., "DOOR_001")

**Returns**:
- String with door information

**Example**:
```python
from src.agent.tools import get_door_info

info = get_door_info.invoke({"door_id": "DOOR_001"})
print(info)
```

**Output**:
```
Door: DOOR_001
Width: 0.90 m (90 cm)
Height: 2.10 m
Position: [5.0, 0.0]
Fire-rated: No
Egress door: Yes
```

---

#### @tool list_all_doors()

List all doors in project.

```python
@tool
def list_all_doors() -> str
```

**Returns**:
- String with list of all doors

**Example**:
```python
from src.agent.tools import list_all_doors

doors = list_all_doors.invoke({})
print(doors)
```

**Output**:
```
Total doors: 3

- DOOR_001: 90cm wide [EGRESS]
- DOOR_002: 80cm wide
- DOOR_003: 75cm wide
```

---

#### @tool check_door_width_compliance()

Check if door meets width requirements.

```python
@tool
def check_door_width_compliance(door_id: str) -> str
```

**Parameters**:
- `door_id`: Door ID to check

**Returns**:
- Compliance status string

**Example**:
```python
from src.agent.tools import check_door_width_compliance

result = check_door_width_compliance.invoke({"door_id": "DOOR_001"})
print(result)
```

**Output**:
```
✅ COMPLIANT: Door DOOR_001 width (90cm) meets minimum requirement (80cm)
```

---

#### @tool query_normativa()

Query building codes (CTE).

```python
@tool
def query_normativa(question: str) -> str
```

**Parameters**:
- `question`: Question about building codes (Spanish)

**Returns**:
- Answer with citations

**Example**:
```python
from src.agent.tools import query_normativa

answer = query_normativa.invoke({
    "question": "¿Ancho mínimo de puerta de evacuación?"
})
print(answer)
```

**Output**:
```
Según el CTE DB-SI, el ancho mínimo de las puertas de evacuación
debe ser de 80 cm en edificios de uso residencial.

Sources:
1. CTE_DB-SI.pdf, page 23
2. CTE_DB-SI.pdf, page 45
```

---

#### @tool calculate_egress_distance()

Calculate evacuation distance from a room.

```python
@tool
def calculate_egress_distance(room_id: str) -> str
```

**Parameters**:
- `room_id`: Starting room ID

**Returns**:
- String with path and distance

**Example**:
```python
from src.agent.tools import calculate_egress_distance

result = calculate_egress_distance.invoke({"room_id": "ROOM_001"})
print(result)
```

**Output**:
```
Egress path from ROOM_001:
Path: Oficina A → Pasillo → Salida
Distance: 15.50 meters
Number of rooms traversed: 3
```

---

### create_compliance_agent()

Create ReAct agent for compliance verification.

```python
def create_compliance_agent()
```

**Returns**:
- Compiled LangGraph agent

**Example**:
```python
from src.agent.graph import create_compliance_agent
from langchain_core.messages import HumanMessage

# Create agent
agent = create_compliance_agent()

# Run agent
initial_state = {
    "messages": [
        HumanMessage(content="Verify all doors meet minimum width")
    ],
    "iterations": 0
}

result = agent.invoke(initial_state)

# Print response
for msg in result["messages"]:
    if hasattr(msg, 'content') and msg.content:
        print(msg.content)
```

---

### AgentState

State structure for agent execution.

```python
class AgentState(TypedDict):
    """Agent state during execution."""
    
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
```

**Fields**:
- `messages`: Message history (accumulates)
- `iterations`: Iteration counter

---

## Utilities

### load_config()

Load configuration from .env file.

```python
def load_config() -> Dict[str, str]
```

**Returns**:
- Dictionary with configuration

**Example**:
```python
from src.utils.config import load_config

config = load_config()
api_key = config['GOOGLE_API_KEY']
```

---

### get_logger()

Get configured logger.

```python
def get_logger(name: str) -> logging.Logger
```

**Parameters**:
- `name`: Logger name (usually `__name__`)

**Returns**:
- Configured logger

**Example**:
```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.warning("Missing data")
logger.error("Failed to load file")
```

---

### plot_floorplan()

Visualize floor plan with optional path highlighting.

```python
def plot_floorplan(
    project: Project,
    highlight_path: Optional[List[str]] = None,
    output_path: Optional[Path] = None
) -> None
```

**Parameters**:
- `project`: Project data
- `highlight_path`: List of room IDs to highlight
- `output_path`: Path to save image (optional)

**Example**:
```python
from src.utils.visualization import plot_floorplan
from src.calculations.graph import CirculationGraph

# Basic plot
plot_floorplan(project)

# With highlighted path
graph = CirculationGraph(project)
path = graph.shortest_path("ROOM_001", "ROOM_005")
plot_floorplan(project, highlight_path=path, output_path=Path("output.png"))
```

---

## Complete Usage Example

```python
from pathlib import Path
from src.extraction.dxf_export import extract_from_dxf, save_to_json
from src.calculations.geometry import calculate_room_area
from src.calculations.graph import CirculationGraph
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain
from src.agent.tools import load_project_data, set_vectorstore_manager
from src.agent.graph import create_compliance_agent
from langchain_core.messages import HumanMessage

# 1. Extract data from DXF
dxf_path = Path("data/raw/dwg/building.dxf")
project = extract_from_dxf(dxf_path, level_name="Planta 2")
save_to_json(project, Path("data/extracted/building.json"))

# 2. Perform calculations
for room in project.rooms:
    area = calculate_room_area(room)
    print(f"{room.name}: {area:.2f} m²")

graph = CirculationGraph(project)
egress = graph.shortest_egress_path("ROOM_001")
print(f"Egress distance: {egress['distance']:.2f} m")

# 3. Setup RAG
rag = VectorstoreManager(Path("vectorstore/normativa_db"))
if not Path("vectorstore/normativa_db").exists():
    rag.create_from_pdfs(Path("data/normativa"))
else:
    rag.load_existing()

retriever = rag.get_retriever(k=3)
qa_chain = create_qa_chain(retriever)

# 4. Create and run agent
load_project_data(Path("data/extracted/building.json"))

class RAGWrapper:
    def query(self, question):
        return qa_chain({"query": question})

set_vectorstore_manager(RAGWrapper())

agent = create_compliance_agent()

result = agent.invoke({
    "messages": [HumanMessage(content="Verify all doors meet requirements")],
    "iterations": 0
})

for msg in result["messages"]:
    if hasattr(msg, 'content') and msg.content:
        print(msg.content)
```

---

## Type Hints

### Common Types

```python
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from src.schemas import Project, Room, Door
from shapely.geometry import Polygon
from langchain.schema import Document
```

### Function Signatures

```python
# Extraction
def extract_from_dxf(dxf_path: Path, level_name: str = "Planta 2") -> Project: ...
def save_to_json(project: Project, output_path: Path) -> None: ...

# Geometry
def get_room_polygon(room: Room) -> Polygon: ...
def calculate_room_area(room: Room) -> float: ...
def get_room_centroid(room: Room) -> Tuple[float, float]: ...

# RAG
def load_pdfs(pdf_dir: Path) -> List[Document]: ...
def get_embeddings() -> HuggingFaceEmbeddings: ...

# Agent
def load_project_data(json_path: Path) -> None: ...
def create_compliance_agent() -> CompiledGraph: ...
```

---

## Error Handling

### Common Exceptions

```python
# File operations
try:
    project = extract_from_dxf(dxf_path)
except FileNotFoundError:
    print("DXF file not found")
except DXFStructureError:
    print("Invalid DXF file")

# Validation
try:
    project = Project(**data)
except ValidationError as e:
    print(f"Validation error: {e}")

# RAG operations
try:
    rag.load_existing()
except ValueError:
    print("Vectorstore not found")
    rag.create_from_pdfs(pdf_dir)

# Graph operations
path = graph.shortest_path("ROOM_A", "ROOM_B")
if path is None:
    print("No path found between rooms")
```

---

## Performance Considerations

### Vectorstore Creation

- **Time**: ~2-5 minutes for 200 pages
- **Memory**: ~2GB RAM during creation
- **Disk**: ~50MB for vectorstore

### RAG Query

- **Time**: ~2-3 seconds per query
- **Factors**: Number of retrieved chunks (k), LLM speed

### Agent Execution

- **Time**: ~10-30 seconds depending on complexity
- **Factors**: Number of tool calls, RAG queries

### Optimization Tips

```python
# Use smaller chunk size for faster indexing
rag.create_from_pdfs(pdf_dir, chunk_size=500)

# Reduce retrieval count for faster queries
retriever = rag.get_retriever(k=2)  # Instead of k=5

# Use GPU for embeddings if available
embeddings = HuggingFaceEmbeddings(
    model_kwargs={'device': 'cuda'}
)
```

---

**Version**: 1.0  
**Last Updated**: October 2025