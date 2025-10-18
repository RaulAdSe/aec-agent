# 🛠️ Development Guide - AEC Compliance Agent

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Development Workflow](#development-workflow)
5. [Pilar 1: Data Extraction](#pilar-1-data-extraction)
6. [Pilar 2: Calculations](#pilar-2-calculations)
7. [Pilar 3: RAG System](#pilar-3-rag-system)
8. [Pilar 4: Agent](#pilar-4-agent)
9. [Integration](#integration)
10. [Common Issues](#common-issues)

---

## Overview

### Project Goals

Build a **Proof of Concept (POC)** for an AI agent that:
1. Extracts data from architectural drawings (DWG/DXF)
2. Performs geometric calculations and analysis
3. Queries building code documentation using RAG
4. Autonomously verifies building code compliance using ReAct framework

### Timeline

**15 days** (October 15-29, 2025)

### Target Presentation

**October 29, 2025** at UPC-EPSEB for "Intel·ligència Artificial Aplicada a la Construcció"

---

## Project Structure

```
aec-compliance-agent/
├── README.md                      # Quick start guide
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
│
├── data/                          # Data directory
│   ├── raw/                       # Raw input files
│   │   ├── rvt/                   # Revit files (if any)
│   │   └── dwg/                   # DWG files (AutoCAD)
│   ├── extracted/                 # Extracted JSON data
│   │   └── tech_tower_p2.json     # Example project
│   └── normativa/                 # Building codes (PDFs)
│       ├── CTE_DB-SI.pdf
│       ├── CTE_DB-SUA.pdf
│       └── README.md
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── schemas.py                 # Pydantic data models
│   │
│   ├── extraction/                # Pilar 1: Data extraction
│   │   ├── __init__.py
│   │   ├── dxf_export.py          # DXF extraction
│   │   └── json_validator.py     # Schema validation
│   │
│   ├── calculations/              # Pilar 2: Calculations
│   │   ├── __init__.py
│   │   ├── geometry.py            # Shapely operations
│   │   └── graph.py               # NetworkX circulation
│   │
│   ├── rag/                       # Pilar 3: RAG system
│   │   ├── __init__.py
│   │   ├── document_loader.py     # PDF loading
│   │   ├── embeddings_config.py   # Embeddings setup
│   │   ├── vectorstore_manager.py # ChromaDB management
│   │   └── qa_chain.py            # QA chain
│   │
│   ├── agent/                     # Pilar 4: ReAct agent
│   │   ├── __init__.py
│   │   ├── state.py               # Agent state
│   │   ├── prompts.py             # System prompts
│   │   ├── tools.py               # Agent tools
│   │   └── graph.py               # LangGraph orchestration
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── config.py              # Configuration
│       ├── logger.py              # Logging setup
│       └── visualization.py       # Plotting functions
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── unit/                      # Unit tests
│   │   ├── test_schemas.py
│   │   ├── test_extraction.py
│   │   ├── test_geometry.py
│   │   ├── test_graph.py
│   │   └── test_tools.py
│   ├── integration/               # Integration tests
│   │   ├── test_rag_pipeline.py
│   │   ├── test_agent_simple.py
│   │   └── test_e2e_pipeline.py
│   └── fixtures/                  # Test data
│       ├── sample.dxf
│       └── test_project.json
│
├── notebooks/                     # Tutorial notebooks
│   ├── 00_complete_demo.ipynb     # Full demo
│   ├── 01_extraction_tutorial.ipynb
│   ├── 02_calculations_tutorial.ipynb
│   ├── 03_rag_tutorial.ipynb      # ⭐ Key for presentation
│   └── 04_agent_tutorial.ipynb    # ⭐ Key for presentation
│
├── scripts/                       # Utility scripts
│   ├── create_example_data.py     # Generate test data
│   ├── create_vectorstore.py      # Build RAG vectorstore
│   └── run_full_pipeline.py       # Execute complete workflow
│
├── outputs/                       # Generated outputs
│   ├── reports/                   # Compliance reports
│   ├── logs/                      # Execution logs
│   └── visualizations/            # Floor plan images
│
├── vectorstore/                   # RAG vector database
│   └── normativa_db/              # ChromaDB storage
│
├── docs/                          # Documentation
│   ├── TECH_STACK.md
│   ├── PROJECT_STRUCTURE.md
│   ├── DEVELOPMENT_GUIDE.md       # ← This file
│   ├── RAG_EXPLAINED.md
│   ├── REACT_EXPLAINED.md
│   ├── TESTING_GUIDE.md
│   ├── API_REFERENCE.md
│   ├── GIT_WORKFLOW.md
│   └── IMPLEMENTATION_ROADMAP.md
│
└── .github/                       # GitHub Actions
    └── workflows/
        ├── tests.yml              # CI/CD tests
        └── docs.yml               # Documentation checks
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or 3.11
- Git
- Google API key (for Gemini)
- 4GB+ RAM (for embeddings)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/aec-compliance-agent.git
cd aec-compliance-agent

# 2. Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Verify installation
python -c "import langchain; import ezdxf; import shapely; print('✅ All imports successful')"
```

### Environment Variables

```bash
# .env
GOOGLE_API_KEY=your-api-key-here
PROJECT_ROOT=/path/to/aec-compliance-agent
VECTORSTORE_DIR=./vectorstore
DATA_DIR=./data
```

---

## Development Workflow

### Git Workflow

Follow the feature branch workflow:

```bash
# Create feature branch
git checkout develop
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add: feature description"

# Push and create PR
git push origin feature/my-feature
# Create PR on GitHub: feature/my-feature → develop
```

See [GIT_WORKFLOW.md](GIT_WORKFLOW.md) for detailed instructions.

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_schemas.py

# Run with coverage
pytest --cov=src

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Code Style

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking (optional)
mypy src/
```

---

## Pilar 1: Data Extraction

### Goal

Extract structured data from DWG/DXF files into JSON format.

### Data Model

```python
# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ProjectMetadata(BaseModel):
    """Project-level metadata"""
    project_name: str
    level_name: str  # e.g., "Planta 2"
    dwg_file: str
    extracted_date: str

class Level(BaseModel):
    """Building level/floor"""
    name: str
    elevation: float = 0.0

class Room(BaseModel):
    """Room with boundaries"""
    id: str
    name: str
    level: str
    boundary: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    use_type: Optional[str] = None

class Door(BaseModel):
    """Door entity"""
    id: str
    width: float  # meters
    height: float = 2.10
    position: List[float]  # [x, y]
    level: str
    room_from: Optional[str] = None
    room_to: Optional[str] = None
    is_fire_rated: bool = False
    is_egress: bool = False

class Wall(BaseModel):
    """Wall entity"""
    id: str
    start: List[float]  # [x, y]
    end: List[float]  # [x, y]
    length: float
    level: str
    fire_rating: Optional[int] = None  # minutes (e.g., 60, 90, 120)

class Project(BaseModel):
    """Complete project data"""
    metadata: ProjectMetadata
    levels: List[Level] = []
    rooms: List[Room] = []
    doors: List[Door] = []
    walls: List[Wall] = []
```

### DXF Extraction

```python
# src/extraction/dxf_export.py
import ezdxf
import json
from pathlib import Path
from typing import Dict, Any, List
from src.schemas import Project, ProjectMetadata, Door, Room, Wall, Level

def extract_from_dxf(dxf_path: Path, level_name: str = "Planta 2") -> Project:
    """
    Extract data from DXF file.
    
    Args:
        dxf_path: Path to DXF file
        level_name: Name of the level being extracted
    
    Returns:
        Project object with extracted data
    """
    # Load DXF
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    
    # Initialize collections
    doors = []
    rooms = []
    walls = []
    
    # Extract doors from INSERT blocks
    door_id = 0
    for entity in msp.query('INSERT'):
        if 'DOOR' in entity.dxf.name.upper() or 'PUERTA' in entity.dxf.name.upper():
            # Get position
            position = [entity.dxf.insert.x, entity.dxf.insert.y]
            
            # Try to get width from attributes or use default
            width = 0.80  # Default 80cm
            if entity.has_attrib('WIDTH') or entity.has_attrib('ANCHO'):
                width_str = entity.get_attrib_text('WIDTH') or entity.get_attrib_text('ANCHO')
                try:
                    width = float(width_str)
                except:
                    pass
            
            doors.append(Door(
                id=f"DOOR_{door_id:03d}",
                width=width,
                position=position,
                level=level_name
            ))
            door_id += 1
    
    # Extract walls from LINE entities
    wall_id = 0
    for entity in msp.query('LINE'):
        start = [entity.dxf.start.x, entity.dxf.start.y]
        end = [entity.dxf.end.x, entity.dxf.end.y]
        
        # Calculate length
        length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
        
        # Only consider walls longer than 10cm
        if length > 0.1:
            walls.append(Wall(
                id=f"WALL_{wall_id:03d}",
                start=start,
                end=end,
                length=length,
                level=level_name
            ))
            wall_id += 1
    
    # Extract rooms from closed POLYLINE/LWPOLYLINE
    room_id = 0
    for entity in msp.query('LWPOLYLINE'):
        if entity.is_closed:
            # Get boundary points
            boundary = [[point[0], point[1]] for point in entity.get_points()]
            
            # Try to get room name from nearby TEXT
            room_name = f"ROOM_{room_id:03d}"
            # (In real implementation, find TEXT near centroid)
            
            rooms.append(Room(
                id=room_name,
                name=room_name,
                level=level_name,
                boundary=boundary
            ))
            room_id += 1
    
    # Create metadata
    metadata = ProjectMetadata(
        project_name=dxf_path.stem,
        level_name=level_name,
        dwg_file=str(dxf_path.name),
        extracted_date=datetime.now().isoformat()
    )
    
    # Create level
    level = Level(name=level_name, elevation=0.0)
    
    # Create project
    project = Project(
        metadata=metadata,
        levels=[level],
        rooms=rooms,
        doors=doors,
        walls=walls
    )
    
    return project

def save_to_json(project: Project, output_path: Path):
    """Save project to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"✅ Saved to: {output_path}")

# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract data from DXF")
    parser.add_argument("dxf_file", type=Path, help="Path to DXF file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path")
    parser.add_argument("-l", "--level", default="Planta 2", help="Level name")
    
    args = parser.parse_args()
    
    # Extract
    project = extract_from_dxf(args.dxf_file, args.level)
    
    # Save
    output = args.output or Path(f"data/extracted/{args.dxf_file.stem}.json")
    save_to_json(project, output)
```

### JSON Validation

```python
# src/extraction/json_validator.py
from pathlib import Path
from src.schemas import Project
import json

def validate_json(json_path: Path) -> bool:
    """
    Validate JSON against Project schema.
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate with Pydantic
        project = Project(**data)
        
        print("✅ JSON is valid!")
        print(f"  - Rooms: {len(project.rooms)}")
        print(f"  - Doors: {len(project.doors)}")
        print(f"  - Walls: {len(project.walls)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate extracted JSON")
    parser.add_argument("json_file", type=Path, help="Path to JSON file")
    
    args = parser.parse_args()
    validate_json(args.json_file)
```

### Example Data

```python
# scripts/create_example_data.py
from pathlib import Path
from src.schemas import Project, ProjectMetadata, Level, Room, Door, Wall
from datetime import datetime
import json

def create_example_project() -> Project:
    """Create example project for testing."""
    
    metadata = ProjectMetadata(
        project_name="Tech Tower",
        level_name="Planta 2",
        dwg_file="tech_tower_p2.dwg",
        extracted_date=datetime.now().isoformat()
    )
    
    level = Level(name="Planta 2", elevation=3.5)
    
    rooms = [
        Room(
            id="ROOM_001",
            name="Oficina A",
            level="Planta 2",
            boundary=[[0, 0], [10, 0], [10, 8], [0, 8], [0, 0]]
        ),
        Room(
            id="ROOM_002",
            name="Oficina B",
            level="Planta 2",
            boundary=[[10, 0], [20, 0], [20, 8], [10, 8], [10, 0]]
        )
    ]
    
    doors = [
        Door(
            id="DOOR_001",
            width=0.90,
            position=[5, 0],
            level="Planta 2",
            is_egress=True
        ),
        Door(
            id="DOOR_002",
            width=0.80,
            position=[10, 4],
            level="Planta 2",
            room_from="ROOM_001",
            room_to="ROOM_002"
        )
    ]
    
    walls = [
        Wall(
            id="WALL_001",
            start=[0, 0],
            end=[20, 0],
            length=20.0,
            level="Planta 2"
        ),
        Wall(
            id="WALL_002",
            start=[0, 8],
            end=[20, 8],
            length=20.0,
            level="Planta 2"
        )
    ]
    
    return Project(
        metadata=metadata,
        levels=[level],
        rooms=rooms,
        doors=doors,
        walls=walls
    )

if __name__ == "__main__":
    project = create_example_project()
    
    output_path = Path("data/extracted/tech_tower_p2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created example data: {output_path}")
```

---

## Pilar 2: Calculations

### Goal

Perform geometric calculations and circulation analysis.

### Geometry Module

```python
# src/calculations/geometry.py
from shapely.geometry import Polygon, Point
from shapely import distance
from typing import List, Tuple
from src.schemas import Room

def get_room_polygon(room: Room) -> Polygon:
    """Convert Room to Shapely Polygon."""
    return Polygon(room.boundary)

def calculate_room_area(room: Room) -> float:
    """Calculate room area in m²."""
    polygon = get_room_polygon(room)
    return polygon.area

def get_room_centroid(room: Room) -> Tuple[float, float]:
    """Get room centroid coordinates."""
    polygon = get_room_polygon(room)
    centroid = polygon.centroid
    return (centroid.x, centroid.y)

def calculate_perimeter(room: Room) -> float:
    """Calculate room perimeter in meters."""
    polygon = get_room_polygon(room)
    return polygon.length

def point_in_room(point: Tuple[float, float], room: Room) -> bool:
    """Check if point is inside room."""
    polygon = get_room_polygon(room)
    return polygon.contains(Point(point))

def distance_between_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
```

### Graph Module

```python
# src/calculations/graph.py
import networkx as nx
from typing import List, Dict, Optional, Tuple
from src.schemas import Project, Room, Door
from src.calculations.geometry import get_room_centroid, point_in_room

class CirculationGraph:
    """
    Represents building circulation as a graph.
    Nodes = Rooms, Edges = Doors connecting rooms.
    """
    
    def __init__(self, project: Project):
        self.project = project
        self.graph = nx.Graph()
        self._build_graph()
    
    def _build_graph(self):
        """Build NetworkX graph from project data."""
        # Add room nodes
        for room in self.project.rooms:
            centroid = get_room_centroid(room)
            self.graph.add_node(
                room.id,
                pos=centroid,
                name=room.name,
                room=room
            )
        
        # Add edges for doors
        for door in self.project.doors:
            # Find which rooms this door connects
            rooms_connected = self._find_connected_rooms(door)
            
            if len(rooms_connected) == 2:
                room1_id, room2_id = rooms_connected
                self.graph.add_edge(
                    room1_id,
                    room2_id,
                    door_id=door.id,
                    width=door.width,
                    is_egress=door.is_egress
                )
    
    def _find_connected_rooms(self, door: Door) -> List[str]:
        """Find which rooms a door connects."""
        # If explicitly defined
        if door.room_from and door.room_to:
            return [door.room_from, door.room_to]
        
        # Otherwise, find by position
        connected = []
        for room in self.project.rooms:
            if point_in_room(door.position, room):
                connected.append(room.id)
        
        return connected
    
    def shortest_path(self, from_room: str, to_room: str) -> Optional[List[str]]:
        """
        Find shortest path between two rooms.
        
        Returns:
            List of room IDs in path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, from_room, to_room)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def shortest_egress_path(self, from_room: str) -> Optional[Dict]:
        """
        Find shortest path to any egress door.
        
        Returns:
            Dict with 'path' (room IDs) and 'distance' (meters)
        """
        # Find all rooms with egress doors
        egress_rooms = []
        for door in self.project.doors:
            if door.is_egress:
                rooms = self._find_connected_rooms(door)
                egress_rooms.extend(rooms)
        
        if not egress_rooms:
            return None
        
        # Find shortest path to any egress
        shortest = None
        min_length = float('inf')
        
        for egress_room in egress_rooms:
            path = self.shortest_path(from_room, egress_room)
            if path:
                length = len(path) - 1  # Number of edges
                if length < min_length:
                    min_length = length
                    shortest = path
        
        if shortest:
            # Calculate actual distance (sum of centroid distances)
            distance = 0
            for i in range(len(shortest) - 1):
                node1 = self.graph.nodes[shortest[i]]
                node2 = self.graph.nodes[shortest[i + 1]]
                pos1 = node1['pos']
                pos2 = node2['pos']
                distance += ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)**0.5
            
            return {
                'path': shortest,
                'distance': distance
            }
        
        return None
    
    def get_path_visualization_data(self, path: List[str]) -> Dict:
        """Get data for visualizing a path."""
        positions = []
        names = []
        
        for room_id in path:
            node = self.graph.nodes[room_id]
            positions.append(node['pos'])
            names.append(node['name'])
        
        return {
            'positions': positions,
            'names': names
        }
```

---

## Pilar 3: RAG System

### Goal

Query building code documentation using Retrieval Augmented Generation.

### Document Loader

```python
# src/rag/document_loader.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.schema import Document

def load_pdfs(pdf_dir: Path) -> List[Document]:
    """
    Load all PDFs from directory.
    
    Args:
        pdf_dir: Directory containing PDF files
    
    Returns:
        List of Document objects
    """
    documents = []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)
            print(f"  ✅ Loaded {len(docs)} pages")
        except Exception as e:
            print(f"  ❌ Error loading {pdf_path.name}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents
```

### Embeddings Configuration

```python
# src/rag/embeddings_config.py
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Create embeddings instance.
    Using multilingual model for Spanish support.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
```

### Vectorstore Manager

```python
# src/rag/vectorstore_manager.py
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embeddings_config import get_embeddings
from .document_loader import load_pdfs

class VectorstoreManager:
    """Manages RAG vectorstore creation and loading."""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vectorstore: Optional[Chroma] = None
    
    def create_from_pdfs(
        self,
        pdf_dir: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Create vectorstore from PDFs.
        
        Args:
            pdf_dir: Directory with PDF files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        print("=" * 60)
        print("CREATING VECTORSTORE")
        print("=" * 60)
        
        # 1. Load documents
        print("\n1. Loading PDFs...")
        documents = load_pdfs(pdf_dir)
        
        if not documents:
            raise ValueError("No documents loaded!")
        
        # 2. Split into chunks
        print("\n2. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # 3. Create vectorstore
        print("\n3. Creating embeddings and vectorstore...")
        print("(This may take a few minutes...)")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        print(f"\n✅ Vectorstore created: {self.persist_directory}")
        print("=" * 60)
    
    def load_existing(self):
        """Load existing vectorstore."""
        if not self.persist_directory.exists():
            raise ValueError(f"Vectorstore not found: {self.persist_directory}")
        
        print(f"Loading vectorstore from: {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings
        )
        print("✅ Vectorstore loaded")
    
    def get_retriever(self, k: int = 3):
        """
        Get retriever for searches.
        
        Args:
            k: Number of documents to retrieve
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def query_simple(self, question: str, k: int = 3):
        """
        Simple query without LLM (just retrieval).
        
        Args:
            question: Query string
            k: Number of results
        
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        docs = self.vectorstore.similarity_search(question, k=k)
        return docs
```

### QA Chain

```python
# src/rag/qa_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = """
Eres un asistente experto en normativa de construcción española (CTE).
Responde preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Si la información está en el contexto, responde de forma clara y precisa
2. Siempre cita la fuente (documento, sección, página)
3. Si no puedes responder con el contexto dado, di "No tengo información suficiente"
4. No inventes ni supongas información

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""

def create_qa_chain(retriever, model_name: str = "gemini-pro", temperature: float = 0.1):
    """
    Create QA chain with retrieval.
    
    Args:
        retriever: Vectorstore retriever
        model_name: Google model name
        temperature: LLM temperature
    
    Returns:
        Configured QA chain
    """
    # LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )
    
    # Prompt
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
    
    # Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain
```

---

## Pilar 4: Agent

### Goal

Create ReAct agent that autonomously verifies building code compliance.

### Agent State

```python
# src/agent/state.py
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """State of the agent during execution."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
```

### Agent Tools

```python
# src/agent/tools.py
from langchain.tools import tool
from pathlib import Path
import json
from src.schemas import Project, Door
from src.calculations.geometry import calculate_room_area
from src.calculations.graph import CirculationGraph
from typing import Optional

# Global variable to hold project data
_project_data: Optional[Project] = None
_rag_manager = None

def load_project_data(json_path: Path):
    """Load project data into memory."""
    global _project_data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    _project_data = Project(**data)

def set_vectorstore_manager(rag):
    """Set RAG manager for normativa queries."""
    global _rag_manager
    _rag_manager = rag

@tool
def get_room_info(room_id: str) -> str:
    """
    Get information about a specific room.
    
    Args:
        room_id: ID of the room (e.g., "ROOM_001")
    
    Returns:
        String with room information
    """
    if not _project_data:
        return "Error: Project data not loaded"
    
    # Find room
    room = next((r for r in _project_data.rooms if r.id == room_id), None)
    if not room:
        return f"Room {room_id} not found"
    
    # Calculate area
    area = calculate_room_area(room)
    
    return f"""Room: {room.name} ({room.id})
Level: {room.level}
Area: {area:.2f} m²
Use: {room.use_type or 'Not specified'}
Boundary: {len(room.boundary)} points"""

@tool
def get_door_info(door_id: str) -> str:
    """
    Get information about a specific door.
    
    Args:
        door_id: ID of the door (e.g., "DOOR_001")
    
    Returns:
        String with door information
    """
    if not _project_data:
        return "Error: Project data not loaded"
    
    door = next((d for d in _project_data.doors if d.id == door_id), None)
    if not door:
        return f"Door {door_id} not found"
    
    return f"""Door: {door.id}
Width: {door.width} m ({door.width * 100:.0f} cm)
Height: {door.height} m
Position: {door.position}
Fire-rated: {'Yes' if door.is_fire_rated else 'No'}
Egress door: {'Yes' if door.is_egress else 'No'}"""

@tool
def list_all_doors() -> str:
    """
    List all doors in the project.
    
    Returns:
        String with list of all doors and their widths
    """
    if not _project_data:
        return "Error: Project data not loaded"
    
    result = f"Total doors: {len(_project_data.doors)}\n\n"
    
    for door in _project_data.doors:
        egress_mark = " [EGRESS]" if door.is_egress else ""
        result += f"- {door.id}: {door.width*100:.0f}cm wide{egress_mark}\n"
    
    return result

@tool
def check_door_width_compliance(door_id: str) -> str:
    """
    Check if a door meets minimum width requirements.
    
    Args:
        door_id: ID of the door to check
    
    Returns:
        Compliance status string
    """
    if not _project_data:
        return "Error: Project data not loaded"
    
    door = next((d for d in _project_data.doors if d.id == door_id), None)
    if not door:
        return f"Door {door_id} not found"
    
    # Minimum width standards (simplified)
    min_width = 0.80  # 80cm general
    if door.is_egress:
        min_width = 0.80  # 80cm for egress (can be higher depending on occupancy)
    
    width_cm = door.width * 100
    min_cm = min_width * 100
    
    if door.width >= min_width:
        return f"✅ COMPLIANT: Door {door.id} width ({width_cm:.0f}cm) meets minimum requirement ({min_cm:.0f}cm)"
    else:
        deficit = (min_width - door.width) * 100
        return f"❌ NON-COMPLIANT: Door {door.id} width ({width_cm:.0f}cm) is {deficit:.0f}cm below minimum ({min_cm:.0f}cm)"

@tool
def query_normativa(question: str) -> str:
    """
    Query building codes (CTE) for information.
    Use this to look up specific requirements in Spanish building regulations.
    
    Args:
        question: Question about building codes (in Spanish)
    
    Returns:
        Answer with citations from CTE
    """
    if not _rag_manager:
        return "Error: RAG system not initialized"
    
    try:
        # Use the QA chain from RAG manager
        result = _rag_manager.query(question)
        
        # Format response with sources
        answer = result['result']
        sources = result.get('source_documents', [])
        
        if sources:
            answer += "\n\nSources:"
            for i, doc in enumerate(sources[:2], 1):  # Show top 2 sources
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                answer += f"\n{i}. {source}, page {page}"
        
        return answer
    except Exception as e:
        return f"Error querying normativa: {str(e)}"

@tool
def calculate_egress_distance(room_id: str) -> str:
    """
    Calculate shortest evacuation distance from a room to an egress door.
    
    Args:
        room_id: ID of the room
    
    Returns:
        Evacuation distance and path
    """
    if not _project_data:
        return "Error: Project data not loaded"
    
    # Build graph
    graph = CirculationGraph(_project_data)
    
    # Find egress path
    result = graph.shortest_egress_path(room_id)
    
    if not result:
        return f"No egress path found from {room_id}"
    
    path = result['path']
    distance = result['distance']
    
    path_str = " → ".join([graph.graph.nodes[r]['name'] for r in path])
    
    return f"""Egress path from {room_id}:
Path: {path_str}
Distance: {distance:.2f} meters
Number of rooms traversed: {len(path)}"""

# List of all tools
TOOLS = [
    get_room_info,
    get_door_info,
    list_all_doors,
    check_door_width_compliance,
    query_normativa,
    calculate_egress_distance
]
```

### Agent Graph

```python
# src/agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from .state import AgentState
from .tools import TOOLS
from .prompts import SYSTEM_PROMPT

def create_compliance_agent():
    """
    Create ReAct agent for building compliance verification.
    
    Returns:
        Compiled LangGraph agent
    """
    # LLM with tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1
    ).bind_tools(TOOLS)
    
    # Define agent node
    def agent_node(state: AgentState) -> AgentState:
        """Agent reasoning step."""
        messages = state["messages"]
        
        # Add system prompt if first iteration
        if len(messages) == 1:
            system_msg = HumanMessage(content=SYSTEM_PROMPT)
            messages = [system_msg] + messages
        
        # Call LLM
        response = llm.invoke(messages)
        
        return {
            "messages": [response],
            "iterations": state["iterations"] + 1
        }
    
    # Define tools node
    tools_node = ToolNode(TOOLS)
    
    # Define decision function
    def should_continue(state: AgentState) -> str:
        """Decide whether to continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If LLM makes a tool call, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # If reached max iterations, end
        if state["iterations"] >= 10:
            return END
        
        # Otherwise, end
        return END
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # Compile
    return workflow.compile()
```

### System Prompts

```python
# src/agent/prompts.py

SYSTEM_PROMPT = """You are an expert building compliance verification agent.

Your job is to verify that building designs comply with Spanish building codes (CTE).

You have access to these tools:
1. get_room_info(room_id) - Get details about a room
2. get_door_info(door_id) - Get details about a door
3. list_all_doors() - List all doors in project
4. check_door_width_compliance(door_id) - Check if door meets width requirements
5. query_normativa(question) - Look up requirements in CTE documents
6. calculate_egress_distance(room_id) - Calculate evacuation distance

IMPORTANT RULES:
- Always gather information before making compliance judgments
- Use query_normativa to look up specific requirements when unsure
- Cite sources from CTE when stating requirements
- Provide clear, specific compliance status (✅ COMPLIANT or ❌ NON-COMPLIANT)
- List all issues found with specific door/room IDs

When asked to verify compliance:
1. Use list_all_doors() to see all doors
2. For each door, use check_door_width_compliance()
3. Use query_normativa() if you need to verify a requirement
4. Summarize findings clearly

Be thorough and precise in your verification."""
```

---

## Integration

### Full Pipeline Script

```python
# scripts/run_full_pipeline.py
from pathlib import Path
from src.agent.tools import load_project_data, set_vectorstore_manager
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain
from src.agent.graph import create_compliance_agent
from langchain_core.messages import HumanMessage

def main():
    """Run complete compliance verification pipeline."""
    
    print("=" * 70)
    print("AEC COMPLIANCE AGENT - FULL PIPELINE")
    print("=" * 70)
    
    # 1. Load project data
    print("\n1. Loading project data...")
    project_path = Path("data/extracted/tech_tower_p2.json")
    load_project_data(project_path)
    print("✅ Project loaded")
    
    # 2. Setup RAG
    print("\n2. Setting up RAG system...")
    vectorstore_dir = Path("vectorstore/normativa_db")
    rag = VectorstoreManager(vectorstore_dir)
    
    if vectorstore_dir.exists():
        rag.load_existing()
    else:
        print("Creating vectorstore (first time only)...")
        pdf_dir = Path("data/normativa")
        rag.create_from_pdfs(pdf_dir)
    
    # Create QA chain
    retriever = rag.get_retriever(k=3)
    qa_chain = create_qa_chain(retriever)
    
    # Set RAG manager for tools
    class SimpleRAGWrapper:
        def query(self, question):
            return qa_chain({"query": question})
    
    set_vectorstore_manager(SimpleRAGWrapper())
    print("✅ RAG system ready")
    
    # 3. Create agent
    print("\n3. Creating compliance agent...")
    agent = create_compliance_agent()
    print("✅ Agent ready")
    
    # 4. Run verification
    print("\n4. Running compliance verification...")
    print("-" * 70)
    
    query = "Verify that all doors in the project meet minimum width requirements for evacuation. Check each door individually."
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "iterations": 0
    }
    
    result = agent.invoke(initial_state)
    
    # Print agent's response
    print("\nAgent Response:")
    print("-" * 70)
    for message in result["messages"]:
        if hasattr(message, 'content') and message.content:
            print(message.content)
            print()
    
    print("=" * 70)
    print(f"Completed in {result['iterations']} iterations")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

---

## Common Issues

### Issue 1: Import Errors

**Problem**:
```
ModuleNotFoundError: No module named 'langchain'
```

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 2: Google API Key Not Found

**Problem**:
```
Error: GOOGLE_API_KEY not found in environment
```

**Solution**:
```bash
# Check .env file exists
ls -la .env

# Edit .env
nano .env  # or code .env

# Add your key
GOOGLE_API_KEY=your-actual-api-key-here
```

### Issue 3: Vectorstore Creation Fails

**Problem**:
```
Error: Out of memory when creating vectorstore
```

**Solution**:
```python
# Reduce chunk size or process fewer documents
rag.create_from_pdfs(
    pdf_dir,
    chunk_size=500,  # Was 1000
    chunk_overlap=100  # Was 200
)
```

### Issue 4: DXF Parsing Errors

**Problem**:
```
DXFStructureError: Invalid DXF file
```

**Solution**:
- Ensure DXF file is valid
- Try exporting from AutoCAD as "AutoCAD 2018 DXF"
- Check file encoding (should be UTF-8)

### Issue 5: Tests Failing

**Problem**:
```
pytest tests/unit/test_geometry.py
FAILED
```

**Solution**:
```bash
# Run with verbose output
pytest tests/unit/test_geometry.py -v

# Check specific test
pytest tests/unit/test_geometry.py::test_calculate_room_area -v

# Run with debugging
pytest tests/unit/test_geometry.py --pdb
```

---

## Next Steps

1. Follow [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for day-by-day development plan
2. Use [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing strategies
3. Refer to [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation
4. Check [GIT_WORKFLOW.md](GIT_WORKFLOW.md) for branching and commit guidelines

---

**Version**: 1.0  
**Last Updated**: October 2025