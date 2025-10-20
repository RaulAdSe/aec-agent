# 📁 Project Structure - Agentic AI for AEC

## 🌳 Árbol Completo del Proyecto

```
aec-compliance-agent/
│
├── 📄 README.md                      # Documentación principal
├── 📄 requirements.txt               # Dependencias Python
├── 📄 .env.example                   # Template para variables de entorno
├── 📄 .env                           # Variables de entorno (NO committear)
├── 📄 .gitignore                     # Archivos a ignorar en Git
├── 📄 LICENSE                        # Licencia MIT
│
├── 📁 docs/                          # 📚 Documentación técnica
│   ├── README.md                     # Índice de documentación
│   ├── TECH_STACK.md                 # Stack tecnológico detallado
│   ├── PROJECT_STRUCTURE.md          # ⭐ Este archivo
│   ├── DEVELOPMENT_GUIDE.md          # Guía para developers
│   ├── RAG_EXPLAINED.md              # Explicación técnica de RAG
│   ├── REACT_EXPLAINED.md            # Explicación técnica de ReAct
│   ├── TESTING_GUIDE.md              # Guía de testing
│   └── API_REFERENCE.md              # Referencia de APIs
│
├── 📁 notebooks/                     # 📓 Tutoriales Jupyter
│   ├── 01_extraction_tutorial.ipynb  # Pilar 1: Extracción de datos
│   ├── 02_calculations_tutorial.ipynb# Pilar 2: Cálculos geométricos
│   ├── 03_rag_tutorial.ipynb         # Pilar 3: RAG
│   ├── 04_agent_tutorial.ipynb       # Pilar 4: Agente ReAct
│   └── 00_complete_demo.ipynb        # Demo completa integrada
│
├── 📁 src/                           # 💻 Código fuente
│   ├── __init__.py
│   │
│   ├── 📁 extraction/                # Pilar 1: Extracción
│   │   ├── __init__.py
│   │   ├── dxf_export.py             # Script para DWG/DXF (ezdxf)
│   │   └── json_validator.py         # Validación de JSON extraídos
│   │
│   ├── 📁 calculations/              # Pilar 2: Cálculos
│   │   ├── __init__.py
│   │   ├── geometry.py               # Funciones de Shapely
│   │   ├── graph.py                  # Funciones de NetworkX
│   │   └── metrics.py                # Métricas de proyecto
│   │
│   ├── 📁 rag/                       # Pilar 3: RAG
│   │   ├── __init__.py
│   │   ├── document_loader.py        # Carga de PDFs
│   │   ├── vectorstore_manager.py    # Setup de ChromaDB
│   │   ├── embeddings_config.py      # Configuración de embeddings
│   │   └── qa_chain.py               # Chains de QA
│   │
│   ├── 📁 agent/                     # Pilar 4: Agente
│   │   ├── __init__.py
│   │   ├── tools.py                  # Definición de tools
│   │   ├── state.py                  # State management
│   │   ├── graph.py                  # LangGraph setup
│   │   └── prompts.py                # System prompts
│   │
│   ├── 📁 utils/                     # Utilidades
│   │   ├── __init__.py
│   │   ├── config.py                 # Configuración global
│   │   ├── logger.py                 # Setup de logging
│   │   └── visualization.py          # Funciones de visualización
│   │
│   └── 📄 schemas.py                 # Pydantic models globales
│
├── 📁 data/                          # 💾 Datos
│   ├── 📁 raw/                       # Archivos originales
│   │   └── 📁 dwg/                   # Archivos CAD
│   │
│   ├── 📁 extracted/                 # JSON extraídos
│   │   ├── project_001.json
│   │   ├── project_002.json
│   │   └── tech_tower_p2.json        # Proyecto de ejemplo
│   │
│   └── 📁 normativa/                 # Documentos de normativa
│       ├── CTE_DB-SI.pdf
│       ├── CTE_DB-SUA.pdf
│       ├── RD_513_2017_RIPCI.pdf
│       └── README.md                 # Índice de documentos
│
├── 📁 vectorstore/                   # 🗄️ Base de datos vectorial
│   └── 📁 normativa_db/              # ChromaDB persistence
│       ├── chroma.sqlite3
│       └── {uuid}/                   # Collections
│
├── 📁 tests/                         # 🧪 Tests
│   ├── __init__.py
│   ├── conftest.py                   # Fixtures de pytest
│   │
│   ├── 📁 unit/                      # Tests unitarios
│   │   ├── test_geometry.py
│   │   ├── test_graph.py
│   │   ├── test_tools.py
│   │   └── test_rag.py
│   │
│   ├── 📁 integration/               # Tests de integración
│   │   ├── test_extraction_pipeline.py
│   │   ├── test_rag_pipeline.py
│   │   └── test_agent_pipeline.py
│   │
│   └── 📁 fixtures/                  # Datos de test
│       ├── sample_project.json
│       ├── sample_normativa.txt
│       └── expected_outputs.json
│
├── 📁 outputs/                       # 📊 Outputs generados
│   ├── 📁 reports/                   # Informes de compliance
│   │   ├── report_20251018_project001.pdf
│   │   └── report_20251018_project001.json
│   │
│   ├── 📁 logs/                      # Logs de ejecución
│   │   └── agent_20251018.log
│   │
│   └── 📁 visualizations/            # Visualizaciones
│       └── floorplan_with_routes.png
│
├── 📁 scripts/                       # 🔧 Scripts auxiliares
│   ├── setup_environment.sh          # Setup inicial
│   ├── download_normativa.py         # Descarga PDFs de normativa
│   ├── create_example_data.py        # Genera datos de ejemplo
│   └── run_full_pipeline.py          # Ejecuta pipeline completo
│
└── 📁 experiments/                   # 🧪 Experimentos (opcional)
    ├── prompt_variations/            # Probar diferentes prompts
    ├── rag_configurations/           # Probar configs de RAG
    └── benchmarks/                   # Benchmarks de rendimiento
```

---

## 📂 Descripción de Carpetas Principales

### 1. `docs/` - Documentación

**Propósito**: Toda la documentación técnica del proyecto.

```
docs/
├── README.md                 # Índice y navegación
├── TECH_STACK.md            # Tecnologías usadas
├── RAG_EXPLAINED.md         # Tutorial RAG
├── REACT_EXPLAINED.md       # Tutorial ReAct
├── TESTING_GUIDE.md         # Cómo hacer tests
└── API_REFERENCE.md         # Docs de funciones/clases
```

**Cuándo añadir archivos**:
- Nuevo concepto técnico → Nuevo .md
- Decisión de arquitectura → Añadir a DEVELOPMENT_GUIDE.md
- Nueva API pública → Actualizar API_REFERENCE.md

---

### 2. `notebooks/` - Tutoriales Jupyter

**Propósito**: Tutoriales interactivos para aprender cada pilar.

```
notebooks/
├── 01_extraction_tutorial.ipynb      # ~45 min
├── 02_calculations_tutorial.ipynb    # ~30 min
├── 03_rag_tutorial.ipynb             # ~60 min (más importante)
├── 04_agent_tutorial.ipynb           # ~60 min (más importante)
└── 00_complete_demo.ipynb            # ~20 min (integración)
```

**Estructura de cada notebook**:
```markdown
# Título del Pilar

## 1. Introducción
- ¿Qué es?
- ¿Por qué lo necesitamos?
- Conceptos clave

## 2. Setup
- Imports
- Configuración

## 3. Teoría
- Explicación técnica
- Diagramas
- Ejemplos

## 4. Práctica
- Código ejecutable
- Visualizaciones
- Ejercicios

## 5. Integración
- Cómo se conecta con otros pilares

## 6. Conclusiones
- Recap
- Próximos pasos
```

---

### 3. `src/` - Código Fuente

**Propósito**: Código modular y reutilizable.

#### 3.1 `src/extraction/`

```python
Extrae rooms, doors, walls, exits → JSON
"""

# src/extraction/dxf_export.py
"""
Script Python puro para parsear DXF con ezdxf.
Extrae bloques, layers, textos → JSON
"""

# src/extraction/json_validator.py
"""
Valida que el JSON extraído cumpla con el schema de Pydantic.
"""
from pydantic import ValidationError
from src.schemas import Project

def validate_project_json(json_path: str) -> bool:
    with open(json_path) as f:
        data = json.load(f)
    try:
        Project(**data)
        return True
    except ValidationError as e:
        print(f"Validation errors: {e}")
        return False
```

#### 3.2 `src/calculations/`

```python
# src/calculations/geometry.py
"""
Funciones de geometría 2D con Shapely.
"""
from shapely.geometry import Polygon, Point
from typing import Tuple

def get_room_polygon(room_data: dict) -> Polygon:
    """Convierte boundary a Shapely Polygon"""
    pass

def calculate_area(polygon: Polygon) -> float:
    """Calcula área del polígono"""
    return polygon.area

def get_centroid(polygon: Polygon) -> Tuple[float, float]:
    """Calcula centroide"""
    return (polygon.centroid.x, polygon.centroid.y)
```

```python
# src/calculations/graph.py
"""
Funciones de análisis de grafos con NetworkX.
"""
import networkx as nx

class CirculationGraph:
    """Grafo de circulación para análisis de evacuación"""
    
    def __init__(self, project_data: dict):
        self.G = self._build_graph(project_data)
    
    def shortest_egress_path(self, start: tuple, exits: list) -> dict:
        """Calcula ruta de evacuación más corta"""
        pass
```

#### 3.3 `src/rag/`

```python
# src/rag/vectorstore_manager.py
"""
Gestión del vectorstore de ChromaDB.
"""
from langchain_community.vectorstores import Chroma
from pathlib import Path

class VectorstoreManager:
    """Gestiona vectorstore de normativa"""
    
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.vectorstore = None
    
    def create_from_pdfs(self, pdf_dir: Path) -> Chroma:
        """Crea vectorstore desde PDFs"""
        pass
    
    def load_existing(self) -> Chroma:
        """Carga vectorstore existente"""
        pass
    
    def query(self, question: str, k: int = 3) -> list:
        """Consulta el vectorstore"""
        pass
```

#### 3.4 `src/agent/`

```python
# src/agent/tools.py
"""
Definición de todas las tools del agente.
"""
from langchain_core.tools import tool

@tool
def get_room_info(room_id: str) -> dict:
    """Get detailed information about a room."""
    pass

@tool
def check_door_width_compliance(door_id: str, min_mm: float) -> dict:
    """Check if a door meets minimum width requirement."""
    pass

# ... más tools
```

```python
# src/agent/graph.py
"""
Configuración del grafo de LangGraph.
"""
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.tools import tools

def create_compliance_agent() -> StateGraph:
    """Crea y configura el agente de compliance"""
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END}
    )
    
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    
    return workflow.compile()
```

---

### 4. `data/` - Datos

```
data/
├── raw/              # Archivos originales (DWG)
├── extracted/        # JSON procesados
└── normativa/        # PDFs de normativa
```

**Naming conventions**:
```
# JSON extraídos
project_{id}_{date}.json
tech_tower_p2_20251018.json

# PDFs normativa
CTE_DB-{section}.pdf
RD_{number}_{year}_{name}.pdf
```

**.gitignore**:
```
# Ignorar archivos grandes
data/raw/*.dwg

# Pero SÍ incluir JSONs y PDFs
!data/extracted/*.json
!data/normativa/*.pdf
```

---

### 5. `vectorstore/` - ChromaDB

```
vectorstore/
└── normativa_db/
    ├── chroma.sqlite3          # Metadata
    └── {uuid}/                 # Vectores
        ├── data_level0.bin
        ├── header.bin
        └── length.bin
```

**No committear** (añadir a `.gitignore`):
```
vectorstore/*
```

Pero incluir script para regenerar:
```python
# scripts/rebuild_vectorstore.py
"""Regenera vectorstore desde PDFs"""
```

---

### 6. `tests/` - Testing

```
tests/
├── unit/             # Tests de funciones individuales
├── integration/      # Tests de pipelines completos
└── fixtures/         # Datos de test
```

**Ejemplo**:
```python
# tests/unit/test_geometry.py
import pytest
from src.calculations.geometry import calculate_area
from shapely.geometry import Polygon

def test_calculate_area_rectangle():
    """Test área de rectángulo simple"""
    rect = Polygon([(0,0), (10,0), (10,5), (0,5)])
    assert calculate_area(rect) == 50.0

def test_calculate_area_complex():
    """Test área de polígono complejo"""
    # ...
```

---

### 7. `outputs/` - Outputs Generados

```
outputs/
├── reports/              # Informes PDF/JSON
├── logs/                 # Logs de ejecución
└── visualizations/       # Gráficos
```

**Auto-generados**, NO committear:
```
outputs/*
```

---

## 🎯 Convenciones de Nomenclatura

### Archivos Python

```
snake_case.py           # Módulos
PascalCase             # Clases
snake_case()           # Funciones
UPPER_SNAKE_CASE       # Constantes
```

### Notebooks

```
{orden}_{nombre}_tutorial.ipynb

Ejemplos:
01_extraction_tutorial.ipynb
02_calculations_tutorial.ipynb
```

### Datos

```
{tipo}_{id}_{fecha}.{ext}

Ejemplos:
project_001_20251018.json
report_tech_tower_20251018.pdf
```

---

## 📝 Archivos de Configuración

### `.env.example`

```bash
# Google AI (Gemini)
GOOGLE_API_KEY=your-api-key-here

# OpenAI (opcional)
# OPENAI_API_KEY=sk-...

# Anthropic (opcional)
# ANTHROPIC_API_KEY=sk-ant-...

# LangSmith (opcional, para debugging)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls__...

# Project settings
PROJECT_ROOT=/path/to/project
VECTORSTORE_DIR=./vectorstore
DATA_DIR=./data
```

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
.env                    # ⚠️ NUNCA committear API keys
vectorstore/            # Regenerable
outputs/                # Auto-generados
data/raw/*.dwg

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
```

### `requirements.txt`

```txt
# Ver docs/TECH_STACK.md para versiones exactas
python-dotenv==1.0.0
pydantic==2.5.3
# ... resto de dependencias
```

---

## 🚀 Inicialización del Proyecto

### Script de Setup

```bash
# scripts/setup_environment.sh

#!/bin/bash

echo "🚀 Setting up AEC Compliance Agent..."

# 1. Crear estructura de carpetas
mkdir -p data/{raw/dwg,extracted,normativa}
mkdir -p vectorstore
mkdir -p outputs/{reports,logs,visualizations}
mkdir -p tests/{unit,integration,fixtures}

# 2. Crear .env desde template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys"
fi

# 3. Instalar dependencias
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# 4. Crear datos de ejemplo
echo "📊 Creating example data..."
python scripts/create_example_data.py

# 5. Verificar instalación
echo "✅ Running verification tests..."
python -m pytest tests/unit/test_geometry.py -v

echo "✨ Setup complete!"
echo "Next steps:"
echo "  1. Edit .env and add your GOOGLE_API_KEY"
echo "  2. Run: jupyter notebook notebooks/01_extraction_tutorial.ipynb"
```

---

## 📋 Checklist de Implementación

Para el developer, orden de implementación:

### Fase 1: Setup (Día 1)
- [ ] Clonar estructura de carpetas
- [ ] Crear `requirements.txt`
- [ ] Setup `.env.example` y `.gitignore`
- [ ] Instalar dependencias
- [ ] Verificar imports

### Fase 2: Pilar 1 - Extracción (Día 2-3)
- [ ] Implementar `src/extraction/dxf_export.py`
- [ ] Crear `src/schemas.py` con Pydantic models
- [ ] Crear datos de ejemplo en `data/extracted/`
- [ ] Crear `notebooks/01_extraction_tutorial.ipynb`
- [ ] Tests: `tests/unit/test_extraction.py`

### Fase 3: Pilar 2 - Cálculos (Día 4-5)
- [ ] Implementar `src/calculations/geometry.py`
- [ ] Implementar `src/calculations/graph.py`
- [ ] Crear visualizaciones en `src/utils/visualization.py`
- [ ] Crear `notebooks/02_calculations_tutorial.ipynb`
- [ ] Tests: `tests/unit/test_geometry.py`, `test_graph.py`

### Fase 4: Pilar 3 - RAG (Día 6-8)
- [ ] Descargar PDFs de normativa a `data/normativa/`
- [ ] Implementar `src/rag/vectorstore_manager.py`
- [ ] Implementar `src/rag/qa_chain.py`
- [ ] Crear `notebooks/03_rag_tutorial.ipynb`
- [ ] Tests: `tests/integration/test_rag_pipeline.py`

### Fase 5: Pilar 4 - Agente (Día 9-12)
- [ ] Implementar `src/agent/tools.py` (todas las tools)
- [ ] Implementar `src/agent/state.py`
- [ ] Implementar `src/agent/graph.py`
- [ ] Crear `notebooks/04_agent_tutorial.ipynb`
- [ ] Tests: `tests/integration/test_agent_pipeline.py`

### Fase 6: Integración (Día 13-14)
- [ ] Crear `notebooks/00_complete_demo.ipynb`
- [ ] Crear script `scripts/run_full_pipeline.py`
- [ ] Documentar en `docs/API_REFERENCE.md`
- [ ] README.md completo con ejemplos

### Fase 7: Testing & Docs (Día 15)
- [ ] Ejecutar toda la suite de tests
- [ ] Generar coverage report
- [ ] Revisar toda la documentación
- [ ] Preparar presentación del 29

---

**Total estimado: 15 días de desarrollo**

---

## 🔗 Enlaces Entre Componentes

```
┌─────────────────────────────────────────────────────┐
│              notebooks/                              │
│  (Tutoriales interactivos)                          │
└────────────────┬────────────────────────────────────┘
                 │ importan y usan
                 ▼
┌─────────────────────────────────────────────────────┐
│              src/                                    │
│  (Código fuente modular)                            │
│                                                      │
│  extraction/ → calculations/ → rag/ → agent/       │
└────────────────┬────────────────────────────────────┘
                 │ leen/escriben
                 ▼
┌─────────────────────────────────────────────────────┐
│              data/                                   │
│  (Datos y normativa)                                │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│              outputs/                                │
│  (Resultados generados)                             │
└─────────────────────────────────────────────────────┘
```

---

**Versión**: 1.0  
**Última actualización**: Octubre 2025
