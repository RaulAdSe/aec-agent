# 🏗️ AEC Compliance Agent

**Agentic AI for Building Code Compliance Verification**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Overview

An AI-powered system that automatically verifies building designs against Spanish building codes (CTE) by:

1. **Extracting** data from architectural drawings (DWG/DXF)
2. **Calculating** geometric properties and circulation routes
3. **Querying** building codes using RAG (Retrieval Augmented Generation)
4. **Verifying** compliance autonomously using a ReAct agent

**Demo Date**: October 29, 2025 at UPC-EPSEB

---

## 🎯 Key Features

### ✅ Data Extraction
- Parse DWG/DXF files to extract rooms, doors, and walls
- Structured JSON output with Pydantic validation
- Automatic dimension calculation

### ✅ Geometric Analysis
- Room area and perimeter calculation
- Circulation graph analysis with NetworkX
- Shortest evacuation path finding

### ✅ RAG System
- Query Spanish building codes (CTE DB-SI, CTE DB-SUA)
- Semantic search with multilingual embeddings
- Citation system with source tracking

### ✅ Autonomous Agent
- ReAct framework with LangGraph
- 6 specialized tools for compliance checking
- Transparent reasoning and decision-making

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                               │
│              "Verify all doors meet requirements"            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   REACT AGENT (Pilar 4)                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Reasoning │→ │Tool Calling│→ │ Synthesis  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└───────┬─────────────────────┬─────────────────────┬─────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐   ┌──────────────┐
│ Pilar 1:     │    │ Pilar 2:         │   │ Pilar 3:     │
│ Extraction   │    │ Calculations     │   │ RAG System   │
│              │    │                  │   │              │
│ • DXF Parse  │    │ • Geometry       │   │ • Vectorstore│
│ • JSON Valid │    │ • Graph Analysis │   │ • QA Chain   │
│ • Schemas    │    │ • Routes         │   │ • Citations  │
└──────────────┘    └──────────────────┘   └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Compliance Report│
                    └──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- Google API key (for Gemini)
- 4GB+ RAM

### Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/aec-compliance-agent.git
cd aec-compliance-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Basic Usage

```python
from pathlib import Path
from src.agent.tools import load_project_data, set_vectorstore_manager
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain
from src.agent.graph import create_compliance_agent
from langchain_core.messages import HumanMessage

# 1. Load project data
load_project_data(Path("data/extracted/tech_tower_p2.json"))

# 2. Setup RAG
rag = VectorstoreManager(Path("vectorstore/normativa_db"))
rag.load_existing()
retriever = rag.get_retriever(k=3)
qa_chain = create_qa_chain(retriever)

class RAGWrapper:
    def query(self, question):
        return qa_chain({"query": question})

set_vectorstore_manager(RAGWrapper())

# 3. Create agent
agent = create_compliance_agent()

# 4. Verify compliance
result = agent.invoke({
    "messages": [HumanMessage(content="Verify all doors meet minimum width requirements")],
    "iterations": 0
})

# Print results
for msg in result["messages"]:
    if hasattr(msg, 'content') and msg.content:
        print(msg.content)
```

### Run Full Pipeline

```bash
# Create example data (first time only)
python scripts/create_example_data.py

# Create vectorstore (first time only)
python scripts/create_vectorstore.py

# Run complete verification
python scripts/run_full_pipeline.py
```

---

## 📁 Project Structure

```
aec-compliance-agent/
├── src/                    # Source code
│   ├── extraction/         # Pilar 1: DXF extraction
│   ├── calculations/       # Pilar 2: Geometry & graphs
│   ├── rag/               # Pilar 3: RAG system
│   ├── agent/             # Pilar 4: ReAct agent
│   └── utils/             # Utilities
│
├── data/                   # Data directory
│   ├── raw/dwg/           # Input DWG files
│   ├── extracted/         # Extracted JSON
│   └── normativa/         # Building codes (PDFs)
│
├── notebooks/              # Tutorial notebooks ⭐
│   ├── 01_extraction_tutorial.ipynb
│   ├── 02_calculations_tutorial.ipynb
│   ├── 03_rag_tutorial.ipynb
│   └── 04_agent_tutorial.ipynb
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
│
└── docs/                   # Documentation
    ├── TECH_STACK.md
    ├── DEVELOPMENT_GUIDE.md
    ├── RAG_EXPLAINED.md
    ├── TESTING_GUIDE.md
    └── API_REFERENCE.md
```

---

## 🛠️ Tech Stack

### Core Technologies

- **Python 3.10+**: Main language
- **LangGraph**: Agent orchestration
- **LangChain**: RAG pipeline
- **Google Gemini**: LLM (gemini-pro)

### Data Processing

- **ezdxf**: DXF file parsing
- **Pydantic**: Data validation
- **Shapely**: Geometric calculations
- **NetworkX**: Graph analysis

### RAG System

- **ChromaDB**: Vector database
- **HuggingFace Embeddings**: Multilingual embeddings
- **PyPDFLoader**: PDF document loading

### Development

- **pytest**: Testing framework
- **black**: Code formatting
- **Jupyter**: Interactive notebooks

See [TECH_STACK.md](docs/TECH_STACK.md) for detailed version info.

---

## 📚 Documentation

### For Developers

- **[DEVELOPMENT_GUIDE.md](docs/DEVELOPMENT_GUIDE.md)** - Complete implementation guide
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Full API documentation
- **[TESTING_GUIDE.md](docs/TESTING_GUIDE.md)** - Testing strategies
- **[GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)** - Git branching and commits

### Conceptual Explanations

- **[RAG_EXPLAINED.md](docs/RAG_EXPLAINED.md)** - Deep dive into RAG ⭐
- **[REACT_EXPLAINED.md](docs/REACT_EXPLAINED.md)** - ReAct framework explained ⭐
- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Architecture details

### Planning

- **[IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md)** - 15-day development plan

---

## 🎓 Tutorial Notebooks

Interactive Jupyter notebooks for hands-on learning:

1. **[01_extraction_tutorial.ipynb](notebooks/01_extraction_tutorial.ipynb)**
   - Extract data from DXF files
   - Validate with Pydantic schemas
   - Visualize extracted data

2. **[02_calculations_tutorial.ipynb](notebooks/02_calculations_tutorial.ipynb)**
   - Calculate room areas and perimeters
   - Build circulation graphs
   - Find evacuation routes

3. **[03_rag_tutorial.ipynb](notebooks/03_rag_tutorial.ipynb)** ⭐
   - Create vectorstore from PDFs
   - Query building codes semantically
   - Understand RAG pipeline

4. **[04_agent_tutorial.ipynb](notebooks/04_agent_tutorial.ipynb)** ⭐
   - Run ReAct agent
   - Observe reasoning process
   - Verify compliance autonomously

**Start notebooks**:
```bash
jupyter notebook notebooks/
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Fast tests only (skip slow integration tests)
pytest -m "not slow"

# Specific test file
pytest tests/unit/test_geometry.py
```

### Test Coverage

- **Target**: 85%+ overall
- **Unit Tests**: 90%+ coverage
- **Integration Tests**: Key workflows covered

See [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for detailed testing documentation.

---

## 🎯 Use Cases

### 1. Door Width Verification

```python
# Agent automatically:
# 1. Lists all doors
# 2. Checks each door width
# 3. Queries CTE for requirements
# 4. Reports compliance status
```

**Output**:
```
Checking door compliance:
- DOOR_001 (90cm): ✅ COMPLIANT
- DOOR_002 (80cm): ✅ COMPLIANT
- DOOR_003 (75cm): ❌ NON-COMPLIANT (5cm below minimum)

Summary: 2/3 doors compliant
```

### 2. Evacuation Distance Check

```python
# Agent automatically:
# 1. Builds circulation graph
# 2. Calculates shortest paths to exits
# 3. Queries CTE for max distances
# 4. Reports exceedances
```

**Output**:
```
Evacuation distances:
- ROOM_001 → EXIT: 12.5m ✅ (max: 25m)
- ROOM_002 → EXIT: 18.3m ✅ (max: 25m)
- ROOM_003 → EXIT: 28.1m ❌ EXCEEDS by 3.1m

Action required: ROOM_003 needs additional exit
```

### 3. Natural Language Queries

```python
# Query: "What are the fire rating requirements for walls?"
```

**Output**:
```
According to CTE DB-SI Section 1:
- Compartmentation walls: EI-60 minimum
- Sector walls: EI-90 minimum
- Facade walls: EI-60 exterior side

Sources:
- CTE_DB-SI.pdf, page 15
- CTE_DB-SI.pdf, page 23
```

---

## 🚧 Limitations (POC)

This is a **Proof of Concept** with known limitations:

### Data Extraction
- ❌ Only supports basic DXF elements (LINE, LWPOLYLINE, INSERT)
- ❌ No support for complex blocks or nested structures
- ❌ Manual door-to-room assignment needed in some cases

### RAG System
- ❌ Limited to CTE documents only
- ❌ No real-time updates to building codes
- ❌ Spanish-only (multilingual support possible but not tested)

### Agent
- ❌ Single agent (no multi-agent collaboration)
- ❌ Limited to 6 tools
- ❌ No learning or memory across sessions

### Compliance
- ❌ Simplified rules (not comprehensive)
- ❌ No handling of exceptions or special cases
- ⚠️ **NOT FOR PRODUCTION USE**

---

## 🔮 Future Enhancements

### Short-term (v1.1)

- [ ] Support for more DXF entities (CIRCLE, ARC, SPLINE)
- [ ] Better door-to-room automatic detection
- [ ] More comprehensive compliance rules
- [ ] Export compliance reports to PDF

### Medium-term (v2.0)

- [ ] Computer vision for PDF plan analysis
- [ ] Multi-agent collaboration (specialist agents)
- [ ] Real-time code updates

### Long-term (v3.0)

- [ ] Automated design suggestions for compliance
- [ ] 3D model analysis
- [ ] Multi-country building code support
- [ ] Cloud-based service

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add: amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

See [GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for detailed guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linter
flake8 src/ tests/

# Format code
black src/ tests/

# Run tests
pytest
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Raúl Adell** - *Initial work* - [GitHub](https://github.com/your-username)

---

## 🙏 Acknowledgments

- **UPC-EPSEB** for the invitation to present
- **Antoni Guillamon, Adrià Tauste, Semih Gönen** for organizing the course
- **LangChain** and **LangGraph** teams for the frameworks
- **Google** for Gemini API access

---

## 📞 Contact

- **Email**: rauladellsegarra@gmail.com
- **LinkedIn**: [Raúl Adell](https://linkedin.com/in/your-profile)
- **Website**: [raul.sinluz.com](https://raul.sinluz.com)

---

## 📅 Presentation Info

**Date**: October 29, 2025  
**Time**: 10:00 - 10:45 AM  
**Location**: UPC-EPSEB  
**Course**: Intel·ligència Artificial Aplicada a la Construcció

**Topics Covered**:
1. AI for technical office workflows
2. RAG for building code queries
3. Autonomous agents for compliance verification
4. Live demo of the system

---

## 🎬 Demo Queries

Try these queries with the agent:

```python
# Basic verification
"List all doors in the project"

# Compliance checking
"Verify that all doors meet minimum width requirements"

# Evacuation analysis
"Calculate evacuation distances from all rooms"

# Normativa queries
"What does CTE say about fire ratings for walls?"

# Complex analysis
"Check if any rooms have excessive evacuation distances and list non-compliant doors"
```

---

## 📊 Project Status

- [x] Pilar 1: Data Extraction - **COMPLETE**
- [x] Pilar 2: Calculations - **COMPLETE**
- [x] Pilar 3: RAG System - **COMPLETE**
- [x] Pilar 4: ReAct Agent - **COMPLETE**
- [x] Documentation - **COMPLETE**
- [x] Tutorial Notebooks - **COMPLETE**
- [x] Testing Suite - **COMPLETE**
- [ ] Production Deployment - **NOT PLANNED** (POC only)

---

## ⚡ Performance

Typical execution times:

| Operation | Time |
|-----------|------|
| DXF Extraction (single floor) | 1-2 seconds |
| JSON Validation | < 1 second |
| Area Calculation (100 rooms) | < 1 second |
| Graph Analysis | < 1 second |
| Vectorstore Creation (200 pages) | 2-5 minutes |
| RAG Query | 2-3 seconds |
| Agent Execution (simple) | 10-20 seconds |
| Agent Execution (complex) | 30-60 seconds |

**Hardware**: Intel i5, 8GB RAM, No GPU

---

## 🐛 Known Issues

1. **DXF Parsing**: Some complex blocks not recognized
2. **RAG Accuracy**: Varies with query phrasing (Spanish)
3. **Agent Iterations**: Sometimes requires retry for complex queries
4. **Memory**: Vectorstore creation requires 2GB+ RAM

See [Issues](https://github.com/your-username/aec-compliance-agent/issues) for full list.

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{aec_compliance_agent,
  author = {Adell, Raúl},
  title = {AEC Compliance Agent: Agentic AI for Building Code Verification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/aec-compliance-agent}
}
```

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**Built with ❤️ for the AEC industry**

**Last Updated**: October 2025  
**Version**: 1.0.0