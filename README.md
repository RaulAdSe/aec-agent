# 🏗️ AEC Compliance Agent

## Agentic AI for Building Code Compliance Verification

An autonomous AI agent that verifies building code compliance by extracting data from architectural drawings (DWG/DXF), performing geometric calculations, and querying Spanish building codes using RAG technology.

## 🎯 Project Overview

**Presentation**: October 29, 2025 at UPC-EPSEB  
**Course**: Intel·ligència Artificial Aplicada a la Construcció  
**Type**: Proof of Concept (POC)

### Key Features

- **📦 Data Extraction**: Parse DWG/DXF files
- **📐 Geometric Analysis**: Calculate areas, distances, and evacuation routes
- **📚 RAG System**: Query Spanish building codes (CTE DB-SI, CTE DB-SUA)
- **🤖 ReAct Agent**: Autonomous compliance verification with LangGraph

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google API key (for Gemini LLM)
- 4GB RAM (for embeddings)

### Installation

```bash
# Clone repository
git clone https://github.com/RaulAdSe/aec-agent.git
cd aec-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run Example

```bash
# Extract data from CAD files
python scripts/extract_cad_files.py

# Run compliance verification
python scripts/run_full_pipeline.py
```

### Tutorial Notebooks

Open Jupyter and explore the tutorial notebooks:

```bash
jupyter notebook

# Navigate to notebooks/
# Start with 01_data_extraction_simple.ipynb
```

## 📁 Project Structure

```
aec-agent/
├── src/                    # Source code
│   ├── extraction/         # DWG/DXF extraction
│   ├── calculations/       # Geometry and graph analysis
│   ├── rag/               # RAG system for normativa
│   └── agent/             # ReAct agent implementation
├── data/                  # Data files
│   ├── blueprints/        # Original CAD files
│   ├── extracted/         # JSON extracted data
│   └── normativa/         # PDF building codes
├── notebooks/             # Tutorial notebooks
├── tests/                 # Test suite
└── scripts/               # Utility scripts
```

## 🛠️ Technology Stack

- **LLM**: Google Gemini (gemini-2.0-flash-exp)
- **Agent Framework**: LangChain + LangGraph
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace Multilingual
- **CAD Processing**: ezdxf
- **Geometry**: Shapely + NetworkX
- **Validation**: Pydantic

## 📚 Documentation

- [Development Guide](docs/DEVELOPMENT_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [RAG Explained](docs/RAG_EXPLAINED.md)
- [ReAct Framework](docs/REACT_EXPLAINED.md)
- [Testing Guide](docs/TESTING_GUIDE.md)

## 🔬 Current Test Files

- **CAD Files**:
  - `I01.4 PCI - EXTINCIÓN AUTOMÁTICA.dwg` - Fire extinguishing systems
  - `I01.6 PCI - SECTORIZACIÓN.dwg` - Fire compartmentation

## 👥 Author

**Raul Adell Segarra**  
UPC-EPSEB Student  
[GitHub](https://github.com/RaulAdSe)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- UPC-EPSEB for the opportunity
- Course instructors for guidance
- Open source community for amazing tools

---

*Built with ❤️ for the AEC industry*