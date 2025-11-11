# AEC Compliance Agent 🏗️

**Building Code Compliance Verification System** with Document Retrieval and AI-powered Analysis.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY="your-gemini-api-key"
```

### 2. Add Documents & Sync Knowledge Base
```bash
# Add your compliance documents to data/doc/
cp your-building-codes.pdf data/doc/

# Sync documents to knowledge base
./kb sync
# or: python3 scripts/manage_knowledge_base.py sync
```

### 3. Query Compliance Information
```bash
# Query from command line
./kb query "minimum door width for accessibility"

# Check status
./kb status
```

### 4. Use in Your Agent
```python
from aec_agent.tools import search_compliance_docs

result = search_compliance_docs("fire safety door requirements")
print(result["answer"])
```

## 🔑 API Keys Required

| Service | Purpose | Required |
|---------|---------|----------|
| **Google Gemini** | Document search & AI analysis | ✅ Yes |
| **OpenAI** | Alternative LLM (optional) | 🔄 Optional |

## 🎯 Features

### 📚 Document Retrieval System
- **Gemini File Search**: Semantic search across compliance documents
- **Multi-format Support**: PDF, TXT, DOCX, JSON, MD files
- **Automatic Indexing**: Smart duplicate detection and incremental updates
- **Citation Support**: Track sources for compliance answers

### 🤖 AI-Powered Analysis
- **Building Data Tools**: IFC file processing and element extraction
- **Compliance Validation**: Rule-based compliance checking
- **Spatial Analysis**: Geometric calculations and relationships
- **Natural Language Queries**: Ask questions in plain language

### 🏗️ Professional Architecture
- **Modular Design**: Separate tools for different functions
- **Comprehensive Testing**: Unit and integration tests
- **Clean APIs**: Easy integration with agents
- **Extensible**: Add new compliance domains easily

## 📖 **Complete Documentation**

📚 **[View Full Documentation →](docs/README.md)**

| Guide | Purpose |
|-------|---------|
| **[Getting Started](docs/GETTING_STARTED.md)** | Complete setup and tutorial |
| **[System Overview](docs/SYSTEM_OVERVIEW.md)** | Architecture and components |
| **[API Reference](docs/API_REFERENCE.md)** | Function documentation |
| **[Script Reference](docs/SCRIPT_REFERENCE.md)** | All scripts explained |
| **[Citation Features](docs/CITATION_FEATURES.md)** | Source tracking system |

## 📁 Architecture

```
aec-compliance-agent/
├── aec_agent/                            # 📦 Main Package
│   ├── tools/                            # 🔧 Agent tools
│   │   ├── building_data_toolkit.py     # IFC & building analysis
│   │   ├── document_retrieval_toolkit.py # Gemini File Search
│   │   └── compliance_search.py          # Agent-friendly search
│   ├── agents/                           # 🤖 Agent implementations
│   ├── core/                             # ⚙️ Framework
│   └── utils/                            # 🛠️ Utilities
├── bin/                                  # 🎯 Executable scripts
│   └── kb-manager                        # Knowledge base management
├── examples/                             # 📚 Usage examples
├── tests/                                # 🧪 Test suite
├── docs/                                 # 📖 Documentation
├── data/                                 # 📄 Documents & data
│   ├── doc/                              # Your compliance docs
│   └── out/                              # Processed outputs
└── kb                                    # 🎯 Convenience script
```

## 🛠️ Development

### Add New Tools
```python
# aec_agent/tools/your_new_tool.py
def your_compliance_function(input_params: str) -> str:
    # Your compliance logic here
    return "Analysis result"
```

### Add New Agents
```bash
mkdir aec_agent/agents/your_agent
# Implement agent.py, config.py, prompts.py
```

### LangSmith Monitoring
- View traces in [LangSmith dashboard](https://smith.langchain.com/)
- Use LangChain Studio for visual debugging
- All agent operations automatically tracked

## 🔧 Configuration

```bash
# .env file
GOOGLE_API_KEY=your_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=aec-compliance-agent
```

## 📋 Usage Examples

### Streamlit Web App
1. Open `http://localhost:8501`
2. Upload JSON building data or use sample
3. Select analysis type (general/fire_safety/accessibility)
4. Get compliance report with downloadable results

### CLI Analysis
```bash
# Analyze building file
python -m src.main analyze data/building.json --analysis-type fire_safety

# Initialize new project
python -m src.main init-project my-building

# Check status
python -m src.main status
```

## 🎯 Spanish Building Codes Supported

- **CTE DB-SI**: Fire Safety (Seguridad en caso de incendio)
- **CTE DB-SUA**: Accessibility (Seguridad de utilización y accesibilidad)
- **General Compliance**: Spatial and structural requirements

## 🚀 Deployment

### Streamlit Cloud
```bash
# Deploy to Streamlit Cloud
# Point to: streamlit_app.py
# Add secrets for API keys
```

### Local Development
```bash
streamlit run streamlit_app.py --server.port 8501
```

## 👥 Author

**Raul Adell Segarra**  
UPC-EPSEB Student  
[GitHub](https://github.com/RaulAdSe)

Ready for building code compliance analysis! 🎉