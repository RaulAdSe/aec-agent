# AEC Compliance Agent - Project Structure

## 🎯 **Clean, Logical Organization**

```
aec-compliance-agent/
├── aec_agent/                   # 📦 Main Package
│   ├── __init__.py
│   ├── tools/                   # 🔧 Agent Tools
│   │   ├── building_data_toolkit.py      # IFC & building analysis
│   │   ├── document_retrieval_toolkit.py # Gemini File Search
│   │   └── compliance_search.py          # Agent-friendly search
│   ├── agents/                  # 🤖 Agent Implementations
│   │   ├── base_agent.py
│   │   └── compliance_agent/
│   ├── core/                    # ⚙️ Core Framework
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── registry.py
│   ├── memory/                  # 🧠 Memory Systems
│   ├── services/                # 🌐 External Services
│   └── utils/                   # 🛠️ Utilities
├── bin/                         # 🎯 Executable Scripts
│   └── kb-manager               # Knowledge base management
├── data/                        # 📄 Documents & Data
│   ├── doc/                     # Your compliance documents
│   ├── in/                      # Input files (IFC, etc.)
│   └── out/                     # Processed outputs
├── tests/                       # 🧪 Test Suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test fixtures
├── examples/                    # 📚 Usage Examples
│   ├── debug_upload.py
│   ├── quick_sync.py
│   └── upload_docs.py
├── docs/                        # 📖 Documentation
│   └── PROJECT_STRUCTURE.md    # This file
├── kb                           # 🎯 Convenience script
├── README.md                    # Main documentation
├── requirements.txt             # Dependencies
└── pytest.ini                  # Test configuration
```

## 🎯 **Design Philosophy**

### Single-Purpose Directories
- **`aec_agent/`** - The main Python package with all core functionality
- **`bin/`** - Executable scripts (like `/usr/bin/`)
- **`examples/`** - Sample usage and debug scripts
- **`docs/`** - All documentation
- **`tests/`** - Complete test suite
- **`data/`** - User documents and data files

### Clean Imports
```python
# Agent code can cleanly import tools
from aec_agent.tools import search_compliance_docs

# Or import specific tools
from aec_agent.tools.building_data_toolkit import load_building_data
```

### Simple Usage
```bash
# Knowledge base management
./kb sync                    # Sync documents
./kb query "door width"      # Search
./kb status                  # Check status

# Direct script access
bin/kb-manager sync
```

## 📦 **Package Structure**

### `aec_agent/` - Main Package
The core Python package containing all functionality:

**`tools/`** - Agent Tools
- `building_data_toolkit.py` - IFC file processing, element analysis, calculations
- `document_retrieval_toolkit.py` - Gemini File Search integration  
- `compliance_search.py` - Simple agent interface for document search

**`agents/`** - Agent Implementations
- `base_agent.py` - Base agent class
- `compliance_agent/` - AEC compliance-specific agent

**`core/`** - Framework Components
- `config.py` - Configuration management
- `logger.py` - Logging system
- `registry.py` - Agent registry

### `bin/` - Executable Scripts
Unix-style executable scripts:

**`kb-manager`** - Main knowledge base management script
- Handles document uploading, syncing, querying
- Used by the `./kb` convenience wrapper

### `examples/` - Usage Examples
Sample scripts showing how to use the system:
- `debug_upload.py` - Debug document upload process
- `quick_sync.py` - Quick test sync script  
- `upload_docs.py` - Basic upload example

## 🚀 **Usage Patterns**

### For End Users
```bash
# Simple commands
./kb sync                    # Sync your documents
./kb query "door width"      # Search compliance docs
```

### For Developers
```python
# Import and use agent tools
from aec_agent.tools import search_compliance_docs, load_building_data

# Search compliance documents  
result = search_compliance_docs("minimum door width requirements")
answer = result["answer"]

# Load and analyze building data
building_data = load_building_data("data/out/building.json")
```

### For Testing
```bash
# Run tests
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests (requires API key)
pytest                      # All tests
```

## ✅ **Benefits**

1. **Clean Separation** - Each directory has a single, clear purpose
2. **Standard Python Package** - `aec_agent/` follows Python packaging conventions
3. **Unix-style Scripts** - `bin/` directory for executable scripts
4. **Easy Imports** - Clean, predictable import paths
5. **Logical Organization** - Examples, docs, tests all properly separated
6. **No Confusion** - No ambiguous "scripts vs src" structure

## 🔄 **Migration from Old Structure**

**Old (Confusing):**
```
├── src/                     # ❌ Generic name
├── scripts/                 # ❌ Separate from main code
│   ├── manage_knowledge_base.py
│   ├── examples/
│   └── utils/
```

**New (Clear):**
```
├── aec_agent/              # ✅ Clear package name
├── bin/                    # ✅ Standard Unix convention
│   └── kb-manager
├── examples/               # ✅ Clear purpose
```

This structure follows Python and Unix conventions, making it immediately familiar to developers! 🎉