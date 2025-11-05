# 🏗️ AEC Compliance Agent 2.0

## Clean Architecture with TOON Optimization

A fresh, minimal implementation of an autonomous AI agent for building code compliance verification. Optimized for LLM token efficiency using TOON (Token-Oriented Object Notation) format.

## 🎯 Project Overview

**Status**: Clean restart with minimal scaffold  
**Architecture**: Token-optimized agent with TOON format support  
**Purpose**: Building code compliance verification for Spanish regulations

### Key Features

- **🎒 TOON Format**: 30-60% token reduction for LLM efficiency
- **🤖 Clean Agent**: Minimal, focused compliance verification
- **📐 Basic Geometry**: Essential spatial calculations
- **🔧 Modular Design**: Clean separation of concerns

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google API key (for Gemini LLM)

### Installation

```bash
# Clone repository
git clone https://github.com/RaulAdSe/aec-agent.git
cd aec-agent

# Switch to clean version
git checkout main-2.0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (create .env file)
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Basic Usage

```python
from src.agent import ComplianceAgent

# Initialize agent
agent = ComplianceAgent()

# Sample building data
building_data = {
    "metadata": {"project_name": "Test Building", "total_area": 500.0},
    "rooms": [{"id": "R001", "area": 25.0, "use": "office"}],
    "doors": [{"id": "D001", "width": 0.9, "fire_rated": True}]
}

# Run compliance analysis
result = agent.analyze_building(building_data)
print(result)
```

## 📁 Clean Project Structure

```
aec-agent/
├── src/                    # Clean source code
│   ├── agent/              # Core agent implementation
│   │   ├── core.py         # Main agent class
│   │   ├── tools.py        # Compliance tools
│   │   └── memory.py       # Agent memory system
│   └── utils/              # Utilities
│       └── toon_converter.py  # TOON format support
├── data/                   # Minimal data
│   ├── blueprints/         # IFC files
│   └── archive/            # Archived old data
├── tests/                  # Clean test structure
└── requirements.txt        # Dependencies with TOON support
```

## 🛠️ Technology Stack

- **LLM**: Google Gemini (gemini-2.0-flash-exp)
- **Agent Framework**: LangChain
- **Data Format**: TOON (Token-Oriented Object Notation)
- **Validation**: Pydantic
- **Testing**: Minimal pytest structure

## 🎒 TOON Format Benefits

TOON (Token-Oriented Object Notation) provides:
- 30-60% token reduction vs JSON
- Human-readable format
- LLM-optimized structure
- Maintains data integrity

### Example Comparison

**JSON (verbose):**
```json
{
  "rooms": [
    {"id": "R001", "area": 25.0, "type": "office"},
    {"id": "R002", "area": 30.0, "type": "office"}
  ]
}
```

**TOON (compact):**
```
rooms[2]{id,area,type}:
R001,25.0,office
R002,30.0,office
```

## 🔧 Development

This is a clean restart focusing on:
- Minimal viable functionality
- Token-optimized data exchange
- Clean, maintainable code
- Essential compliance features only

### Adding Features

1. Keep additions minimal and focused
2. Use TOON format for data exchange
3. Follow clean architecture principles
4. Test with real building data

## 📚 Documentation

- Architecture documentation coming soon
- API reference will be minimal and focused
- Focus on practical usage examples

## 👥 Author

**Raul Adell Segarra**  
UPC-EPSEB Student  
[GitHub](https://github.com/RaulAdSe)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

*Clean architecture for the AEC industry - TOON optimized* 🎒