# AEC Compliance Agent

Simple, powerful agent for building compliance analysis with access to all your tools.

## What It Does

One clean agent that can:
- 🏗️ **Load and analyze IFC building data**
- 📐 **Query elements, calculate metrics, find relationships**  
- ✅ **Validate compliance rules**
- 📚 **Search compliance documents**
- 💬 **Answer questions in natural language**

## Quick Start

### 1. Install Dependencies
```bash
pip install langchain langchain-openai langserve uvicorn fastapi
```

### 2. Set API Key
```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Use the Agent
```bash
# Interactive chat
python -m aec_agent.main chat

# Single query
python -m aec_agent.main query "Load building data from data/out/FM-ARC_v2.json and analyze it"

# API server
python -m aec_agent.main serve

# File analysis
python -m aec_agent.main analyze data/out/FM-ARC_v2.json --analysis-type fire_safety
```

## Available Tools

The agent has 9 tools:
- `load_building_data` - Load IFC JSON files
- `get_all_elements` - Get elements by type  
- `get_element_properties` - Get element properties
- `query_elements` - Filter elements with criteria
- `calculate_metrics` - Perform calculations
- `find_related_elements` - Find relationships
- `validate_compliance_rule` - Check compliance
- `search_compliance_documents` - Search regulations
- `check_knowledge_base_status` - Check knowledge base

## Example Queries

```bash
"Load building data from data/out/FM-ARC_v2.json and tell me how many spaces there are"

"What are the fire safety requirements for emergency exits?"

"Find all doors wider than 0.9m and check if they meet accessibility standards"

"Calculate the total floor area for level E00_OKRD"
```

## Files

```
aec_agent/
├── agent.py              # Main compliance agent
├── main.py               # CLI interface  
├── tools/                # Building data & compliance tools
├── services/             # API services
└── app.py                # Streamlit web interface
```

## Architecture

Simple and clean:
```
ComplianceAgent
├── LangChain Agent (GPT-4)
├── 9 Tools (building data + compliance)
└── Natural Language Interface
```

The agent is integrated directly into your existing `aec_agent/` structure - no nested complexity, just one agent with access to all your tools.