# AEC Compliance Agent 🏗️

**Building Code Compliance Verification System** with Streamlit deployment and LangSmith integration.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and install
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

### 3. Use CLI (Optional)
```bash
python -m src.main status
python -m src.main analyze data/sample.json
```

## 🔑 API Keys Required

| Service | Purpose | Required |
|---------|---------|----------|
| **OpenAI** | LLM for compliance analysis | ✅ Yes |
| **LangSmith** | Tracing & monitoring | ✅ Recommended |

## 🎯 Features

### 🌐 Streamlit Deployment
- **Web Interface**: User-friendly building analysis
- **File Upload**: JSON building data support
- **Real-time Analysis**: Instant compliance checking
- **Results Export**: Download analysis reports

### 📊 LangSmith Integration
- **Tracing**: Track all agent operations
- **Monitoring**: Performance metrics
- **Debugging**: Step-by-step analysis
- **LangChain Studio**: Visual workflow analysis

### 🏗️ Professional Architecture
- **Modular Design**: Scalable agent system
- **Clean Separation**: Core/Agents/Tools/Services
- **Extensible**: Easy to add new compliance checks

## 📁 Architecture

```
src/
├── streamlit_app.py          # 🌐 Streamlit deployment
├── main.py                   # 🖥️ CLI interface
├── core/                     # ⚙️ Framework
│   ├── config.py             # Configuration + LangSmith
│   ├── logger.py             # Logging system
│   └── registry.py           # Agent registry
├── agents/                   # 🤖 Agent implementations
│   └── compliance_agent/     # AEC compliance agent
├── memory/                   # 🧠 Memory systems
├── tools/                    # 🔧 Analysis tools
├── services/                 # 🌍 External integrations
└── utils/                    # 🛠️ Utilities
```

## 🛠️ Development

### Add New Tools
```python
# src/tools/compliance_toolkit.py
def _your_new_tool(self, input_params: str) -> str:
    # Your compliance logic here
    return "Analysis result"
```

### Add New Agents
```bash
mkdir src/agents/your_agent
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