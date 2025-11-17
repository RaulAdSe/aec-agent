# Streamlit Development Strategy

## Branch Structure

This document outlines the development strategy for the AEC Compliance Agent Streamlit web interface.

### Main Development Branches

```
main-2.0
└── streamlit-app (base branch)
    ├── ifc-upload-feature
    └── pdf-upload-rag
```

#### Branch Descriptions

1. **streamlit-app** (Base Branch)
   - Main Streamlit application structure
   - UI/UX components and navigation
   - Integration layer for features
   - Base functionality and routing

2. **ifc-upload-feature** (Feature Branch)
   - IFC file upload capability
   - Automatic conversion to JSON format
   - Storage system for agent access
   - IFC data visualization components

3. **pdf-upload-rag** (Feature Branch)
   - PDF document upload functionality
   - RAG system integration
   - Document processing pipeline
   - Knowledge base management

## Development Principles

### Local-First Development
- All development and testing done locally
- No external dependencies for core functionality
- Self-contained deployment capability

### Simple Architecture
- Modular design with clear separation of concerns
- Minimal external dependencies
- Easy to understand and maintain

### Testing Strategy
- Unit tests for core functionality
- Integration tests for file processing
- UI testing for Streamlit components
- All tests documented and automated

### Documentation Requirements
- All features must be documented in `docs/`
- Code comments for complex functionality
- User guides for Streamlit interface
- API documentation for integration points

## File Structure

```
aec-compliance-agent/
├── app.py                     # Main Streamlit application
├── requirements-streamlit.txt # Streamlit dependencies
├── pages/                     # Streamlit pages
│   ├── ifc_analysis.py       # IFC analysis page
│   ├── document_upload.py    # Document upload page
│   └── query_assistant.py    # Query interface page
├── components/               # Reusable UI components
├── services/                # Backend services
│   ├── ifc_processor.py     # IFC file processing
│   ├── pdf_processor.py     # PDF document processing
│   └── agent_interface.py   # Agent communication
└── tests/                   # Test suite
    ├── test_ifc_upload.py
    ├── test_pdf_upload.py
    └── test_streamlit_ui.py
```

## Getting Started

### Installation

1. Install Streamlit dependencies:
   ```bash
   pip install -r requirements-streamlit.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Development Workflow

1. Start from the appropriate branch:
   - For UI/base features: `streamlit-app`
   - For IFC features: `ifc-upload-feature`
   - For PDF/RAG features: `pdf-upload-rag`

2. Implement and test locally

3. Document changes in `docs/`

4. Create pull request to merge back to `streamlit-app`

## Integration Points

### IFC Processing
- Leverages existing IFC extraction scripts
- Stores processed data in agent-accessible format
- Provides real-time processing status

### RAG System
- Integrates with existing RAG library
- Processes uploaded PDFs for knowledge base
- Enables document-based query responses

### Agent Communication
- Interfaces with the compliance agent
- Handles query routing and response formatting
- Manages session state and context

## Testing Requirements

All branches must include:
- Unit tests with >80% coverage
- Integration tests for file processing
- UI testing for user workflows
- Performance testing for file uploads
- Documentation of test procedures

## Deployment Considerations

- Local deployment capability
- Docker containerization support
- Environment configuration management
- Security considerations for file uploads