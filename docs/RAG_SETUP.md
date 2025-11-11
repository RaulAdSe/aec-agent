# RAG Document Retrieval Setup

## 🎯 **Simple RAG System for Compliance Documents**

This document explains the RAG (Retrieval-Augmented Generation) system for searching compliance documents with citations.

## 📚 **How It Works**

### **1. Document Storage**
- Put PDF/TXT/DOCX files in `data/doc/`
- Run `./kb sync` to upload to Gemini File Search
- Gemini automatically indexes and embeds documents

### **2. Search & Retrieval**
- Query with natural language: `./kb query "door width requirements"`
- Gemini searches semantically across all documents
- Returns answer with source citations

### **3. Citation System**
- Tracks which documents provided information
- Shows specific quoted text from sources
- Provides document names and confidence scores

## 🚀 **Quick Usage**

### **Command Line**
```bash
# Add documents
cp building-code.pdf data/doc/
cp fire-safety.pdf data/doc/

# Sync to knowledge base
./kb sync

# Query with citations
./kb query "minimum door width for accessibility"
```

### **In Code**
```python
from aec_agent.tools import search_compliance_docs

result = search_compliance_docs("fire exit requirements")
print(result["answer"])

# Show citations
for citation in result["formatted_citations"]:
    print(f"Source: {citation['display_name']}")
```

## 🔧 **System Components**

### **Core Files**
- **`bin/kb-manager`** - Main script for document management
- **`aec_agent/tools/document_retrieval_toolkit.py`** - Gemini File Search integration
- **`aec_agent/tools/compliance_search.py`** - Agent-friendly search interface
- **`aec_agent/tools/citation_utils.py`** - Citation formatting utilities

### **Data Flow**
```
Documents (data/doc/) 
    ↓ [./kb sync]
Gemini File Search 
    ↓ [search_compliance_docs()]
Answers + Citations
```

## ⚙️ **Configuration**

### **Required**
- `GEMINI_API_KEY` environment variable
- Internet connection for Gemini API

### **Supported File Types**
- PDF (`.pdf`)
- Text (`.txt`) 
- Word (`.docx`)
- JSON (`.json`)
- Markdown (`.md`)

## 📊 **Response Format**

Every search returns:
```python
{
    "status": "success",
    "answer": "Natural language answer...",
    "citations": [...],           # Raw citation data
    "formatted_citations": [...], # Human-readable citations
    "citation_count": 2,
    "documents_searched": 5
}
```

## 🎯 **Key Features**

✅ **Semantic Search** - Finds relevant info even without exact keywords  
✅ **Automatic Citations** - Shows which documents provided answers  
✅ **Multi-format Support** - PDF, Word, text files  
✅ **Agent-Ready** - Simple API for AI agents  
✅ **Professional Citations** - Proper source tracking for compliance  

## 🔍 **Examples**

### **Fire Safety**
```bash
./kb query "sprinkler system requirements for office buildings"
```

### **Accessibility** 
```bash
./kb query "wheelchair ramp slope requirements"
```

### **Building Codes**
```bash
./kb query "minimum ceiling height for residential spaces"
```

Each query returns answers with citations showing exactly which compliance documents support the information.

## 🛠️ **Troubleshooting**

**No documents:** Add files to `data/doc/` and run `./kb sync`  
**No citations:** Wait a few seconds after sync for Gemini processing  
**Upload fails:** Check file sizes and internet connection  

The RAG system provides reliable compliance answers with proper source tracking! 🏗️