# RAG System - Pilar 3

This module implements a Retrieval Augmented Generation (RAG) system for querying Spanish building code documents.

## Features

- **Document Loading**: Load PDF documents from the normativa directory
- **Multilingual Embeddings**: Use sentence transformers optimized for Spanish
- **Vector Storage**: ChromaDB for efficient similarity search
- **OpenAI Integration**: GPT models for generating contextual answers
- **Source Citation**: Always cite the source document and page

## Quick Start

### 1. Set up environment

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt
```

### 2. Test the system

```bash
# Run the test script
python scripts/test_rag_system.py
```

### 3. Use in code

```python
from pathlib import Path
from src.rag import VectorstoreManager, create_qa_chain, query

# Initialize
rag = VectorstoreManager(Path("vectorstore/normativa_db"))
rag.create_from_pdfs(Path("data/normativa"))

# Create QA chain
retriever = rag.get_retriever(k=3)
qa_chain = create_qa_chain(retriever)

# Query
result = query(qa_chain, "¿Cuál es el ancho mínimo de una puerta de evacuación?")
print(result["result"])
```

## Components

### Document Loader (`document_loader.py`)
- Loads PDF files from directories
- Extracts text and metadata
- Handles errors gracefully

### Embeddings (`embeddings_config.py`)
- Multilingual sentence transformers
- Optimized for Spanish text
- Configurable models (fast, balanced, high-quality)

### Vectorstore Manager (`vectorstore_manager.py`)
- ChromaDB integration
- Document chunking and indexing
- Similarity search and retrieval

### QA Chain (`qa_chain.py`)
- OpenAI LLM integration
- Spanish-optimized prompts
- Source citation and formatting

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI LLM

### Model Options
- **Default**: `gpt-3.5-turbo` (cost-effective)
- **High Quality**: `gpt-4` (better accuracy)
- **Custom**: Any OpenAI model

### Embedding Models
- **Default**: `paraphrase-multilingual-MiniLM-L12-v2` (balanced)
- **Fast**: `all-MiniLM-L6-v2` (development)
- **High Quality**: `paraphrase-multilingual-mpnet-base-v2` (production)

## Testing

Run unit tests:
```bash
pytest tests/unit/rag/ -v
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Set the environment variable: `export OPENAI_API_KEY="your-key"`

2. **"No PDF files found"**
   - Ensure PDFs are in `data/normativa/` directory
   - Check file extensions are `.pdf`

3. **Slow performance**
   - Use `get_fast_embeddings()` for development
   - Reduce chunk size or k parameter

4. **Poor answers**
   - Increase k parameter for more context
   - Use higher quality embedding model
   - Check if documents contain relevant information

## Architecture

```
PDF Documents → Document Loader → Text Splitter → Embeddings → Vectorstore
                                                                    ↓
User Query → Embeddings → Similarity Search → Retrieved Docs → QA Chain → Answer
```

## Performance Tips

- **Chunk Size**: 1000 characters (default) works well for technical documents
- **Overlap**: 200 characters preserves context between chunks
- **Retrieval**: k=3 provides good balance of context and precision
- **Model**: GPT-3.5-turbo is cost-effective for most queries
