"""
RAG (Retrieval Augmented Generation) module for AEC Compliance Agent.

This module provides functionality to:
- Load and process building code documents (PDFs)
- Create embeddings and vectorstore
- Query documents using OpenAI LLM
- Provide contextual answers about building regulations
"""

from .document_loader import load_pdfs
from .embeddings_config import get_embeddings
from .vectorstore_manager import VectorstoreManager
from .qa_chain import create_qa_chain, query

__all__ = [
    "load_pdfs",
    "get_embeddings", 
    "VectorstoreManager",
    "create_qa_chain",
    "query"
]
