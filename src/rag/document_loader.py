"""
Document loader for RAG system.

This module handles loading PDF documents for the vectorstore.
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_pdfs(pdf_dir: Path) -> List[Document]:
    """
    Load all PDFs from directory.
    
    Args:
        pdf_dir: Directory containing PDF files
    
    Returns:
        List of Document objects
    """
    documents = []
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)
            print(f"  ✅ Loaded {len(docs)} pages")
        except Exception as e:
            print(f"  ❌ Error loading {pdf_path.name}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents
