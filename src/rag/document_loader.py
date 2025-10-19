"""
Document loader for RAG system.

Handles loading PDF documents from the normativa directory.
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_pdfs(pdf_dir: Path) -> List[Document]:
    """
    Load all PDFs from a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        
    Returns:
        List of loaded documents with metadata
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no PDFs found in directory
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory {pdf_dir} does not exist")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    
    documents = []
    
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            
            # Add source metadata to each document
            for doc in docs:
                doc.metadata.update({
                    "source": pdf_path.name,
                    "source_path": str(pdf_path)
                })
            
            documents.extend(docs)
            print(f"  - Loaded {len(docs)} pages")
            
        except Exception as e:
            print(f"  - Error loading {pdf_path.name}: {e}")
            continue
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of documents from the PDF
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"File {pdf_path} does not exist")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"File {pdf_path} is not a PDF")
    
    print(f"Loading: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    # Add source metadata
    for doc in docs:
        doc.metadata.update({
            "source": pdf_path.name,
            "source_path": str(pdf_path)
        })
    
    print(f"Loaded {len(docs)} pages")
    return docs
