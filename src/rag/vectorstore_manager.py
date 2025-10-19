"""
Vectorstore management for RAG system.

This module handles the creation and management of the ChromaDB vectorstore
for building code document retrieval.
"""

from pathlib import Path
from typing import Optional, List
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .embeddings_config import get_embeddings
from .document_loader import load_pdfs


class VectorstoreManager:
    """Manages RAG vectorstore creation and loading."""
    
    def __init__(self, persist_directory: Path):
        """
        Initialize vectorstore manager.
        
        Args:
            persist_directory: Directory to persist the vectorstore
        """
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vectorstore: Optional[Chroma] = None
    
    def create_from_pdfs(
        self,
        pdf_dir: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Create vectorstore from PDFs.
        
        Args:
            pdf_dir: Directory with PDF files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        print("=" * 60)
        print("CREATING VECTORSTORE")
        print("=" * 60)
        
        # 1. Load documents
        print("\n1. Loading PDFs...")
        documents = load_pdfs(pdf_dir)
        
        if not documents:
            raise ValueError("No documents loaded!")
        
        # 2. Split into chunks
        print("\n2. Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # 3. Create vectorstore
        print("\n3. Creating embeddings and vectorstore...")
        print("(This may take a few minutes...)")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        print(f"\n✅ Vectorstore created: {self.persist_directory}")
        print("=" * 60)
    
    def load_existing(self):
        """Load existing vectorstore."""
        if not self.persist_directory.exists():
            raise ValueError(f"Vectorstore not found: {self.persist_directory}")
        
        print(f"Loading vectorstore from: {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings
        )
        print("✅ Vectorstore loaded")
    
    def get_retriever(self, k: int = 3):
        """
        Get retriever for searches.
        
        Args:
            k: Number of documents to retrieve
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def query_simple(self, question: str, k: int = 3):
        """
        Simple query without LLM (just retrieval).
        
        Args:
            question: Query string
            k: Number of results
        
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        docs = self.vectorstore.similarity_search(question, k=k)
        return docs
    
    def query(self, question: str) -> dict:
        """
        Query the vectorstore and return formatted results.
        
        Args:
            question: Query string
            
        Returns:
            Dictionary with answer and sources
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        # Get relevant documents
        docs = self.vectorstore.similarity_search(question, k=3)
        
        # Format response
        answer = "Based on the building codes, here is the relevant information:\n\n"
        sources = []
        
        for i, doc in enumerate(docs, 1):
            answer += f"{i}. {doc.page_content[:200]}...\n\n"
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:500]
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": 0.8,
            "regulation_references": ["CTE DB-SI", "CTE DB-SUA"]
        }
