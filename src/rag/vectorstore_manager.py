"""
Vectorstore manager for RAG system.

Handles creation, loading, and management of ChromaDB vectorstore.
"""

from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .embeddings_config import get_embeddings
from .document_loader import load_pdfs


class VectorstoreManager:
    """Manages ChromaDB vectorstore for document retrieval."""
    
    def __init__(self, persist_directory: Path, embeddings_model: str = None):
        """
        Initialize vectorstore manager.
        
        Args:
            persist_directory: Directory to persist vectorstore
            embeddings_model: Optional embeddings model name
        """
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings(embeddings_model)
        self.vectorstore: Optional[Chroma] = None
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    def create_from_pdfs(
        self, 
        pdf_dir: Path, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: List[str] = None
    ) -> None:
        """
        Create vectorstore from PDF documents.
        
        Args:
            pdf_dir: Directory containing PDF files
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks in characters
            separators: List of separators for text splitting
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        print("Loading PDF documents...")
        documents = load_pdfs(pdf_dir)
        
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        print("Creating embeddings and vectorstore...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        print(f"Vectorstore created and saved to: {self.persist_directory}")
    
    def load_existing(self) -> None:
        """
        Load existing vectorstore from disk.
        
        Raises:
            ValueError: If vectorstore doesn't exist or can't be loaded
        """
        if not self.persist_directory.exists():
            raise ValueError(f"Vectorstore directory {self.persist_directory} does not exist")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            print(f"Vectorstore loaded from: {self.persist_directory}")
        except Exception as e:
            raise ValueError(f"Failed to load vectorstore: {e}")
    
    def get_retriever(self, k: int = 3, search_type: str = "similarity"):
        """
        Get retriever for document search.
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity" or "mmr")
            
        Returns:
            Configured retriever
            
        Raises:
            ValueError: If vectorstore is not initialized
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Use create_from_pdfs() or load_existing()")
        
        search_kwargs = {"k": k}
        
        if search_type == "mmr":
            # Maximal Marginal Relevance for diversity
            search_kwargs.update({
                "fetch_k": k * 3,  # Fetch more candidates
                "lambda_mult": 0.5  # Balance relevance vs diversity
            })
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform similarity search directly.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_vectorstore_info(self) -> dict:
        """
        Get information about the vectorstore.
        
        Returns:
            Dictionary with vectorstore information
        """
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "status": "initialized",
                "persist_directory": str(self.persist_directory),
                "document_count": count,
                "embedding_model": self.embeddings.model_name
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
