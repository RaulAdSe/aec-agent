"""
PDF RAG Manager - Interface for Streamlit app to manage PDF documents and RAG queries
Integrates with the existing Gemini File Search knowledge base system.
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from aec_agent.tools.document_retrieval_toolkit import (
    initialize_gemini_client,
    create_document_store,
    upload_documents,
    get_store_info,
    list_uploaded_documents
)
from aec_agent.tools.compliance_search import search_compliance_docs


class PDFRAGManager:
    """Manages PDF documents and RAG functionality for the Streamlit app."""
    
    def __init__(self, store_name: str = "compliance_knowledge_base"):
        """Initialize with document store name."""
        self.store_name = store_name
        self.store_description = "Building compliance documents uploaded via Streamlit"
        self._client_initialized = False
    
    def _ensure_initialized(self) -> Dict[str, Any]:
        """Ensure Gemini client is initialized."""
        if self._client_initialized:
            return {"status": "success"}
        
        init_result = initialize_gemini_client()
        if init_result["status"] == "success":
            self._client_initialized = True
            
            # Also ensure store exists
            store_result = create_document_store(self.store_name, self.store_description)
            if store_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Failed to create document store: {store_result['logs']}"
                }
        
        return init_result
    
    def upload_pdf_from_streamlit(self, uploaded_file, file_content: bytes) -> Dict[str, Any]:
        """
        Upload a PDF file from Streamlit to the RAG knowledge base.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            file_content: Raw bytes content of the file
            
        Returns:
            Dict with upload status and metadata
        """
        # Initialize client if needed
        init_result = self._ensure_initialized()
        if init_result["status"] != "success":
            return {
                "status": "error", 
                "message": f"Failed to initialize RAG system: {init_result.get('logs', ['Unknown error'])}",
                "data": None
            }
        
        try:
            # Save file temporarily for upload
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', prefix=f"streamlit_{uploaded_file.name}_") as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Check if file is already uploaded to avoid duplicates
            uploaded_docs = self._get_uploaded_documents()
            if uploaded_file.name in uploaded_docs:
                # Cleanup temp file
                os.unlink(temp_file_path)
                return {
                    "status": "already_exists",
                    "message": f"Document '{uploaded_file.name}' already exists in knowledge base",
                    "data": {
                        "file_name": uploaded_file.name,
                        "file_size": uploaded_file.size,
                        "already_uploaded": True
                    }
                }
            
            # Upload to Gemini File Search
            upload_result = upload_documents(
                self.store_name, 
                [temp_file_path],
                ["legal_document"]  # Document type for compliance docs
            )
            
            # Cleanup temporary file
            os.unlink(temp_file_path)
            
            if upload_result["status"] == "success":
                # Extract file metadata
                file_info = {
                    "file_name": uploaded_file.name,
                    "file_size": uploaded_file.size,
                    "file_type": uploaded_file.type,
                    "upload_time": "now",
                    "document_type": "legal_document",
                    "processed_for_search": True
                }
                
                return {
                    "status": "success",
                    "message": f"Successfully uploaded '{uploaded_file.name}' to knowledge base",
                    "data": file_info
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to upload '{uploaded_file.name}': {upload_result.get('logs', ['Unknown error'])}",
                    "data": None
                }
                
        except Exception as e:
            # Cleanup temp file if it exists
            if 'temp_file_path' in locals():
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            return {
                "status": "error",
                "message": f"Error processing upload: {str(e)}",
                "data": None
            }
    
    def _get_uploaded_documents(self) -> set:
        """Get set of already uploaded document names."""
        result = list_uploaded_documents(self.store_name)
        
        if result["status"] == "success":
            uploaded_names = set()
            for doc in result["data"]["documents"]:
                display_name = doc.get("display_name", "")
                if display_name:
                    uploaded_names.add(display_name)
            return uploaded_names
        
        return set()
    
    def get_knowledge_base_summary(self) -> Dict[str, Any]:
        """Get summary of the knowledge base contents."""
        init_result = self._ensure_initialized()
        if init_result["status"] != "success":
            return {
                "status": "error",
                "document_count": 0,
                "documents": [],
                "message": "RAG system not initialized"
            }
        
        # Get store info
        store_result = get_store_info(self.store_name)
        
        if store_result["status"] == "success":
            store_data = store_result["data"]
            
            # Get detailed document list
            docs_result = list_uploaded_documents(self.store_name)
            documents = []
            if docs_result["status"] == "success":
                documents = docs_result["data"]["documents"]
            
            return {
                "status": "ready",
                "document_count": store_data.get("document_count", len(documents)),
                "documents": documents,
                "store_name": self.store_name,
                "message": f"Knowledge base ready with {len(documents)} documents"
            }
        elif store_result["status"] == "not_found":
            return {
                "status": "empty",
                "document_count": 0,
                "documents": [],
                "message": "Knowledge base not found - upload documents to get started"
            }
        else:
            return {
                "status": "error", 
                "document_count": 0,
                "documents": [],
                "message": f"Error accessing knowledge base: {store_result.get('logs', ['Unknown error'])}"
            }
    
    def search_legal_documents(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the legal documents knowledge base.
        
        Args:
            query: Natural language query
            max_results: Maximum results to return
            
        Returns:
            Dict with search results and citations
        """
        init_result = self._ensure_initialized()
        if init_result["status"] != "success":
            return {
                "status": "error",
                "answer": "RAG system not available",
                "citations": [],
                "query": query,
                "message": "RAG system initialization failed"
            }
        
        # Use the compliance search function
        search_result = search_compliance_docs(query, max_results)
        
        # Add additional metadata for Streamlit display
        search_result["search_performed"] = True
        search_result["knowledge_base"] = self.store_name
        
        return search_result
    
    def is_ready(self) -> bool:
        """Check if the RAG system is ready for queries."""
        summary = self.get_knowledge_base_summary()
        return summary["status"] == "ready" and summary["document_count"] > 0


# Utility functions for direct use in Streamlit
def process_uploaded_pdf(uploaded_file, file_content: bytes) -> Dict[str, Any]:
    """Quick function to process an uploaded PDF for RAG."""
    manager = PDFRAGManager()
    return manager.upload_pdf_from_streamlit(uploaded_file, file_content)


def query_legal_knowledge_base(query: str) -> Dict[str, Any]:
    """Quick function to query the legal knowledge base."""
    manager = PDFRAGManager()
    return manager.search_legal_documents(query)


def get_rag_status() -> Dict[str, Any]:
    """Quick function to get RAG system status."""
    manager = PDFRAGManager()
    return manager.get_knowledge_base_summary()