"""
Compliance Search Tool - Simple interface for agents to query compliance documents.

This provides a single, simple function for agents to search the compliance knowledge base.
It automatically handles initialization and provides clean results.
"""

from typing import Dict, Any

# Import the document retrieval toolkit
from .document_retrieval_toolkit import (
    initialize_gemini_client,
    search_documents,
    get_store_info
)

# Configuration  
KNOWLEDGE_BASE_STORE = "compliance_knowledge_base"

# Global state to track if client is initialized
_client_initialized = False


def _format_citations(citations: list) -> list:
    """
    Format citations for better readability and compliance reporting.
    
    Args:
        citations: Raw citation data from Gemini
        
    Returns:
        List of formatted citation objects
    """
    formatted = []
    
    for i, citation in enumerate(citations, 1):
        # Extract source information
        source = citation.get("source", "Unknown source")
        title = citation.get("title", None)
        uri = citation.get("uri", None)
        cited_text = citation.get("cited_text", None)
        confidence = citation.get("confidence", None)
        
        # Format the citation
        formatted_citation = {
            "id": i,
            "source": source,
            "display_name": title or _extract_document_name(source),
            "cited_text": cited_text,
            "confidence": confidence,
            "formatted_reference": _create_citation_reference(i, source, title, uri)
        }
        
        formatted.append(formatted_citation)
    
    return formatted


def _extract_document_name(source_path: str) -> str:
    """Extract a readable document name from source path."""
    if not source_path:
        return "Unknown Document"
    
    # Extract filename from path-like strings
    if '/' in source_path:
        name = source_path.split('/')[-1]
    elif '\\' in source_path:
        name = source_path.split('\\')[-1]
    else:
        name = source_path
    
    # Remove file extension for cleaner display
    if '.' in name:
        name = '.'.join(name.split('.')[:-1])
    
    # Clean up common patterns
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Capitalize
    return ' '.join(word.capitalize() for word in name.split())


def _create_citation_reference(citation_id: int, source: str, title: str = None, uri: str = None) -> str:
    """Create a formatted citation reference string."""
    display_name = title or _extract_document_name(source)
    
    # Create standard citation format
    citation_ref = f"[{citation_id}] {display_name}"
    
    if uri and uri != source:
        citation_ref += f" ({uri})"
    elif source and source != display_name:
        citation_ref += f" (Source: {source})"
    
    return citation_ref

def search_compliance_docs(query: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Search compliance documents for relevant information.
    
    Main function for agents to query compliance documentation.
    Automatically handles initialization and provides structured results.
    
    Args:
        query: Natural language question about compliance requirements
        max_results: Maximum number of results (not used in current Gemini implementation)
        
    Returns:
        Dict with search results:
        - status: "success" | "error" | "no_documents"
        - answer: Natural language answer
        - citations: Source citations if available
        - documents_searched: Number of documents in knowledge base
        - query: Original query
        - logs: Relevant logs/messages
    """
    global _client_initialized
    logs = []
    
    # Auto-initialize client if needed
    if not _client_initialized:
        init_result = initialize_gemini_client()
        if init_result["status"] != "success":
            return {
                "status": "error",
                "answer": "",
                "citations": [],
                "documents_searched": 0,
                "query": query,
                "logs": [f"Failed to initialize compliance search: {init_result['logs'][0]}"]
            }
        _client_initialized = True
        logs.append("Compliance search initialized")
    
    # Check if knowledge base has documents
    store_result = get_store_info(KNOWLEDGE_BASE_STORE)
    if store_result["status"] == "not_found":
        return {
            "status": "no_documents",
            "answer": "No compliance knowledge base found. Please run './kb sync' or 'python3 bin/kb-manager sync' to create and populate the knowledge base.",
            "citations": [],
            "formatted_citations": [],
            "source_documents": [],
            "citation_count": 0,
            "documents_searched": 0,
            "query": query,
            "logs": ["Knowledge base not found"]
        }
    
    document_count = store_result["data"].get("document_count", 0) if store_result["status"] == "success" else 0
    if document_count == 0:
        return {
            "status": "no_documents", 
            "answer": "No documents in compliance knowledge base. Please add compliance documents to data/doc/ folder and run './kb sync'.",
            "citations": [],
            "formatted_citations": [],
            "source_documents": [],
            "citation_count": 0,
            "documents_searched": 0,
            "query": query,
            "logs": ["No documents in knowledge base"]
        }
    
    # Perform the search
    search_result = search_documents(KNOWLEDGE_BASE_STORE, query, max_results)
    
    if search_result["status"] == "success":
        data = search_result["data"]
        
        # Format citations for better readability
        formatted_citations = _format_citations(data.get("citations", []))
        
        return {
            "status": "success",
            "answer": data.get("content", "No answer found"),
            "citations": data.get("citations", []),
            "formatted_citations": formatted_citations,
            "source_documents": data.get("source_documents", []),
            "citation_count": data.get("citation_count", 0),
            "documents_searched": data.get("documents_in_store", document_count),
            "query": query,
            "logs": logs + search_result.get("logs", [])
        }
    else:
        return {
            "status": "error",
            "answer": "",
            "citations": [],
            "formatted_citations": [],
            "source_documents": [],
            "citation_count": 0,
            "documents_searched": document_count,
            "query": query,
            "logs": logs + search_result.get("logs", ["Search failed"])
        }


def check_knowledge_base_status() -> Dict[str, Any]:
    """
    Check the status of the compliance knowledge base.
    
    Returns:
        Dict with status: "ready" | "empty" | "not_found", document_count, message, logs
    """
    global _client_initialized
    logs = []
    
    # Auto-initialize if needed
    if not _client_initialized:
        init_result = initialize_gemini_client()
        if init_result["status"] != "success":
            return {
                "status": "error",
                "document_count": 0,
                "message": f"Cannot check status: {init_result['logs'][0]}",
                "logs": logs
            }
        _client_initialized = True
        logs.append("Client initialized for status check")
    
    # Check store status
    store_result = get_store_info(KNOWLEDGE_BASE_STORE)
    
    if store_result["status"] == "not_found":
        return {
            "status": "not_found",
            "document_count": 0,
            "message": "Knowledge base not found. Run 'python3 manage_knowledge_base.py sync' to create it.",
            "logs": logs
        }
    
    if store_result["status"] == "success":
        document_count = store_result["data"].get("document_count", 0)
        
        if document_count == 0:
            return {
                "status": "empty",
                "document_count": 0,
                "message": "Knowledge base exists but is empty. Add documents to data/doc/ and sync.",
                "logs": logs
            }
        else:
            return {
                "status": "ready",
                "document_count": document_count,
                "message": f"Knowledge base ready with {document_count} documents",
                "logs": logs
            }
    
    return {
        "status": "error",
        "document_count": 0,
        "message": "Could not determine knowledge base status",
        "logs": logs + store_result.get("logs", [])
    }