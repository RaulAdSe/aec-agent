"""Document Retrieval Toolkit - Gemini File Search for compliance document retrieval.

This toolkit provides document search and retrieval capabilities using Google's Gemini File Search API.
Designed to work alongside the building_data_toolkit for comprehensive AEC compliance checking.

The toolkit manages document stores, uploads, and provides semantic search capabilities
for building codes, regulations, and compliance documentation.

These functions are designed to be called directly by AI agents.

AGENT USAGE FLOW:
1. create_document_store() - Create a document store for specific domain (e.g., "fire_safety", "accessibility")
2. upload_documents() - Upload compliance documents to the store  
3. search_documents() - Semantic search for relevant information
4. get_store_info() - Check store status and contents
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

# Global variables to store Gemini client and document stores
_gemini_client: Optional[Any] = None
_document_stores: Dict[str, Dict] = {}
_logger = logging.getLogger(__name__)

def initialize_gemini_client(api_key: Optional[str] = None) -> Dict:
    """
    Initialize the Gemini client for document retrieval.
    
    Args:
        api_key: Gemini API key (if not provided, tries environment variable GEMINI_API_KEY)
        
    Returns:
        Dict with status and initialization result
        
    Examples:
        # Initialize with API key
        result = initialize_gemini_client("your-api-key")
        
        # Initialize using environment variable
        os.environ["GEMINI_API_KEY"] = "your-api-key"
        result = initialize_gemini_client()
    """
    global _gemini_client
    logs = []
    
    try:
        from google import genai
        from google.genai import types
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            return {
                "status": "error",
                "data": None,
                "logs": ["API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter."]
            }
        
        # Initialize the client
        _gemini_client = genai.Client(api_key=api_key)
        
        logs.append("Gemini client initialized successfully")
        return {
            "status": "success", 
            "data": {"client_ready": True},
            "logs": logs
        }
        
    except ImportError:
        error_msg = "google-genai package not installed. Run: pip install google-genai"
        logs.append(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": logs
        }
    except Exception as e:
        error_msg = f"Error initializing Gemini client: {e}"
        logs.append(error_msg)
        return {
            "status": "error", 
            "data": None,
            "logs": logs
        }


def create_document_store(store_name: str, description: str = "") -> Dict:
    """
    Create a new document store for organizing compliance documents.
    
    Args:
        store_name: Unique name for the document store (e.g., "fire_safety", "accessibility") 
        description: Optional description of the store's purpose
        
    Returns:
        Dict with status, store data, and logs
        
    Examples:
        # Create store for fire safety documents
        create_document_store("fire_safety", "Fire safety codes and regulations")
        
        # Create store for accessibility standards  
        create_document_store("accessibility", "Accessibility compliance standards")
        
        # Create general building codes store
        create_document_store("building_codes", "General building codes and regulations")
    """
    global _gemini_client, _document_stores
    logs = []
    
    if _gemini_client is None:
        return {
            "status": "error",
            "data": None, 
            "logs": ["Gemini client not initialized. Call initialize_gemini_client() first."]
        }
    
    try:
        # Check if store already exists
        if store_name in _document_stores:
            logs.append(f"Document store '{store_name}' already exists")
            return {
                "status": "success",
                "data": _document_stores[store_name],
                "logs": logs
            }
        
        # Create the file search store
        file_search_store = _gemini_client.file_search_stores.create(
            config={'display_name': f'aec-compliance-{store_name}'}
        )
        
        # Store the store info locally
        store_info = {
            "name": store_name,
            "description": description,
            "gemini_store_name": file_search_store.name,
            "gemini_store_id": file_search_store.name.split('/')[-1],
            "created_at": str(file_search_store.create_time) if hasattr(file_search_store, 'create_time') else "unknown",
            "document_count": 0,
            "documents": []
        }
        
        _document_stores[store_name] = store_info
        
        logs.append(f"Created document store '{store_name}' with ID: {store_info['gemini_store_id']}")
        return {
            "status": "success",
            "data": store_info,
            "logs": logs
        }
        
    except Exception as e:
        error_msg = f"Error creating document store '{store_name}': {e}"
        logs.append(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": logs
        }


def upload_documents(store_name: str, file_paths: List[str], document_types: Optional[List[str]] = None) -> Dict:
    """
    Upload documents to a document store for retrieval.
    
    Args:
        store_name: Name of the document store
        file_paths: List of file paths to upload
        document_types: Optional list of document types (e.g., ["building_code", "fire_regulation"])
        
    Returns:
        Dict with upload results and status
        
    Examples:
        # Upload fire safety documents
        upload_documents("fire_safety", [
            "docs/fire_code_2023.pdf",
            "docs/sprinkler_requirements.pdf"
        ], ["fire_code", "sprinkler_regulation"])
        
        # Upload accessibility standards
        upload_documents("accessibility", [
            "docs/ada_guidelines.pdf",
            "docs/barrier_free_design.pdf"  
        ])
    """
    global _gemini_client, _document_stores
    logs = []
    
    if _gemini_client is None:
        return {
            "status": "error",
            "data": None,
            "logs": ["Gemini client not initialized. Call initialize_gemini_client() first."]
        }
    
    if store_name not in _document_stores:
        return {
            "status": "error",
            "data": None,
            "logs": [f"Document store '{store_name}' not found. Create it first with create_document_store()."]
        }
    
    store_info = _document_stores[store_name]
    upload_results = []
    successful_uploads = 0
    
    for i, file_path in enumerate(file_paths):
        try:
            if not os.path.exists(file_path):
                logs.append(f"File not found: {file_path}")
                upload_results.append({
                    "file_path": file_path,
                    "status": "error",
                    "message": "File not found"
                })
                continue
            
            # Upload file to the store
            operation = _gemini_client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=store_info["gemini_store_name"]
            )
            
            # Track the uploaded document with enhanced metadata
            doc_type = document_types[i] if document_types and i < len(document_types) else "compliance_document"
            file_path_obj = Path(file_path)
            
            # Extract document metadata
            file_stat = os.stat(file_path)
            doc_info = {
                "file_path": file_path,
                "file_name": file_path_obj.name,
                "file_stem": file_path_obj.stem,  # filename without extension
                "file_extension": file_path_obj.suffix.lower(),
                "file_size": file_stat.st_size,
                "document_type": doc_type,
                "upload_operation": operation.name if hasattr(operation, 'name') else str(operation),
                "uploaded_at": "now",  # Could use datetime.now() for precise timestamp
                "description": f"{doc_type.replace('_', ' ').title()} document: {file_path_obj.stem}"
            }
            
            # Try to infer document type from filename if not provided
            if doc_type == "compliance_document":
                filename_lower = file_path_obj.name.lower()
                if any(keyword in filename_lower for keyword in ['fire', 'incendio', 'si']):
                    doc_info["document_type"] = "fire_safety"
                elif any(keyword in filename_lower for keyword in ['access', 'sua', 'barrier']):
                    doc_info["document_type"] = "accessibility"
                elif any(keyword in filename_lower for keyword in ['struct', 'se']):
                    doc_info["document_type"] = "structural"
                elif any(keyword in filename_lower for keyword in ['boe', 'cte', 'codigo']):
                    doc_info["document_type"] = "building_code"
            
            store_info["documents"].append(doc_info)
            store_info["document_count"] += 1
            
            upload_results.append({
                "file_path": file_path,
                "status": "success", 
                "message": f"Uploaded as {doc_type}"
            })
            
            successful_uploads += 1
            logs.append(f"Uploaded: {Path(file_path).name}")
            
        except Exception as e:
            error_msg = f"Error uploading {file_path}: {e}"
            logs.append(error_msg)
            upload_results.append({
                "file_path": file_path,
                "status": "error",
                "message": str(e)
            })
    
    # Update store info
    _document_stores[store_name] = store_info
    
    logs.append(f"Upload complete: {successful_uploads}/{len(file_paths)} files successful")
    return {
        "status": "success" if successful_uploads > 0 else "error",
        "data": {
            "store_name": store_name,
            "total_files": len(file_paths),
            "successful_uploads": successful_uploads,
            "upload_results": upload_results,
            "store_info": store_info
        },
        "logs": logs
    }


def search_documents(store_name: str, query: str, max_results: int = 5) -> Dict:
    """
    Search for relevant information in uploaded documents.
    
    This is the core retrieval function - it performs semantic search
    across all documents in the specified store.
    
    Args:
        store_name: Name of the document store to search
        query: Search query (natural language)
        max_results: Maximum number of results to return
        
    Returns:
        Dict with search results and relevant document passages
        
    Examples:
        # Search fire safety requirements
        search_documents("fire_safety", "minimum door width for emergency exits")
        
        # Search accessibility requirements  
        search_documents("accessibility", "wheelchair ramp slope requirements", max_results=3)
        
        # Search building codes
        search_documents("building_codes", "minimum ceiling height for residential spaces")
    """
    global _gemini_client, _document_stores
    logs = []
    
    if _gemini_client is None:
        return {
            "status": "error",
            "data": None,
            "logs": ["Gemini client not initialized. Call initialize_gemini_client() first."]
        }
    
    if store_name not in _document_stores:
        return {
            "status": "error", 
            "data": None,
            "logs": [f"Document store '{store_name}' not found. Create it first with create_document_store()."]
        }
    
    store_info = _document_stores[store_name]
    
    if store_info["document_count"] == 0:
        return {
            "status": "warning",
            "data": {
                "query": query,
                "results": [],
                "total_results": 0
            },
            "logs": [f"No documents in store '{store_name}'. Upload documents first."]
        }
    
    try:
        from google.genai import types
        
        # Perform the search using Gemini File Search  
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Based on the uploaded documents, please provide relevant information about: {query}",
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_info["gemini_store_name"]]
                        )
                    )
                ]
            )
        )
        
        # Extract the response content
        if hasattr(response, 'text') and response.text:
            content = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            content = response.candidates[0].content.parts[0].text if response.candidates[0].content.parts else ""
        else:
            content = str(response)
        
        # Extract detailed citations with source information
        citations = []
        source_documents = []
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Extract citation metadata from Gemini response
            if hasattr(candidate, 'citation_metadata') and candidate.citation_metadata:
                for citation in candidate.citation_metadata.citations:
                    # Extract basic citation info
                    citation_info = {
                        "start_index": getattr(citation, 'start_index', None),
                        "end_index": getattr(citation, 'end_index', None),
                        "source": getattr(citation, 'source', None),
                        "confidence": getattr(citation, 'confidence', None)
                    }
                    
                    # Try to extract more detailed source information
                    if hasattr(citation, 'uri') and citation.uri:
                        citation_info["uri"] = citation.uri
                    
                    if hasattr(citation, 'title') and citation.title:
                        citation_info["title"] = citation.title
                    
                    # Extract the cited text portion
                    if citation_info["start_index"] and citation_info["end_index"] and content:
                        try:
                            start = citation_info["start_index"]
                            end = citation_info["end_index"]
                            citation_info["cited_text"] = content[start:end]
                        except:
                            citation_info["cited_text"] = None
                    
                    citations.append(citation_info)
            
            # Also check for grounding metadata (alternative citation source)
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                for source in candidate.grounding_metadata.grounding_chunks:
                    source_info = {
                        "web_source": getattr(source, 'web', None),
                        "retrieved_context": getattr(source, 'retrieved_context', None)
                    }
                    if source_info["web_source"] or source_info["retrieved_context"]:
                        source_documents.append(source_info)
        
        # Format results with enhanced citation information
        search_results = {
            "query": query,
            "content": content,
            "citations": citations,
            "source_documents": source_documents,
            "total_results": 1,  # Gemini returns consolidated response
            "store_searched": store_name,
            "documents_in_store": store_info["document_count"],
            "citation_count": len(citations)
        }
        
        logs.append(f"Search completed for query: '{query}' in store '{store_name}'")
        logs.append(f"Found information from {store_info['document_count']} documents")
        
        return {
            "status": "success",
            "data": search_results,
            "logs": logs
        }
        
    except Exception as e:
        error_msg = f"Error searching documents in store '{store_name}': {e}"
        logs.append(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": logs
        }


def get_store_info(store_name: Optional[str] = None) -> Dict:
    """
    Get information about document stores.
    
    Args:
        store_name: Specific store to get info for (if None, returns all stores)
        
    Returns:
        Dict with store information
        
    Examples:
        # Get info for specific store
        get_store_info("fire_safety")
        
        # Get info for all stores
        get_store_info()
    """
    global _document_stores
    logs = []
    
    if store_name is None:
        # Return all stores
        logs.append(f"Retrieved information for {len(_document_stores)} document stores")
        return {
            "status": "success",
            "data": {
                "total_stores": len(_document_stores),
                "stores": _document_stores
            },
            "logs": logs
        }
    else:
        # Return specific store
        if store_name not in _document_stores:
            return {
                "status": "not_found",
                "data": None,
                "logs": [f"Document store '{store_name}' not found"]
            }
        
        store_info = _document_stores[store_name]
        logs.append(f"Retrieved information for store '{store_name}' ({store_info['document_count']} documents)")
        return {
            "status": "success",
            "data": store_info,
            "logs": logs
        }


def delete_document_store(store_name: str) -> Dict:
    """
    Delete a document store and all its contents.
    
    WARNING: This permanently deletes all documents in the store.
    
    Args:
        store_name: Name of the store to delete
        
    Returns:
        Dict with deletion status
    """
    global _gemini_client, _document_stores
    logs = []
    
    if store_name not in _document_stores:
        return {
            "status": "not_found",
            "data": None,
            "logs": [f"Document store '{store_name}' not found"]
        }
    
    if _gemini_client is None:
        return {
            "status": "error",
            "data": None,
            "logs": ["Gemini client not initialized. Cannot delete remote store."]
        }
    
    try:
        store_info = _document_stores[store_name]
        
        # Delete the remote Gemini store
        _gemini_client.file_search_stores.delete(
            name=store_info["gemini_store_name"]
        )
        
        # Remove from local tracking
        del _document_stores[store_name]
        
        logs.append(f"Deleted document store '{store_name}' and all its documents")
        return {
            "status": "success",
            "data": {
                "deleted_store": store_name,
                "documents_deleted": store_info["document_count"]
            },
            "logs": logs
        }
        
    except Exception as e:
        error_msg = f"Error deleting document store '{store_name}': {e}"
        logs.append(error_msg)
        return {
            "status": "error",
            "data": None,
            "logs": logs
        }


def list_available_stores() -> Dict:
    """
    List all available document stores and their basic information.
    
    Returns:
        Dict with list of stores and their summary info
    """
    global _document_stores
    logs = []
    
    store_summaries = []
    for store_name, store_info in _document_stores.items():
        summary = {
            "name": store_name,
            "description": store_info.get("description", ""),
            "document_count": store_info.get("document_count", 0),
            "created_at": store_info.get("created_at", "unknown")
        }
        store_summaries.append(summary)
    
    logs.append(f"Listed {len(store_summaries)} available document stores")
    return {
        "status": "success",
        "data": {
            "total_stores": len(store_summaries),
            "stores": store_summaries
        },
        "logs": logs
    }