"""Tools for the AEC compliance agent."""

from .building_data_toolkit import (
    load_building_data,
    get_all_elements, 
    get_all_properties,
    query_elements,
    calculate,
    find_related,
    validate_rule
)

from .document_retrieval_toolkit import (
    initialize_gemini_client,
    create_document_store,
    upload_documents,
    search_documents,
    get_store_info,
    delete_document_store,
    list_available_stores
)

from .compliance_search import (
    search_compliance_docs,
    check_knowledge_base_status
)

from .citation_utils import (
    format_citation_text,
    format_citation_markdown,
    extract_document_sources,
    get_citations_by_document,
    validate_citation_quality
)

__all__ = [
    # Building Data Tools
    'load_building_data',
    'get_all_elements',
    'get_all_properties', 
    'query_elements',
    'calculate',
    'find_related',
    'validate_rule',
    # Document Retrieval Tools
    'initialize_gemini_client',
    'create_document_store',
    'upload_documents', 
    'search_documents',
    'get_store_info',
    'delete_document_store',
    'list_available_stores',
    # Agent-Friendly Compliance Search
    'search_compliance_docs',
    'check_knowledge_base_status',
    # Citation Utilities
    'format_citation_text',
    'format_citation_markdown',
    'extract_document_sources',
    'get_citations_by_document',
    'validate_citation_quality'
]