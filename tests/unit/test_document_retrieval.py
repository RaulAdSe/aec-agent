"""
Unit tests for document retrieval toolkit.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tools.document_retrieval_toolkit import (
    initialize_gemini_client,
    create_document_store,
    upload_documents,
    search_documents,
    get_store_info,
    delete_document_store,
    list_available_stores
)


class TestDocumentRetrievalToolkit:
    """Test cases for document retrieval functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = None
        toolkit._document_stores = {}

    @patch('google.genai.Client')
    def test_initialize_gemini_client_success(self, mock_client_class):
        """Test successful Gemini client initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        result = initialize_gemini_client("test-api-key")
        
        assert result["status"] == "success"
        assert result["data"]["client_ready"] is True
        assert "Gemini client initialized successfully" in result["logs"]
        mock_client_class.assert_called_once_with(api_key="test-api-key")

    @patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"})
    @patch('google.genai.Client')
    def test_initialize_gemini_client_from_env(self, mock_client_class):
        """Test Gemini client initialization from environment variable."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        result = initialize_gemini_client()
        
        assert result["status"] == "success"
        mock_client_class.assert_called_once_with(api_key="env-api-key")

    def test_initialize_gemini_client_no_api_key(self):
        """Test Gemini client initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            result = initialize_gemini_client()
            
            assert result["status"] == "error"
            assert "API key not provided" in result["logs"][0]

    @patch('google.genai.Client')
    def test_create_document_store_success(self, mock_client_class):
        """Test successful document store creation."""
        # Setup mock client
        mock_client = Mock()
        mock_store = Mock()
        mock_store.name = "fileSearchStores/test-store-id"
        mock_store.create_time = "2023-01-01T00:00:00Z"
        mock_client.file_search_stores.create.return_value = mock_store
        
        # Initialize client first
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = mock_client
        
        result = create_document_store("test_store", "Test store description")
        
        assert result["status"] == "success"
        assert result["data"]["name"] == "test_store"
        assert result["data"]["description"] == "Test store description"
        assert result["data"]["document_count"] == 0
        assert "test-store-id" in result["data"]["gemini_store_id"]

    def test_create_document_store_no_client(self):
        """Test document store creation without initialized client."""
        result = create_document_store("test_store", "Test description")
        
        assert result["status"] == "error"
        assert "Gemini client not initialized" in result["logs"][0]

    @patch('google.genai.Client')
    def test_create_document_store_duplicate(self, mock_client_class):
        """Test creating duplicate document store returns existing."""
        # Setup mock client
        mock_client = Mock()
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = mock_client
        
        # Pre-populate store
        toolkit._document_stores["existing_store"] = {
            "name": "existing_store",
            "description": "Existing store",
            "document_count": 5
        }
        
        result = create_document_store("existing_store", "New description")
        
        assert result["status"] == "success"
        assert result["data"]["document_count"] == 5
        assert "already exists" in result["logs"][0]

    def test_upload_documents_no_client(self):
        """Test document upload without initialized client."""
        result = upload_documents("test_store", ["test.pdf"])
        
        assert result["status"] == "error"
        assert "Gemini client not initialized" in result["logs"][0]

    def test_upload_documents_no_store(self):
        """Test document upload to non-existent store."""
        # Setup mock client
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = Mock()
        
        result = upload_documents("nonexistent_store", ["test.pdf"])
        
        assert result["status"] == "error"
        assert "Document store 'nonexistent_store' not found" in result["logs"][0]

    @patch('google.genai.Client')
    @patch('os.path.exists')
    def test_upload_documents_success(self, mock_exists, mock_client_class):
        """Test successful document upload."""
        # Setup mocks
        mock_exists.return_value = True
        mock_client = Mock()
        mock_operation = Mock()
        mock_operation.name = "operations/upload-123"
        mock_client.file_search_stores.upload_to_file_search_store.return_value = mock_operation
        
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = mock_client
        toolkit._document_stores["test_store"] = {
            "name": "test_store",
            "gemini_store_name": "fileSearchStores/test-store-id",
            "documents": [],
            "document_count": 0
        }
        
        result = upload_documents("test_store", ["test.pdf"], ["test_doc"])
        
        assert result["status"] == "success"
        assert result["data"]["successful_uploads"] == 1
        assert result["data"]["total_files"] == 1
        assert len(result["data"]["upload_results"]) == 1
        assert result["data"]["upload_results"][0]["status"] == "success"

    @patch('google.genai.Client')
    @patch('os.path.exists')
    def test_upload_documents_file_not_found(self, mock_exists, mock_client_class):
        """Test document upload with non-existent file."""
        # Setup mocks
        mock_exists.return_value = False
        mock_client = Mock()
        
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = mock_client
        toolkit._document_stores["test_store"] = {
            "name": "test_store",
            "gemini_store_name": "fileSearchStores/test-store-id",
            "documents": [],
            "document_count": 0
        }
        
        result = upload_documents("test_store", ["nonexistent.pdf"])
        
        assert result["status"] == "error"
        assert result["data"]["successful_uploads"] == 0
        assert "File not found: nonexistent.pdf" in result["logs"]

    def test_search_documents_no_client(self):
        """Test search without initialized client."""
        result = search_documents("test_store", "test query")
        
        assert result["status"] == "error"
        assert "Gemini client not initialized" in result["logs"][0]

    def test_search_documents_no_store(self):
        """Test search on non-existent store."""
        # Setup mock client
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = Mock()
        
        result = search_documents("nonexistent_store", "test query")
        
        assert result["status"] == "error"
        assert "Document store 'nonexistent_store' not found" in result["logs"][0]

    @patch('google.genai.Client')
    def test_search_documents_empty_store(self, mock_client_class):
        """Test search on empty document store."""
        # Setup mock client
        mock_client = Mock()
        
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = mock_client
        toolkit._document_stores["empty_store"] = {
            "name": "empty_store",
            "document_count": 0
        }
        
        result = search_documents("empty_store", "test query")
        
        assert result["status"] == "warning"
        assert result["data"]["total_results"] == 0
        assert "No documents in store" in result["logs"][0]

    @patch('google.genai.Client')
    def test_search_documents_success(self, mock_client_class):
        """Test successful document search."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "This is the search result text"
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].citation_metadata = None
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = mock_client
        toolkit._document_stores["test_store"] = {
            "name": "test_store",
            "gemini_store_name": "fileSearchStores/test-store-id", 
            "document_count": 5
        }
        
        result = search_documents("test_store", "test query")
        
        assert result["status"] == "success"
        assert result["data"]["content"] == "This is the search result text"
        assert result["data"]["query"] == "test query"
        assert result["data"]["store_searched"] == "test_store"
        assert result["data"]["documents_in_store"] == 5

    def test_get_store_info_not_found(self):
        """Test get store info for non-existent store."""
        result = get_store_info("nonexistent_store")
        
        assert result["status"] == "not_found"
        assert "Document store 'nonexistent_store' not found" in result["logs"]

    def test_get_store_info_success(self):
        """Test successful get store info."""
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._document_stores["test_store"] = {
            "name": "test_store",
            "description": "Test store",
            "document_count": 3
        }
        
        result = get_store_info("test_store")
        
        assert result["status"] == "success"
        assert result["data"]["name"] == "test_store"
        assert result["data"]["document_count"] == 3

    def test_get_store_info_all_stores(self):
        """Test get info for all stores."""
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._document_stores = {
            "store1": {"name": "store1", "document_count": 5},
            "store2": {"name": "store2", "document_count": 3}
        }
        
        result = get_store_info()
        
        assert result["status"] == "success"
        assert result["data"]["total_stores"] == 2
        assert "store1" in result["data"]["stores"]
        assert "store2" in result["data"]["stores"]

    def test_list_available_stores(self):
        """Test list available stores."""
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._document_stores = {
            "store1": {
                "name": "store1",
                "description": "First store", 
                "document_count": 5,
                "created_at": "2023-01-01"
            },
            "store2": {
                "name": "store2",
                "description": "Second store",
                "document_count": 3,
                "created_at": "2023-01-02"
            }
        }
        
        result = list_available_stores()
        
        assert result["status"] == "success"
        assert result["data"]["total_stores"] == 2
        
        stores = result["data"]["stores"]
        assert len(stores) == 2
        assert stores[0]["name"] == "store1"
        assert stores[0]["document_count"] == 5
        assert stores[1]["name"] == "store2"
        assert stores[1]["document_count"] == 3

    def test_delete_document_store_not_found(self):
        """Test delete non-existent store."""
        result = delete_document_store("nonexistent_store")
        
        assert result["status"] == "not_found"
        assert "Document store 'nonexistent_store' not found" in result["logs"]

    def test_delete_document_store_no_client(self):
        """Test delete store without initialized client."""
        # Setup toolkit state  
        import tools.document_retrieval_toolkit as toolkit
        toolkit._document_stores["test_store"] = {"name": "test_store"}
        
        result = delete_document_store("test_store")
        
        assert result["status"] == "error"
        assert "Gemini client not initialized" in result["logs"][0]

    @patch('google.genai.Client')
    def test_delete_document_store_success(self, mock_client_class):
        """Test successful store deletion."""
        # Setup mock client
        mock_client = Mock()
        mock_client.file_search_stores.delete.return_value = None
        
        # Setup toolkit state
        import tools.document_retrieval_toolkit as toolkit
        toolkit._gemini_client = mock_client
        toolkit._document_stores["test_store"] = {
            "name": "test_store",
            "gemini_store_name": "fileSearchStores/test-store-id",
            "document_count": 5
        }
        
        result = delete_document_store("test_store")
        
        assert result["status"] == "success"
        assert result["data"]["deleted_store"] == "test_store"
        assert result["data"]["documents_deleted"] == 5
        assert "test_store" not in toolkit._document_stores


class TestComplianceSearch:
    """Test cases for compliance search functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset global state
        import tools.compliance_search as compliance_search
        compliance_search._client_initialized = False

    @patch('tools.compliance_search.initialize_gemini_client')
    @patch('tools.compliance_search.get_store_info')
    def test_search_compliance_docs_no_store(self, mock_get_store, mock_init_client):
        """Test search when knowledge base doesn't exist."""
        from tools.compliance_search import search_compliance_docs
        
        mock_init_client.return_value = {"status": "success", "logs": ["Client initialized"]}
        mock_get_store.return_value = {"status": "not_found"}
        
        result = search_compliance_docs("test query")
        
        assert result["status"] == "no_documents"
        assert "No compliance knowledge base found" in result["answer"]
        assert result["documents_searched"] == 0

    @patch('tools.compliance_search.initialize_gemini_client')
    @patch('tools.compliance_search.get_store_info')
    def test_search_compliance_docs_empty_store(self, mock_get_store, mock_init_client):
        """Test search when knowledge base is empty."""
        from tools.compliance_search import search_compliance_docs
        
        mock_init_client.return_value = {"status": "success", "logs": ["Client initialized"]}
        mock_get_store.return_value = {
            "status": "success",
            "data": {"document_count": 0}
        }
        
        result = search_compliance_docs("test query")
        
        assert result["status"] == "no_documents"
        assert "No documents in compliance knowledge base" in result["answer"]
        assert result["documents_searched"] == 0

    @patch('tools.compliance_search.initialize_gemini_client')
    @patch('tools.compliance_search.get_store_info') 
    @patch('tools.compliance_search.search_documents')
    def test_search_compliance_docs_success(self, mock_search, mock_get_store, mock_init_client):
        """Test successful compliance search."""
        from tools.compliance_search import search_compliance_docs
        
        mock_init_client.return_value = {"status": "success", "logs": ["Client initialized"]}
        mock_get_store.return_value = {
            "status": "success", 
            "data": {"document_count": 5}
        }
        mock_search.return_value = {
            "status": "success",
            "data": {
                "content": "This is the search result",
                "citations": [],
                "documents_in_store": 5
            },
            "logs": ["Search completed"]
        }
        
        result = search_compliance_docs("minimum door width")
        
        assert result["status"] == "success"
        assert result["answer"] == "This is the search result"
        assert result["documents_searched"] == 5
        assert result["query"] == "minimum door width"

    @patch('tools.compliance_search.initialize_gemini_client')
    def test_search_compliance_docs_init_failure(self, mock_init_client):
        """Test search when client initialization fails."""
        from tools.compliance_search import search_compliance_docs
        
        mock_init_client.return_value = {
            "status": "error", 
            "logs": ["API key not found"]
        }
        
        result = search_compliance_docs("test query")
        
        assert result["status"] == "error"
        assert "Failed to initialize compliance search" in result["logs"][0]
        assert result["documents_searched"] == 0

    @patch('tools.compliance_search.initialize_gemini_client')
    @patch('tools.compliance_search.get_store_info')
    def test_check_knowledge_base_status_not_found(self, mock_get_store, mock_init_client):
        """Test knowledge base status check when not found."""
        from tools.compliance_search import check_knowledge_base_status
        
        mock_init_client.return_value = {"status": "success", "logs": ["Client initialized"]}
        mock_get_store.return_value = {"status": "not_found"}
        
        result = check_knowledge_base_status()
        
        assert result["status"] == "not_found"
        assert result["document_count"] == 0
        assert "Knowledge base not found" in result["message"]

    @patch('tools.compliance_search.initialize_gemini_client')
    @patch('tools.compliance_search.get_store_info')
    def test_check_knowledge_base_status_ready(self, mock_get_store, mock_init_client):
        """Test knowledge base status check when ready."""
        from tools.compliance_search import check_knowledge_base_status
        
        mock_init_client.return_value = {"status": "success", "logs": ["Client initialized"]}
        mock_get_store.return_value = {
            "status": "success",
            "data": {"document_count": 10}
        }
        
        result = check_knowledge_base_status()
        
        assert result["status"] == "ready"
        assert result["document_count"] == 10
        assert "Knowledge base ready with 10 documents" in result["message"]


# Test fixtures and utilities
@pytest.fixture
def temp_doc_file():
    """Create a temporary document file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test compliance document.\n")
        f.write("It contains sample building regulations and requirements.\n")
        f.write("Door width must be minimum 80cm for accessibility.\n")
    
    yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture 
def mock_gemini_response():
    """Create a mock Gemini API response."""
    response = Mock()
    response.text = "Based on the building codes, minimum door width is 80cm."
    response.candidates = [Mock()]
    response.candidates[0].citation_metadata = Mock()
    response.candidates[0].citation_metadata.citations = [
        Mock(start_index=10, end_index=50, source="building_code.pdf")
    ]
    return response


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])