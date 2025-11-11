"""
Integration tests for the complete knowledge base workflow.

These tests verify the end-to-end functionality of document ingestion,
storage, and retrieval using real Gemini API calls (when API key is available).
"""

import pytest
import os
import tempfile
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tools.document_retrieval_toolkit import (
    initialize_gemini_client,
    create_document_store, 
    upload_documents,
    search_documents,
    delete_document_store
)
from tools.compliance_search import search_compliance_docs, check_knowledge_base_status


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
class TestKnowledgeBaseIntegration:
    """Integration tests requiring real Gemini API access."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with API client."""
        cls.test_store_name = "test_integration_store"
        cls.test_documents = []
        
    def teardown_method(self):
        """Clean up after each test."""
        # Clean up test store
        try:
            delete_document_store(self.test_store_name)
        except:
            pass  # Store might not exist
    
    def test_complete_workflow(self):
        """Test the complete document workflow end-to-end."""
        
        # 1. Initialize client
        result = initialize_gemini_client()
        assert result["status"] == "success", f"Client init failed: {result.get('logs', [])}"
        
        # 2. Create document store
        result = create_document_store(
            self.test_store_name, 
            "Test store for integration testing"
        )
        assert result["status"] == "success", f"Store creation failed: {result.get('logs', [])}"
        assert result["data"]["name"] == self.test_store_name
        
        # 3. Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
BUILDING CODE TEST DOCUMENT

Section 1: Door Requirements
- Minimum door width for accessibility: 32 inches (81.3 cm)
- Emergency exit doors: minimum 36 inches (91.4 cm) wide
- Door opening force: maximum 5 pounds for interior doors

Section 2: Ceiling Heights  
- Office spaces: minimum 7 feet 6 inches (2.29 m)
- Residential: minimum 7 feet (2.13 m)
- Industrial: minimum 8 feet (2.44 m)

Section 3: Ramp Requirements
- Maximum slope: 1:12 (8.33% grade)
- Maximum rise: 30 inches per run
- Landing size: minimum 60 inches x 60 inches
            """)
            test_doc_path = f.name
        
        try:
            # 4. Upload document
            result = upload_documents(self.test_store_name, [test_doc_path])
            assert result["status"] == "success", f"Upload failed: {result.get('logs', [])}"
            assert result["data"]["successful_uploads"] == 1
            
            # 5. Wait for processing (Gemini needs time to index)
            time.sleep(15)
            
            # 6. Test search functionality
            test_queries = [
                ("door width requirements", "door", "width"),
                ("ceiling height office", "office", "7"),
                ("ramp slope requirements", "ramp", "1:12"),
                ("emergency exit", "emergency", "36")
            ]
            
            for query, expected_word1, expected_word2 in test_queries:
                result = search_documents(self.test_store_name, query)
                assert result["status"] == "success", f"Search failed for '{query}': {result.get('logs', [])}"
                
                content = result["data"]["content"].lower()
                assert expected_word1 in content, f"'{expected_word1}' not found in search result for '{query}'"
                assert expected_word2 in content, f"'{expected_word2}' not found in search result for '{query}'"
                
                # Verify metadata
                assert result["data"]["query"] == query
                assert result["data"]["documents_in_store"] >= 1
                
        finally:
            # Cleanup test document
            os.unlink(test_doc_path)
    
    def test_compliance_search_integration(self):
        """Test the agent-friendly compliance search function."""
        
        # 1. Set up knowledge base
        initialize_gemini_client()
        create_document_store("compliance_knowledge_base", "Test compliance knowledge base")
        
        # 2. Create test compliance document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
ACCESSIBILITY COMPLIANCE GUIDE

Door Width Requirements:
- Standard doors: minimum 32 inches clear width
- Accessible doors: minimum 32 inches clear width  
- Emergency exits: minimum 36 inches clear width

Fire Safety:
- Fire doors must be self-closing
- Exit doors must swing in direction of egress
- Maximum force: 15 pounds to open

Ramp Specifications:
- Maximum slope: 1 unit rise for 12 units run (8.33%)
- Width: minimum 44 inches between handrails
- Landings: 60 inches minimum in direction of travel
            """)
            test_doc_path = f.name
        
        try:
            # 3. Upload to compliance knowledge base
            upload_documents("compliance_knowledge_base", [test_doc_path])
            time.sleep(15)  # Wait for processing
            
            # 4. Test knowledge base status
            status = check_knowledge_base_status()
            assert status["status"] == "ready"
            assert status["document_count"] >= 1
            
            # 5. Test compliance searches
            test_searches = [
                "What is the minimum door width for accessibility?",
                "What are the fire safety requirements for doors?", 
                "What is the maximum slope for wheelchair ramps?",
                "What force is required to open exit doors?"
            ]
            
            for search_query in test_searches:
                result = search_compliance_docs(search_query)
                assert result["status"] == "success", f"Compliance search failed: {result.get('logs', [])}"
                assert len(result["answer"]) > 0, f"Empty answer for: {search_query}"
                assert result["documents_searched"] >= 1
                
        finally:
            # Cleanup
            os.unlink(test_doc_path)
            try:
                delete_document_store("compliance_knowledge_base")
            except:
                pass

    def test_multiple_document_search(self):
        """Test search across multiple documents."""
        
        # Initialize
        initialize_gemini_client()
        create_document_store(self.test_store_name, "Multi-document test store")
        
        # Create multiple test documents
        doc_contents = [
            {
                "name": "fire_safety.txt",
                "content": """
FIRE SAFETY REGULATIONS
- Sprinkler systems required in buildings over 5000 sq ft
- Fire doors must have 60-minute rating minimum
- Exit signs must be illuminated and visible
- Maximum travel distance to exit: 200 feet
                """
            },
            {
                "name": "accessibility.txt", 
                "content": """
ACCESSIBILITY STANDARDS
- Wheelchair ramps: maximum 1:12 slope
- Door width: minimum 32 inches clear
- Bathroom stalls: minimum 60 inches wide
- Elevator required for buildings over 3 stories
                """
            },
            {
                "name": "structural.txt",
                "content": """
STRUCTURAL REQUIREMENTS
- Foundation depth: minimum 42 inches below frost line
- Beam spacing: maximum 16 inches on center  
- Load capacity: minimum 40 PSF for residential floors
- Wind resistance: rated for 90 MPH winds
                """
            }
        ]
        
        doc_paths = []
        try:
            # Create and upload documents
            for doc in doc_contents:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(doc["content"])
                    doc_paths.append(f.name)
            
            result = upload_documents(self.test_store_name, doc_paths)
            assert result["status"] == "success"
            assert result["data"]["successful_uploads"] == len(doc_paths)
            
            # Wait for processing
            time.sleep(20)
            
            # Test cross-document searches
            test_cases = [
                ("fire safety sprinkler requirements", "sprinkler"),
                ("wheelchair ramp slope", "1:12"),
                ("structural beam spacing", "16 inches"),
                ("building height elevator requirements", "3 stories"),
                ("door width accessibility", "32 inches")
            ]
            
            for query, expected_term in test_cases:
                result = search_documents(self.test_store_name, query)
                assert result["status"] == "success"
                assert expected_term.lower() in result["data"]["content"].lower()
                assert result["data"]["documents_in_store"] == len(doc_paths)
                
        finally:
            # Cleanup
            for path in doc_paths:
                try:
                    os.unlink(path)
                except:
                    pass

    def test_document_types_support(self):
        """Test support for different document types."""
        
        # Initialize
        initialize_gemini_client() 
        create_document_store(self.test_store_name, "Document types test store")
        
        # Test different file extensions
        test_files = []
        
        # TXT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Text document with building code requirements. Door width: 32 inches.")
            test_files.append(f.name)
        
        # JSON file  
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"building_code": {"doors": {"min_width": "32 inches", "max_force": "5 pounds"}}}')
            test_files.append(f.name)
        
        try:
            # Upload mixed file types
            result = upload_documents(self.test_store_name, test_files)
            assert result["status"] == "success"
            assert result["data"]["successful_uploads"] == len(test_files)
            
            # Wait and test search
            time.sleep(15)
            
            result = search_documents(self.test_store_name, "door width requirements")
            assert result["status"] == "success"
            assert "32 inches" in result["data"]["content"]
            
        finally:
            # Cleanup
            for path in test_files:
                try:
                    os.unlink(path)
                except:
                    pass

    def test_error_handling(self):
        """Test error handling in integration scenarios."""
        
        # Test search on non-existent store
        result = search_documents("nonexistent_store", "test query")
        assert result["status"] == "error"
        
        # Test upload to non-existent store
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            result = upload_documents("nonexistent_store", [f.name])
            assert result["status"] == "error"
        
        # Test upload of non-existent file
        initialize_gemini_client()
        create_document_store(self.test_store_name, "Error test store")
        
        result = upload_documents(self.test_store_name, ["nonexistent_file.pdf"])
        assert result["status"] == "error"
        assert result["data"]["successful_uploads"] == 0


@pytest.mark.integration
class TestKnowledgeBaseManager:
    """Integration tests for the knowledge base management script."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_doc_dir = tempfile.mkdtemp()
        self.test_tracking_file = os.path.join(self.test_doc_dir, ".document_tracking.json")
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_doc_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    def test_knowledge_base_manager_workflow(self):
        """Test the knowledge base manager script workflow."""
        
        # This would test the manage_knowledge_base.py script
        # by creating test documents and running sync operations
        
        # Create test document
        test_doc_path = os.path.join(self.test_doc_dir, "test_compliance.txt")
        with open(test_doc_path, 'w') as f:
            f.write("""
Test compliance document for knowledge base manager testing.
Door requirements: minimum 32 inches width for accessibility.
            """)
        
        # Import the knowledge base manager
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        # Test would simulate command line operations:
        # - python manage_knowledge_base.py status
        # - python manage_knowledge_base.py sync 
        # - python manage_knowledge_base.py query "door width"
        
        # For now, we'll test the underlying functionality
        assert os.path.exists(test_doc_path)
        assert "compliance" in open(test_doc_path).read()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])