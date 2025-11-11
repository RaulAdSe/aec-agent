"""
Test fixtures and sample data for compliance document testing.

This module provides reusable test data, mock responses, and fixtures
for testing the document retrieval and compliance search functionality.
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, List
from unittest.mock import Mock


class ComplianceDocumentFixtures:
    """Test fixtures for compliance documents."""
    
    # Sample document contents for testing
    SAMPLE_BUILDING_CODE = """
BUILDING CODE REQUIREMENTS

Section 1: Door Specifications
1.1 Width Requirements:
- Standard doors: minimum 30 inches
- Accessible doors: minimum 32 inches clear width
- Emergency exit doors: minimum 36 inches

1.2 Hardware Requirements:
- Maximum opening force: 5 pounds (interior), 8.5 pounds (exterior)
- Handle height: 34-48 inches above floor
- Panic hardware required for occupancy >50 persons

1.3 Fire Rating:
- Fire doors must maintain wall assembly rating
- Self-closing mechanism required
- Maximum gap under door: 3/4 inch

Section 2: Ceiling Heights
2.1 Minimum Heights:
- Residential: 7 feet (2.13 m)
- Office: 7 feet 6 inches (2.29 m)
- Industrial: 8 feet (2.44 m)
- Retail: 8 feet (2.44 m)

2.2 Exceptions:
- Basement ceilings: 6 feet 8 inches minimum
- Sloped ceilings: average height may be used
"""

    SAMPLE_ACCESSIBILITY_GUIDE = """
ACCESSIBILITY STANDARDS

Ramp Requirements:
- Maximum slope: 1:12 (8.33% grade)
- Maximum rise per run: 30 inches
- Minimum width: 36 inches (44 inches between handrails)
- Landing size: 60 inches x 60 inches minimum

Door Requirements:
- Clear width: 32 inches minimum
- Maneuvering clearance: 18 inches (pull side), 12 inches (push side)
- Threshold: maximum 1/2 inch height
- Opening force: 5 pounds maximum

Parking Requirements:
- Accessible spaces: 1 per 25 parking spaces
- Van accessible: 1 per 6 accessible spaces
- Space width: 96 inches (132 inches for van accessible)
"""

    SAMPLE_FIRE_SAFETY = """
FIRE SAFETY REGULATIONS

Sprinkler Systems:
- Required in buildings over 5,000 sq ft
- Required in high-rise buildings (>75 feet)
- Residential: required in buildings >3 stories

Exit Requirements:
- Maximum travel distance: 200 feet (unsprinklered), 250 feet (sprinklered)
- Exit width: 0.3 inches per occupant (stairs), 0.2 inches per occupant (doors)
- Number of exits: minimum 2 for occupancy >49 persons

Fire Doors:
- Rating must match wall assembly
- Self-closing required
- Maximum opening force: 15 pounds
- Annual inspection required
"""

    SAMPLE_STRUCTURAL_REQUIREMENTS = """
STRUCTURAL REQUIREMENTS

Foundation Requirements:
- Depth: minimum 42 inches below frost line
- Width: minimum 6 inches, or width of wall above
- Reinforcement: #4 rebar minimum

Floor Load Requirements:
- Residential: 40 PSF live load, 10 PSF dead load
- Office: 50 PSF live load, 20 PSF dead load  
- Retail: 75 PSF live load, 20 PSF dead load
- Storage: 125 PSF live load, 20 PSF dead load

Wind Load Requirements:
- Basic wind speed: varies by geographic location
- Exposure category: B (urban), C (open), D (flat coastal)
- Building height factor applies to structures >60 feet
"""

    @classmethod
    def create_temp_documents(cls) -> List[str]:
        """
        Create temporary files with sample compliance documents.
        
        Returns:
            List of file paths to created temporary documents
        """
        documents = [
            ("building_code.txt", cls.SAMPLE_BUILDING_CODE),
            ("accessibility_guide.txt", cls.SAMPLE_ACCESSIBILITY_GUIDE),
            ("fire_safety.txt", cls.SAMPLE_FIRE_SAFETY),
            ("structural_requirements.txt", cls.SAMPLE_STRUCTURAL_REQUIREMENTS)
        ]
        
        file_paths = []
        for filename, content in documents:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix='.txt',
                prefix=filename.split('.')[0] + '_',
                delete=False
            ) as f:
                f.write(content.strip())
                file_paths.append(f.name)
        
        return file_paths

    @classmethod
    def cleanup_temp_documents(cls, file_paths: List[str]):
        """Clean up temporary document files."""
        for path in file_paths:
            try:
                os.unlink(path)
            except OSError:
                pass  # File might already be deleted


class MockGeminiResponses:
    """Mock responses for Gemini API calls."""
    
    @staticmethod
    def create_mock_search_response(content: str, citations: List[Dict] = None) -> Mock:
        """Create a mock Gemini search response."""
        response = Mock()
        response.text = content
        
        # Mock candidates structure
        response.candidates = [Mock()]
        response.candidates[0].content.parts = [Mock()]
        response.candidates[0].content.parts[0].text = content
        
        # Mock citations
        if citations:
            response.candidates[0].citation_metadata = Mock()
            response.candidates[0].citation_metadata.citations = []
            
            for citation in citations:
                cite_mock = Mock()
                cite_mock.start_index = citation.get("start_index", 0)
                cite_mock.end_index = citation.get("end_index", 10)
                cite_mock.source = citation.get("source", "test_document.pdf")
                response.candidates[0].citation_metadata.citations.append(cite_mock)
        else:
            response.candidates[0].citation_metadata = None
        
        return response

    @staticmethod
    def create_mock_upload_operation() -> Mock:
        """Create a mock upload operation response."""
        operation = Mock()
        operation.name = "fileSearchStores/test-store/upload/operations/test-operation-123"
        return operation

    @staticmethod  
    def create_mock_file_search_store() -> Mock:
        """Create a mock file search store."""
        store = Mock()
        store.name = "fileSearchStores/test-compliance-store-abc123"
        store.create_time = "2023-01-01T00:00:00Z"
        return store


class TestQueries:
    """Common test queries for compliance testing."""
    
    DOOR_WIDTH_QUERIES = [
        "What is the minimum door width for accessibility?",
        "Emergency exit door width requirements",
        "Accessible door clear width specifications",
        "ADA door width standards"
    ]
    
    CEILING_HEIGHT_QUERIES = [
        "Minimum ceiling height for office spaces",
        "Residential ceiling height requirements", 
        "Industrial building ceiling standards",
        "Building code ceiling height minimums"
    ]
    
    RAMP_SLOPE_QUERIES = [
        "Maximum wheelchair ramp slope",
        "ADA ramp slope requirements",
        "Accessible ramp grade specifications",
        "Handicap ramp maximum incline"
    ]
    
    FIRE_SAFETY_QUERIES = [
        "Sprinkler system requirements",
        "Fire door specifications",
        "Emergency exit travel distance",
        "Fire safety building codes"
    ]
    
    @classmethod
    def get_all_test_queries(cls) -> List[str]:
        """Get all test queries combined."""
        return (cls.DOOR_WIDTH_QUERIES + 
                cls.CEILING_HEIGHT_QUERIES + 
                cls.RAMP_SLOPE_QUERIES + 
                cls.FIRE_SAFETY_QUERIES)


class ExpectedAnswers:
    """Expected answer patterns for test validation."""
    
    DOOR_WIDTH_PATTERNS = [
        "32 inches",
        "minimum",
        "accessible",
        "clear width"
    ]
    
    CEILING_HEIGHT_PATTERNS = [
        "7 feet",
        "2.13 m",
        "7 feet 6 inches",
        "minimum"
    ]
    
    RAMP_SLOPE_PATTERNS = [
        "1:12",
        "8.33%",
        "maximum slope",
        "grade"
    ]
    
    FIRE_SAFETY_PATTERNS = [
        "5,000",
        "sprinkler",
        "travel distance",
        "200 feet"
    ]
    
    @classmethod
    def get_expected_patterns(cls, query_type: str) -> List[str]:
        """Get expected patterns for a query type."""
        pattern_map = {
            "door_width": cls.DOOR_WIDTH_PATTERNS,
            "ceiling_height": cls.CEILING_HEIGHT_PATTERNS,
            "ramp_slope": cls.RAMP_SLOPE_PATTERNS, 
            "fire_safety": cls.FIRE_SAFETY_PATTERNS
        }
        return pattern_map.get(query_type, [])


def create_test_store_info(store_name: str, doc_count: int = 5) -> Dict:
    """Create test store information data."""
    return {
        "name": store_name,
        "description": f"Test store for {store_name}",
        "gemini_store_name": f"fileSearchStores/{store_name.replace('_', '')}-test123",
        "gemini_store_id": f"{store_name.replace('_', '')}-test123",
        "created_at": "2023-01-01T00:00:00Z",
        "document_count": doc_count,
        "documents": [
            {
                "file_path": f"/test/doc{i}.pdf",
                "file_name": f"doc{i}.pdf",
                "document_type": "compliance",
                "uploaded_at": "2023-01-01T00:00:00Z"
            } for i in range(doc_count)
        ]
    }


def assert_valid_search_result(result: Dict, expected_patterns: List[str] = None):
    """Assert that a search result has the expected structure and content."""
    # Check result structure
    assert "status" in result
    assert "answer" in result
    assert "citations" in result
    assert "documents_searched" in result
    assert "query" in result
    assert "logs" in result
    
    # Check successful result
    if result["status"] == "success":
        assert len(result["answer"]) > 0, "Answer should not be empty"
        assert result["documents_searched"] > 0, "Should have searched documents"
        assert len(result["query"]) > 0, "Query should not be empty"
        
        # Check for expected patterns if provided
        if expected_patterns:
            answer_lower = result["answer"].lower()
            for pattern in expected_patterns:
                assert pattern.lower() in answer_lower, f"Expected pattern '{pattern}' not found in answer"


def assert_valid_store_info(store_info: Dict):
    """Assert that store info has the expected structure."""
    required_fields = ["name", "description", "document_count", "gemini_store_name"]
    
    for field in required_fields:
        assert field in store_info, f"Missing required field: {field}"
    
    assert isinstance(store_info["document_count"], int)
    assert store_info["document_count"] >= 0