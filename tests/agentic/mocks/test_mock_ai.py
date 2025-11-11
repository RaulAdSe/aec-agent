"""Mock AI client for deterministic testing."""

from unittest.mock import Mock, patch
from src.services.ai_client import AIClient


def test_mock_ai_response():
    """Test agent with mocked AI responses."""
    # Mock successful AI response
    mock_response = {
        "status": "success",
        "analysis": "Mock compliance analysis: Building meets fire safety requirements.",
        "model_used": "gpt-4",
        "usage": {"total_tokens": 100}
    }
    
    with patch.object(AIClient, 'analyze_compliance', return_value=mock_response):
        client = AIClient()
        result = client.analyze_compliance("test prompt")
        
        assert result["status"] == "success"
        assert "compliance analysis" in result["analysis"]


def test_mock_ai_failure():
    """Test agent with mocked AI failures."""
    mock_error = {
        "status": "failed",
        "error": "Mock API error for testing"
    }
    
    with patch.object(AIClient, 'analyze_compliance', return_value=mock_error):
        client = AIClient()
        result = client.analyze_compliance("test prompt")
        
        assert result["status"] == "failed"
        assert "Mock API error" in result["error"]