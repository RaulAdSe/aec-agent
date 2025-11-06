"""Unit tests for AI client."""

from src.services.ai_client import AIClient


def test_ai_client_init_no_key():
    """Test AI client initializes without API key."""
    client = AIClient()
    assert client.client is None
    assert client.api_key is None


def test_ai_client_analyze_no_key():
    """Test analyze_compliance fails gracefully without API key."""
    client = AIClient()
    result = client.analyze_compliance("test prompt")
    assert result["status"] == "failed"
    assert "No OpenAI API key" in result["error"]


def test_ai_client_connection_check_no_key():
    """Test connection check fails without API key."""
    client = AIClient()
    assert client.check_connection() is False