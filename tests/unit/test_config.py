"""Unit tests for configuration module."""

from src.core.config import AppConfig


def test_default_config():
    """Test default configuration values."""
    config = AppConfig()
    assert config.default_model == "gpt-4"
    assert config.temperature == 0.1
    assert config.max_tokens == 8192
    assert config.use_toon is True


def test_config_environment_loading():
    """Test configuration loads from environment."""
    config = AppConfig()
    # Should handle missing env vars gracefully
    assert config.openai_api_key is None  # No env var set in test