"""Central configuration for the AEC compliance agent."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application configuration."""
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Model Configuration
    default_model: str = Field(default="gemini-2.0-flash-exp", description="Default LLM model")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_tokens: int = Field(default=8192, description="Maximum tokens per response")
    
    # TOON Configuration
    use_toon: bool = Field(default=True, description="Enable TOON format")
    
    # API Keys (loaded from environment)
    google_api_key: Optional[str] = Field(default=None, description="Google AI API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Load API keys from environment
        if not self.google_api_key:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")


# Global config instance
config = AppConfig()