"""Base agent class for all agents in the system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..core.logger import get_logger
from ..core.config import config


class BaseAgentConfig(BaseModel):
    """Base configuration for all agents."""
    
    name: str
    description: Optional[str] = None
    model_name: str = config.default_model
    temperature: float = config.temperature
    max_tokens: int = config.max_tokens
    verbose: bool = False


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality and enforces interface.
    """
    
    def __init__(self, config: BaseAgentConfig):
        """Initialize the base agent."""
        self.config = config
        self.logger = get_logger(f"agent.{config.name}")
        
        # Initialize memory and tools
        self._initialize_memory()
        self._initialize_tools()
        
        self.logger.info(f"Initialized {config.name} agent")
    
    @abstractmethod
    def _initialize_memory(self) -> None:
        """Initialize agent memory. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _initialize_tools(self) -> None:
        """Initialize agent tools. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processing results
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "model": self.config.model_name,
            "status": "ready"
        }