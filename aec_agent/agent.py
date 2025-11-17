"""
Agent factory - Create and configure AEC Compliance Agent instances.

This module provides simple factory functions for creating configured
agent instances with the TaskGraph-enabled reasoning system.
"""

from typing import Dict, Any, Optional
from .core.reasoning_agent import ReasoningAgent
from .config import AgentConfig


def create_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    verbose: bool = True,
    enable_memory: bool = True,
    session_id: Optional[str] = None,
    max_iterations: int = 20,
    max_execution_time: float = 300.0
) -> ReasoningAgent:
    """
    Create a configured AEC Compliance Agent with TaskGraph support.
    
    Args:
        model_name: LLM model to use
        temperature: LLM temperature (0.0-1.0)
        verbose: Enable verbose logging
        enable_memory: Enable memory system
        session_id: Optional session identifier
        max_iterations: Maximum reasoning iterations
        max_execution_time: Maximum execution time in seconds
        
    Returns:
        Configured ReasoningAgent instance
    """
    return ReasoningAgent(
        model_name=model_name,
        temperature=temperature,
        verbose=verbose,
        enable_memory=enable_memory,
        session_id=session_id,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time
    )


def create_agent_with_config(config: AgentConfig) -> ReasoningAgent:
    """
    Create agent with explicit configuration object.
    
    Args:
        config: Complete agent configuration
        
    Returns:
        Configured ReasoningAgent instance
    """
    return ReasoningAgent(config=config)