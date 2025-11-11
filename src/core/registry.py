"""Central registry for agents and tools."""

from typing import Dict, Type, Any, List
from abc import ABC, abstractmethod


class AgentRegistry:
    """Registry for managing agents and tools."""
    
    def __init__(self):
        self._agents: Dict[str, Type] = {}
        self._tools: Dict[str, Type] = {}
    
    def register_agent(self, name: str, agent_class: Type) -> None:
        """Register an agent class."""
        self._agents[name] = agent_class
    
    def register_tool(self, name: str, tool_class: Type) -> None:
        """Register a tool class."""
        self._tools[name] = tool_class
    
    def get_agent(self, name: str) -> Type:
        """Get an agent class by name."""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found in registry")
        return self._agents[name]
    
    def get_tool(self, name: str) -> Type:
        """Get a tool class by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


# Global registry instance
registry = AgentRegistry()