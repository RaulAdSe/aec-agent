"""
Clean agent memory system with TOON optimization.

Efficient memory management for building data and analysis results.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class MemoryEntry:
    """Single memory entry with timestamp."""
    timestamp: datetime = field(default_factory=datetime.now)
    entry_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentMemory:
    """
    Clean memory system for the AEC compliance agent.
    
    Stores building data, analysis results, and interaction history
    with efficient retrieval and TOON format support.
    """
    
    def __init__(self, max_entries: int = 100):
        """Initialize agent memory with optional size limit."""
        self.max_entries = max_entries
        self.logger = logging.getLogger(__name__)
        
        # Memory storage
        self.building_data_history: List[MemoryEntry] = []
        self.analysis_results: List[MemoryEntry] = []
        self.interaction_history: List[MemoryEntry] = []
        
        self.logger.info("AgentMemory initialized")
    
    def add_building_data(self, building_data: Dict[str, Any], 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store building data in memory.
        
        Args:
            building_data: Building data to store
            metadata: Optional metadata about the building data
        """
        entry = MemoryEntry(
            entry_type="building_data",
            data=building_data,
            metadata=metadata or {}
        )
        
        self.building_data_history.append(entry)
        self._trim_memory_if_needed()
        
        project_name = building_data.get("metadata", {}).get("project_name", "Unknown")
        self.logger.info(f"Stored building data for project: {project_name}")
    
    def add_analysis_result(self, result: Dict[str, Any],
                          analysis_type: str = "general",
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store analysis result in memory.
        
        Args:
            result: Analysis result to store
            analysis_type: Type of analysis performed
            metadata: Optional metadata about the analysis
        """
        entry = MemoryEntry(
            entry_type=analysis_type,
            data=result,
            metadata=metadata or {}
        )
        
        self.analysis_results.append(entry)
        self._trim_memory_if_needed()
        
        self.logger.info(f"Stored {analysis_type} analysis result")
    
    def add_interaction(self, interaction_data: Dict[str, Any],
                       interaction_type: str = "query") -> None:
        """
        Store user interaction in memory.
        
        Args:
            interaction_data: Interaction data to store
            interaction_type: Type of interaction
        """
        entry = MemoryEntry(
            entry_type=interaction_type,
            data=interaction_data,
            metadata={}
        )
        
        self.interaction_history.append(entry)
        self._trim_memory_if_needed()
        
        self.logger.debug(f"Stored {interaction_type} interaction")
    
    def get_latest_building_data(self) -> Optional[Dict[str, Any]]:
        """Get the most recently stored building data."""
        if self.building_data_history:
            return self.building_data_history[-1].data
        return None
    
    def get_latest_analysis(self, analysis_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent analysis result.
        
        Args:
            analysis_type: Optional filter by analysis type
            
        Returns:
            Latest analysis result or None
        """
        if not self.analysis_results:
            return None
        
        if analysis_type:
            # Filter by analysis type
            filtered_results = [
                entry for entry in self.analysis_results 
                if entry.entry_type == analysis_type
            ]
            if filtered_results:
                return filtered_results[-1].data
            return None
        
        return self.analysis_results[-1].data
    
    def get_building_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent building data history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent building data entries
        """
        recent_entries = self.building_data_history[-limit:]
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "data": entry.data,
                "metadata": entry.metadata
            }
            for entry in recent_entries
        ]
    
    def get_analysis_history(self, limit: int = 10,
                           analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent analysis history.
        
        Args:
            limit: Maximum number of entries to return
            analysis_type: Optional filter by analysis type
            
        Returns:
            List of recent analysis entries
        """
        results = self.analysis_results
        
        if analysis_type:
            results = [entry for entry in results if entry.entry_type == analysis_type]
        
        recent_entries = results[-limit:]
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "type": entry.entry_type,
                "data": entry.data,
                "metadata": entry.metadata
            }
            for entry in recent_entries
        ]
    
    def search_memory(self, query: str, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search memory entries by text query.
        
        Args:
            query: Search query
            entry_type: Optional filter by entry type
            
        Returns:
            List of matching entries
        """
        query_lower = query.lower()
        matching_entries = []
        
        # Search all memory types
        all_entries = (
            self.building_data_history + 
            self.analysis_results + 
            self.interaction_history
        )
        
        for entry in all_entries:
            if entry_type and entry.entry_type != entry_type:
                continue
            
            # Search in data and metadata
            entry_text = str(entry.data).lower() + str(entry.metadata).lower()
            
            if query_lower in entry_text:
                matching_entries.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "type": entry.entry_type,
                    "data": entry.data,
                    "metadata": entry.metadata
                })
        
        return matching_entries
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current memory state.
        
        Returns:
            Memory summary with statistics
        """
        latest_building = self.get_latest_building_data()
        latest_analysis = self.get_latest_analysis()
        
        summary = {
            "memory_stats": {
                "building_data_entries": len(self.building_data_history),
                "analysis_results": len(self.analysis_results),
                "interaction_history": len(self.interaction_history),
                "total_entries": (
                    len(self.building_data_history) + 
                    len(self.analysis_results) + 
                    len(self.interaction_history)
                )
            },
            "latest_building": {
                "project_name": latest_building.get("metadata", {}).get("project_name") if latest_building else None,
                "total_area": latest_building.get("metadata", {}).get("total_area") if latest_building else None,
                "levels": latest_building.get("metadata", {}).get("levels") if latest_building else None
            } if latest_building else None,
            "latest_analysis": {
                "timestamp": self.analysis_results[-1].timestamp.isoformat() if self.analysis_results else None,
                "type": self.analysis_results[-1].entry_type if self.analysis_results else None
            } if latest_analysis else None
        }
        
        return summary
    
    def clear(self) -> None:
        """Clear all memory entries."""
        self.building_data_history.clear()
        self.analysis_results.clear()
        self.interaction_history.clear()
        
        self.logger.info("Memory cleared")
    
    def _trim_memory_if_needed(self) -> None:
        """Trim memory if it exceeds the maximum size."""
        total_entries = (
            len(self.building_data_history) + 
            len(self.analysis_results) + 
            len(self.interaction_history)
        )
        
        if total_entries > self.max_entries:
            # Remove oldest entries first
            if self.interaction_history:
                self.interaction_history.pop(0)
            elif self.analysis_results:
                self.analysis_results.pop(0)
            elif self.building_data_history:
                self.building_data_history.pop(0)
            
            self.logger.debug("Trimmed memory to stay within limits")


# Example usage
if __name__ == "__main__":
    memory = AgentMemory()
    
    # Test adding building data
    sample_building = {
        "metadata": {
            "project_name": "Test Building",
            "total_area": 500.0
        },
        "rooms": [{"id": "R001", "area": 25.0}]
    }
    
    memory.add_building_data(sample_building)
    
    # Test adding analysis result
    sample_analysis = {
        "compliance_status": "COMPLIANT",
        "issues_found": 0
    }
    
    memory.add_analysis_result(sample_analysis, "fire_safety")
    
    # Get summary
    summary = memory.get_summary()
    print(f"Memory summary: {summary}")