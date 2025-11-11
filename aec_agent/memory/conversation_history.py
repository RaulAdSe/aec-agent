"""Conversation history and short-term memory management."""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """Single memory entry with timestamp."""
    timestamp: datetime = field(default_factory=datetime.now)
    entry_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class ConversationHistory:
    """
    Manages conversation history and short-term memory.
    
    Stores building data, analysis results, and interaction history
    with efficient retrieval capabilities.
    """
    
    def __init__(self, max_entries: int = 100):
        """Initialize conversation history."""
        self.max_entries = max_entries
        self.logger = logging.getLogger(__name__)
        self.entries: List[MemoryEntry] = []
    
    def add_building_data(self, building_data: Dict[str, Any]) -> None:
        """Store building data in memory."""
        entry = MemoryEntry(
            entry_type="building_data",
            data=building_data
        )
        self.entries.append(entry)
        self._trim_if_needed()
        
        project_name = building_data.get("metadata", {}).get("project_name", "Unknown")
        self.logger.info(f"Stored building data for project: {project_name}")
    
    def add_analysis_result(self, result: Dict[str, Any]) -> None:
        """Store analysis result in memory."""
        entry = MemoryEntry(
            entry_type="analysis_result",
            data=result
        )
        self.entries.append(entry)
        self._trim_if_needed()
        
        self.logger.info("Stored analysis result")
    
    def add_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Store user interaction in memory."""
        entry = MemoryEntry(
            entry_type="interaction",
            data=interaction_data
        )
        self.entries.append(entry)
        self._trim_if_needed()
    
    def get_latest_building_data(self) -> Optional[Dict[str, Any]]:
        """Get the most recently stored building data."""
        for entry in reversed(self.entries):
            if entry.entry_type == "building_data":
                return entry.data
        return None
    
    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis result."""
        for entry in reversed(self.entries):
            if entry.entry_type == "analysis_result":
                return entry.data
        return None
    
    def get_history(self, entry_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent history entries.
        
        Args:
            entry_type: Optional filter by entry type
            limit: Maximum number of entries to return
            
        Returns:
            List of recent entries
        """
        filtered_entries = self.entries
        
        if entry_type:
            filtered_entries = [e for e in self.entries if e.entry_type == entry_type]
        
        recent_entries = filtered_entries[-limit:]
        
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "type": entry.entry_type,
                "data": entry.data
            }
            for entry in recent_entries
        ]
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search memory entries by text query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching entries
        """
        query_lower = query.lower()
        matching_entries = []
        
        for entry in self.entries:
            entry_text = str(entry.data).lower()
            
            if query_lower in entry_text:
                matching_entries.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "type": entry.entry_type,
                    "data": entry.data
                })
        
        return matching_entries
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory summary with statistics."""
        entry_types = {}
        for entry in self.entries:
            entry_types[entry.entry_type] = entry_types.get(entry.entry_type, 0) + 1
        
        latest_building = self.get_latest_building_data()
        
        return {
            "total_entries": len(self.entries),
            "entry_types": entry_types,
            "latest_building": {
                "project_name": latest_building.get("metadata", {}).get("project_name") if latest_building else None,
                "timestamp": self.entries[-1].timestamp.isoformat() if self.entries else None
            } if latest_building else None
        }
    
    def clear(self) -> None:
        """Clear all memory entries."""
        self.entries.clear()
        self.logger.info("Memory cleared")
    
    def _trim_if_needed(self) -> None:
        """Trim memory if it exceeds maximum size."""
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
            self.logger.debug("Trimmed memory to stay within limits")