"""
Legacy conversation history module.

This module has been replaced by the new memory system components:
- ShortTermMemory: For conversation context management
- SessionMemory: For structured session state
- MemoryManager: For unified memory coordination

For new implementations, use the MemoryManager class instead.
"""

import warnings
from typing import Dict, Any, Optional

from .memory_manager import MemoryManager


def deprecated_conversation_history_warning():
    """Issue deprecation warning for legacy conversation history usage."""
    warnings.warn(
        "conversation_history module is deprecated. Use MemoryManager instead.",
        DeprecationWarning,
        stacklevel=2
    )


class LegacyConversationHistory:
    """
    Legacy conversation history class - deprecated.
    
    Use MemoryManager instead for full memory capabilities.
    """
    
    def __init__(self):
        deprecated_conversation_history_warning()
        self.memory_manager = MemoryManager()
    
    def add_turn(self, user_input: str, ai_response: str) -> None:
        """Add conversation turn - delegates to MemoryManager."""
        self.memory_manager.add_conversation_turn(user_input, ai_response)
    
    def get_history(self) -> str:
        """Get conversation history - delegates to MemoryManager."""
        return self.memory_manager.get_conversation_context()
    
    def clear(self) -> None:
        """Clear conversation history - delegates to MemoryManager."""
        self.memory_manager.clear_conversation_memory()


# Legacy function for backward compatibility
def create_conversation_history() -> LegacyConversationHistory:
    """Create legacy conversation history - deprecated."""
    deprecated_conversation_history_warning()
    return LegacyConversationHistory()