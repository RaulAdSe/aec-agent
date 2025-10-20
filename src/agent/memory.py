"""
Memory management for the ReAct agent.

This module implements a sliding window memory system that maintains a fixed
number of messages and automatically summarizes old messages when the limit is reached.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json


class SlidingWindowMemory:
    """
    Sliding window memory with automatic summarization.
    
    Maintains a fixed number of messages (window_size). When a new message
    would exceed the limit, the oldest messages are summarized into a single
    summary message, keeping the total count at window_size.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1
    ):
        """
        Initialize the sliding window memory.
        
        Args:
            window_size: Maximum number of messages to keep (default: 5)
            model_name: LLM model for summarization
            temperature: Temperature for summarization
        """
        self.window_size = window_size
        self.messages: List[BaseMessage] = []
        self.summary_messages: List[BaseMessage] = []  # Store summarized messages
        
        # Initialize LLM for summarization
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=1000
        )
    
    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to memory, handling window size and summarization.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        
        # Check if we need to summarize
        if len(self.messages) > self.window_size:
            self._summarize_oldest_messages()
    
    def get_messages(self) -> List[BaseMessage]:
        """
        Get all current messages in memory.
        
        Returns:
            List of messages in chronological order
        """
        return self.messages.copy()
    
    def get_messages_for_llm(self) -> List[BaseMessage]:
        """
        Get messages formatted for LLM consumption.
        
        Returns:
            List of messages including summaries
        """
        # Combine summary messages and current messages
        all_messages = []
        
        # Add summary messages first (if any)
        if self.summary_messages:
            summary_content = self._format_summary_messages()
            all_messages.append(SystemMessage(content=summary_content))
        
        # Add current messages
        all_messages.extend(self.messages)
        
        return all_messages
    
    def _summarize_oldest_messages(self) -> None:
        """
        Summarize the oldest messages when window size is exceeded.
        
        When we have window_size + 1 messages, we summarize the first
        window_size messages into a single summary message, then keep
        the newest message.
        """
        if len(self.messages) <= self.window_size:
            return
        
        # Get the oldest messages to summarize (all except the last one)
        messages_to_summarize = self.messages[:-1]
        newest_message = self.messages[-1]
        
        # Create summary
        summary = self._create_summary(messages_to_summarize)
        
        # Store the summary
        summary_message = SystemMessage(content=summary)
        self.summary_messages.append(summary_message)
        
        # Keep only the newest message
        self.messages = [newest_message]
    
    def _create_summary(self, messages: List[BaseMessage]) -> str:
        """
        Create a summary of the given messages.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            Summary string
        """
        if not messages:
            return ""
        
        # Format messages for summarization
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_messages.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                formatted_messages.append(f"System: {msg.content}")
            else:
                formatted_messages.append(f"Other: {msg.content}")
        
        conversation_text = "\n".join(formatted_messages)
        
        # Create summarization prompt
        summary_prompt = f"""Please summarize the following conversation between a user and an AEC compliance verification agent. 
Focus on:
1. The main compliance tasks or questions asked
2. Key findings or results from compliance checks
3. Important decisions or conclusions made
4. Any compliance issues identified

Keep the summary concise but comprehensive. This summary will be used to maintain context in a sliding window memory system.

Conversation to summarize:
{conversation_text}

Summary:"""
        
        try:
            # Get summary from LLM
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            return f"[Previous conversation summary: {response.content}]"
        except Exception as e:
            # Fallback to simple summary if LLM fails
            return f"[Previous conversation summary: {len(messages)} messages about AEC compliance verification]"
    
    def _format_summary_messages(self) -> str:
        """
        Format all summary messages into a single string.
        
        Returns:
            Formatted summary string
        """
        if not self.summary_messages:
            return ""
        
        summaries = []
        for i, summary_msg in enumerate(self.summary_messages, 1):
            summaries.append(f"Summary {i}: {summary_msg.content}")
        
        return "\n\n".join(summaries)
    
    def clear(self) -> None:
        """Clear all messages and summaries."""
        self.messages = []
        self.summary_messages = []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "current_messages": len(self.messages),
            "summary_messages": len(self.summary_messages),
            "window_size": self.window_size,
            "total_context": len(self.messages) + len(self.summary_messages),
            "memory_usage": f"{len(self.messages)}/{self.window_size} messages"
        }
    
    def export_memory(self) -> Dict[str, Any]:
        """
        Export memory state for persistence.
        
        Returns:
            Dictionary containing memory state
        """
        return {
            "messages": [{"type": msg.__class__.__name__, "content": msg.content} for msg in self.messages],
            "summary_messages": [{"type": msg.__class__.__name__, "content": msg.content} for msg in self.summary_messages],
            "window_size": self.window_size
        }
    
    def import_memory(self, memory_data: Dict[str, Any]) -> None:
        """
        Import memory state from persistence.
        
        Args:
            memory_data: Dictionary containing memory state
        """
        self.window_size = memory_data.get("window_size", 5)
        
        # Reconstruct messages
        self.messages = []
        for msg_data in memory_data.get("messages", []):
            msg_type = msg_data["type"]
            content = msg_data["content"]
            
            if msg_type == "HumanMessage":
                self.messages.append(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                self.messages.append(AIMessage(content=content))
            elif msg_type == "SystemMessage":
                self.messages.append(SystemMessage(content=content))
        
        # Reconstruct summary messages
        self.summary_messages = []
        for msg_data in memory_data.get("summary_messages", []):
            msg_type = msg_data["type"]
            content = msg_data["content"]
            
            if msg_type == "SystemMessage":
                self.summary_messages.append(SystemMessage(content=content))


class MemoryManager:
    """
    High-level memory manager for the agent.
    
    Provides convenient methods for managing memory across agent sessions.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the memory manager.
        
        Args:
            window_size: Size of the sliding window
        """
        self.memory = SlidingWindowMemory(window_size=window_size)
        self.session_id: Optional[str] = None
    
    def start_session(self, session_id: str) -> None:
        """
        Start a new session with the given ID.
        
        Args:
            session_id: Unique identifier for the session
        """
        self.session_id = session_id
        self.memory.clear()
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to memory.
        
        Args:
            content: User message content
        """
        self.memory.add_message(HumanMessage(content=content))
    
    def add_agent_message(self, content: str) -> None:
        """
        Add an agent message to memory.
        
        Args:
            content: Agent message content
        """
        self.memory.add_message(AIMessage(content=content))
    
    def get_context_for_llm(self) -> List[BaseMessage]:
        """
        Get formatted context for LLM consumption.
        
        Returns:
            List of messages with summaries
        """
        return self.memory.get_messages_for_llm()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get information about current memory state.
        
        Returns:
            Dictionary with memory information
        """
        stats = self.memory.get_memory_stats()
        stats["session_id"] = self.session_id
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test the sliding window memory
    memory = SlidingWindowMemory(window_size=5)
    
    # Add some test messages
    for i in range(7):
        if i % 2 == 0:
            memory.add_message(HumanMessage(content=f"User message {i}"))
        else:
            memory.add_message(AIMessage(content=f"Agent response {i}"))
        
        print(f"After message {i}: {memory.get_memory_stats()}")
    
    print("\nFinal messages:")
    for msg in memory.get_messages():
        print(f"- {msg.__class__.__name__}: {msg.content}")
    
    print("\nSummary messages:")
    for msg in memory.summary_messages:
        print(f"- {msg.__class__.__name__}: {msg.content}")
