"""
Short-term memory implementation with simple conversation management.

This module provides conversation context management for the AEC compliance agent,
handling recent messages, conversation summaries, and context windowing.
Compatible with current LangChain versions.
"""

import logging
from typing import Any, Dict, List, Optional
from collections import deque
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    user_input: str
    ai_output: str
    timestamp: str


class ShortTermMemoryConfig(BaseModel):
    """Configuration for short-term memory settings."""
    
    window_size: int = Field(default=10, description="Number of recent messages to keep in buffer")
    max_token_limit: int = Field(default=2000, description="Maximum tokens for summary memory")
    model_name: str = Field(default="gpt-4o-mini", description="Model for summarization")
    temperature: float = Field(default=0.1, description="Temperature for summarization")
    
    # Summarization settings
    enable_summarization: bool = Field(default=True, description="Enable conversation summarization")
    summarization_strategy: str = Field(default="async", description="Summarization strategy: sync, async, background")
    summarization_batch_size: int = Field(default=5, description="Batch size for summarization")
    
    # Token-based automatic triggers
    short_term_token_cap: int = Field(default=4000, description="Trigger summarization when conversation exceeds this token count")
    short_term_token_warning_threshold: int = Field(default=3000, description="Warning level before token cap")


class ShortTermMemory:
    """
    Short-term memory manager with simple conversation buffer and summary.
    
    Features:
    - Buffer memory for immediate recent context
    - Summary memory for older conversation context  
    - Automatic management of memory size and token limits
    """
    
    def __init__(self, config: Optional[ShortTermMemoryConfig] = None):
        """Initialize short-term memory with configuration."""
        self.config = config or ShortTermMemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize conversation buffer
        self.conversation_buffer = deque(maxlen=self.config.window_size)
        self.conversation_summary = ""
        
        # Initialize LLM for summarization if enabled
        self.llm = None
        if self.config.enable_summarization:
            try:
                self.llm = ChatOpenAI(
                    model=self.config.model_name,
                    temperature=self.config.temperature
                )
                self.logger.info("LLM initialized for conversation summarization")
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM for summarization: {e}")
        
        self.logger.info(f"ShortTermMemory initialized with window_size={self.config.window_size}")
    
    def add_conversation_turn(self, user_input: str, ai_output: str) -> None:
        """
        Add a conversation turn to memory with automatic token-based summarization trigger.
        
        Args:
            user_input: The user's input message
            ai_output: The AI agent's response
        """
        try:
            from datetime import datetime
            
            turn = ConversationTurn(
                user_input=user_input,
                ai_output=ai_output,
                timestamp=datetime.now().isoformat()
            )
            
            self.conversation_buffer.append(turn)
            
            self.logger.debug(f"Added conversation turn - Input: {user_input[:50]}...")
            
            # Check token count and trigger summarization if needed
            if self.config.enable_summarization and self.llm:
                current_tokens = self._estimate_conversation_tokens()
                
                if current_tokens > self.config.short_term_token_cap:
                    self.logger.warning(
                        f"Conversation token cap exceeded: {current_tokens} > {self.config.short_term_token_cap}. "
                        f"Triggering summarization..."
                    )
                    self._trigger_summarization()
                elif current_tokens > self.config.short_term_token_warning_threshold:
                    self.logger.info(
                        f"Conversation approaching token cap: {current_tokens}/{self.config.short_term_token_cap}"
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation turn: {e}")
            raise
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Get all memory variables for prompt injection.
        
        Returns:
            Dictionary with recent_conversation and conversation_summary
        """
        try:
            # Format recent conversation as messages
            recent_messages = []
            for turn in self.conversation_buffer:
                recent_messages.append(f"User: {turn.user_input}")
                recent_messages.append(f"Assistant: {turn.ai_output}")
            
            return {
                "recent_conversation": recent_messages,
                "conversation_summary": self.conversation_summary
            }
        except Exception as e:
            self.logger.error(f"Failed to load memory variables: {e}")
            return {"recent_conversation": [], "conversation_summary": ""}
    
    def get_conversation_context(self) -> str:
        """
        Get formatted conversation context for prompt injection.
        
        Returns:
            Formatted string with recent conversation and summary
        """
        memory_vars = self.get_memory_variables()
        
        # Format recent conversation
        recent_messages = memory_vars.get("recent_conversation", [])
        recent_context = "\n".join(recent_messages) if recent_messages else ""
        
        # Get conversation summary
        summary = memory_vars.get("conversation_summary", "")
        
        # Combine contexts
        context_parts = []
        if summary:
            context_parts.append(f"Previous Conversation Summary:\n{summary}")
        if recent_context:
            context_parts.append(f"Recent Conversation:\n{recent_context}")
        
        return "\n\n".join(context_parts)
    
    def clear_memory(self) -> None:
        """Clear all memory components."""
        try:
            self.conversation_buffer.clear()
            self.conversation_summary = ""
            self.logger.info("Short-term memory cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            raise
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        recent_messages = len(self.conversation_buffer)
        summary_length = len(self.conversation_summary.split()) if self.conversation_summary else 0
        
        return {
            "recent_messages_count": recent_messages,
            "window_size": self.config.window_size,
            "has_summary": bool(self.conversation_summary),
            "summary_length": summary_length,
            "total_context_length": len(self.get_conversation_context())
        }
    
    def update_config(self, new_config: ShortTermMemoryConfig) -> None:
        """
        Update memory configuration and reinitialize components.
        
        Args:
            new_config: New configuration settings
        """
        self.config = new_config
        
        # Update buffer size
        new_buffer = deque(self.conversation_buffer, maxlen=new_config.window_size)
        self.conversation_buffer = new_buffer
        
        # Reinitialize LLM if needed
        if new_config.enable_summarization and not self.llm:
            try:
                self.llm = ChatOpenAI(
                    model=new_config.model_name,
                    temperature=new_config.temperature
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM: {e}")
        
        self.logger.info("Short-term memory configuration updated")
    
    def _estimate_conversation_tokens(self) -> int:
        """Estimate token count for current conversation."""
        # Get all conversation content
        total_chars = 0
        for turn in self.conversation_buffer:
            total_chars += len(turn.user_input) + len(turn.ai_output)
        
        # Add summary if exists
        if self.conversation_summary:
            total_chars += len(self.conversation_summary)
        
        # Estimate tokens (simple heuristic: ~4 chars per token)
        return total_chars // 4
    
    def _trigger_summarization(self) -> None:
        """Trigger summarization of conversation."""
        if not self.llm:
            self.logger.warning("Cannot summarize: no LLM available")
            return
        
        try:
            # Get conversation for summarization
            conversation_text = []
            for turn in self.conversation_buffer:
                conversation_text.append(f"User: {turn.user_input}")
                conversation_text.append(f"Assistant: {turn.ai_output}")
            
            if not conversation_text:
                return
            
            # Create summarization prompt
            content = "\n".join(conversation_text)
            prompt = f"""Please provide a concise summary of this conversation, focusing on key decisions, 
            important information, and any ongoing context that would be useful for future interactions:

            {content}

            Summary:"""
            
            # Generate summary
            result = self.llm.invoke(prompt)
            new_summary = result.content.strip()
            
            # Update summary (append to existing if any)
            if self.conversation_summary:
                self.conversation_summary = f"{self.conversation_summary}\n\nAdditional context: {new_summary}"
            else:
                self.conversation_summary = new_summary
            
            # Clear buffer to make room
            self.conversation_buffer.clear()
            
            self.logger.info("Conversation summarization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger summarization: {e}")
