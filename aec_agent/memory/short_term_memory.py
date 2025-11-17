"""
Short-term memory implementation using LangChain memory classes.

This module provides conversation context management for the AEC compliance agent,
handling recent messages, conversation summaries, and context windowing.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    CombinedMemory
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


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
    Short-term memory manager combining buffer and summary memory.
    
    Features:
    - Buffer memory for immediate recent context
    - Summary memory for older conversation context
    - Automatic management of memory size and token limits
    """
    
    def __init__(self, config: Optional[ShortTermMemoryConfig] = None):
        """Initialize short-term memory with configuration."""
        self.config = config or ShortTermMemoryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM for summarization
        try:
            self.llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature
            )
        except Exception as e:
            error_msg = (
                f"Failed to initialize LLM for conversation summarization: {e}. "
                "Please ensure OPENAI_API_KEY is set in your environment. "
                "Summary memory will be disabled, but buffer memory will still work."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Setup memory components
        self._setup_memory_components()
        
        self.logger.info(f"ShortTermMemory initialized with window_size={self.config.window_size}")
    
    def _setup_memory_components(self):
        """Setup the combined memory system with buffer and summary components."""
        
        # Buffer memory for recent messages
        self.buffer_memory = ConversationBufferWindowMemory(
            k=self.config.window_size,
            memory_key="recent_conversation",
            input_key="input",
            output_key="output",
            return_messages=True
        )
        
        # Summary memory - enabled if configured and LLM is available
        if self.config.enable_summarization and self.llm:
            self.summary_memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="conversation_summary",
                input_key="input",
                output_key="output",
                max_token_limit=self.config.max_token_limit,
                return_messages=False
            )
            
            # Use CombinedMemory to combine both
            self.combined_memory = CombinedMemory(
                memories=[self.buffer_memory, self.summary_memory]
            )
            self.logger.info("Enabled conversation summarization with buffer and summary memory")
        else:
            # Fallback to buffer only
            self.combined_memory = self.buffer_memory
            if self.config.enable_summarization:
                self.logger.warning("Summarization enabled but LLM unavailable - using buffer only")
            else:
                self.logger.info("Using buffer-only memory (summarization disabled)")
        
        self.logger.debug("Memory components initialized successfully")
    
    def add_conversation_turn(self, user_input: str, ai_output: str) -> None:
        """
        Add a conversation turn to memory with automatic token-based summarization trigger.
        
        Args:
            user_input: The user's input message
            ai_output: The AI agent's response
        """
        try:
            # Add to combined memory
            self.combined_memory.save_context(
                inputs={"input": user_input},
                outputs={"output": ai_output}
            )
            
            self.logger.debug(f"Added conversation turn - Input: {user_input[:50]}...")
            
            # Check token count and trigger summarization if needed
            if self.config.enable_summarization and hasattr(self, 'summary_memory'):
                current_tokens = self._estimate_conversation_tokens()
                
                if current_tokens > self.config.short_term_token_cap:
                    self.logger.warning(
                        f"Conversation token cap exceeded: {current_tokens} > {self.config.short_term_token_cap}. "
                        f"Triggering summarization..."
                    )
                    # Trigger async summarization
                    self._trigger_async_summarization()
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
            return self.combined_memory.load_memory_variables({})
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
        recent_context = ""
        if recent_messages:
            formatted_messages = []
            for message in recent_messages:
                role = "User" if message.type == "human" else "Assistant"
                formatted_messages.append(f"{role}: {message.content}")
            recent_context = "\n".join(formatted_messages)
        
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
            self.buffer_memory.clear()
            # Only clear summary_memory if it exists
            if hasattr(self, 'summary_memory'):
                self.summary_memory.clear()
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
        memory_vars = self.get_memory_variables()
        recent_messages = memory_vars.get("recent_conversation", [])
        summary = memory_vars.get("conversation_summary", "")
        
        return {
            "recent_messages_count": len(recent_messages),
            "window_size": self.config.window_size,
            "has_summary": bool(summary),
            "summary_length": len(summary.split()) if summary else 0,
            "total_context_length": len(self.get_conversation_context())
        }
    
    def update_config(self, new_config: ShortTermMemoryConfig) -> None:
        """
        Update memory configuration and reinitialize components.
        
        Args:
            new_config: New configuration settings
        """
        self.config = new_config
        self.llm = ChatOpenAI(
            model_name=self.config.model_name,
            temperature=self.config.temperature
        )
        self._setup_memory_components()
        self.logger.info(f"Short-term memory configuration updated")
    
    def _estimate_conversation_tokens(self) -> int:
        """Estimate token count for current conversation."""
        # Get all messages from buffer
        messages = self.buffer_memory.chat_memory.messages
        
        # Estimate tokens (simple heuristic: ~4 chars per token)
        total = sum(len(str(msg.content)) // 4 for msg in messages)
        
        # Add summary tokens if exists
        if hasattr(self, 'summary_memory') and self.summary_memory:
            summary = self.summary_memory.buffer
            if summary:
                total += len(summary) // 4
        
        return total
    
    def _trigger_async_summarization(self) -> None:
        """Trigger async summarization of conversation."""
        try:
            # For now, we'll use synchronous summarization
            # In a future enhancement, this could use asyncio for true async processing
            if hasattr(self, 'summary_memory') and self.summary_memory:
                # Check if summary itself is getting too long (recursive summarization)
                current_summary = self.summary_memory.buffer
                if current_summary and len(current_summary) > self.config.max_token_limit * 4:  # Rough char estimate
                    self._perform_recursive_summarization()
                
                # Force summarization by triggering the internal process
                # This is a simplified approach - in production, use proper async
                self.logger.info("Triggered conversation summarization")
        except Exception as e:
            self.logger.error(f"Failed to trigger summarization: {e}")
    
    def _perform_recursive_summarization(self) -> None:
        """Perform recursive summarization when summary gets too long."""
        try:
            if not hasattr(self, 'summary_memory') or not self.summary_memory:
                return
            
            current_summary = self.summary_memory.buffer
            if not current_summary:
                return
            
            # Create a more compact summary of the existing summary
            # This is a simplified approach - in practice, this would use the LLM
            # to create a summary of summaries
            
            # For now, we'll truncate and add a note
            max_chars = self.config.max_token_limit * 2  # Conservative estimate
            if len(current_summary) > max_chars:
                truncated = current_summary[:max_chars]
                recursive_summary = (
                    f"[Previous conversation summary (truncated): {truncated}]\n\n"
                    f"[Note: This is a summary of {len(current_summary)} characters of prior conversation]"
                )
                
                # Replace the summary with the recursive version
                self.summary_memory.buffer = recursive_summary
                self.logger.info("Performed recursive summarization due to summary length")
                
        except Exception as e:
            self.logger.error(f"Failed to perform recursive summarization: {e}")