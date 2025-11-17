"""
Centralized summarization service for memory management.

This module provides LLM-based and rule-based summarization capabilities
for different types of memory content including conversations, tool executions,
subtasks, and other structured data.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .summarization_config import SummarizationConfig


logger = logging.getLogger(__name__)


class SummaryCache(BaseModel):
    """Cache entry for summarization results."""
    content_hash: str
    summary: str
    created_at: datetime
    ttl_hours: int = 24
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age_hours = (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600
        return age_hours > self.ttl_hours


class SummarizationService:
    """
    Centralized summarization service for all memory layers.
    
    Provides both LLM-based and rule-based summarization with caching,
    importance scoring, and async processing capabilities.
    """
    
    def __init__(self, config: SummarizationConfig):
        """Initialize the summarization service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, SummaryCache] = {}
        self._llm: Optional[ChatOpenAI] = None
        
        # Initialize LLM if needed
        if self.config.enable_short_term_summarization:
            try:
                self._llm = ChatOpenAI(
                    model=self.config.summarization_model,
                    temperature=self.config.summarization_temperature,
                    max_tokens=self.config.summarization_max_tokens
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM for summarization: {e}")
                self._llm = None
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content caching."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_summary(self, content: str) -> Optional[str]:
        """Get cached summary if available and not expired."""
        if not self.config.enable_summarization_cache:
            return None
        
        content_hash = self._get_content_hash(content)
        if content_hash in self._cache:
            cache_entry = self._cache[content_hash]
            if not cache_entry.is_expired():
                return cache_entry.summary
            else:
                # Remove expired entry
                del self._cache[content_hash]
        return None
    
    def _cache_summary(self, content: str, summary: str) -> None:
        """Cache a summarization result."""
        if not self.config.enable_summarization_cache:
            return
        
        content_hash = self._get_content_hash(content)
        self._cache[content_hash] = SummaryCache(
            content_hash=content_hash,
            summary=summary,
            created_at=datetime.now(timezone.utc),
            ttl_hours=self.config.cache_ttl_hours
        )
    
    async def summarize_conversation_async(self, messages: List[str]) -> str:
        """
        Asynchronously summarize conversation messages using LLM.
        
        Args:
            messages: List of conversation messages to summarize
            
        Returns:
            Summary of the conversation
        """
        if not self._llm or not messages:
            return ""
        
        # Combine messages for summarization
        content = "\n".join(messages)
        
        # Check cache first
        cached = self._get_cached_summary(content)
        if cached:
            return cached
        
        try:
            # Create summarization prompt
            prompt = f"""Please provide a concise summary of this conversation, focusing on key decisions, 
            important information, and any ongoing context that would be useful for future interactions:

            {content}

            Summary:"""
            
            # Run LLM asynchronously
            result = await self._llm.ainvoke(prompt)
            summary = result.content.strip()
            
            # Cache the result
            self._cache_summary(content, summary)
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to summarize conversation: {e}")
            return f"[Summary unavailable due to error: {str(e)[:100]}]"
    
    def summarize_conversation_sync(self, messages: List[str]) -> str:
        """
        Synchronously summarize conversation messages using LLM.
        
        Args:
            messages: List of conversation messages to summarize
            
        Returns:
            Summary of the conversation
        """
        if not self._llm or not messages:
            return ""
        
        # Combine messages for summarization
        content = "\n".join(messages)
        
        # Check cache first
        cached = self._get_cached_summary(content)
        if cached:
            return cached
        
        try:
            # Create summarization prompt
            prompt = f"""Please provide a concise summary of this conversation, focusing on key decisions, 
            important information, and any ongoing context that would be useful for future interactions:

            {content}

            Summary:"""
            
            result = self._llm.invoke(prompt)
            summary = result.content.strip()
            
            # Cache the result
            self._cache_summary(content, summary)
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to summarize conversation: {e}")
            return f"[Summary unavailable due to error: {str(e)[:100]}]"
    
    def summarize_tool_history(self, executions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a rule-based statistical summary of tool executions.
        
        Args:
            executions: List of tool execution dictionaries
            
        Returns:
            Statistical summary of tool history
        """
        if not executions:
            return {}
        
        # Group by tool name
        by_tool = {}
        for exec_dict in executions:
            tool = exec_dict.get("tool_name", "unknown")
            if tool not in by_tool:
                by_tool[tool] = []
            by_tool[tool].append(exec_dict)
        
        # Calculate statistics
        total = len(executions)
        successful = sum(1 for e in executions if e.get("success", True))
        success_rate = successful / total if total > 0 else 0.0
        
        # Tool usage counts
        tool_counts = {tool: len(execs) for tool, execs in by_tool.items()}
        
        # Error patterns
        errors = [e.get("error_message", "") for e in executions if e.get("error_message")]
        error_patterns = {}
        for error in errors:
            # Extract error type (first part before colon or common patterns)
            error_type = error.split(':')[0] if ':' in error else error[:50]
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        # Time range
        times = [e.get("execution_time") for e in executions if e.get("execution_time")]
        time_range = {}
        if times:
            # Convert to datetime if they're strings
            parsed_times = []
            for t in times:
                if isinstance(t, str):
                    try:
                        parsed_times.append(datetime.fromisoformat(t.replace('Z', '+00:00')))
                    except:
                        continue
                elif isinstance(t, datetime):
                    parsed_times.append(t)
            
            if parsed_times:
                time_range = {
                    "first": min(parsed_times).isoformat(),
                    "last": max(parsed_times).isoformat()
                }
        
        return {
            "_type": "tool_history_summary",
            "total_executions": total,
            "success_rate": success_rate,
            "tool_usage": tool_counts,
            "error_patterns": error_patterns,
            "time_range": time_range,
            "summary_created_at": datetime.now(timezone.utc).isoformat()
        }
    
    def summarize_subtasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a theme-based summary of subtasks.
        
        Args:
            tasks: List of subtask dictionaries
            
        Returns:
            Theme-based summary of subtasks
        """
        if not tasks:
            return {}
        
        # Group by common patterns in task names
        # Extract key themes (e.g., "Load", "Analyze", "Check")
        task_themes = {}
        for task in tasks:
            name = task.get("name", "")
            # Simple heuristic: first word or common prefix
            first_word = name.split()[0] if name else "Other"
            if first_word not in task_themes:
                task_themes[first_word] = []
            task_themes[first_word].append(name)
        
        # Calculate statistics
        total_tasks = len(tasks)
        status_counts = {}
        for task in tasks:
            status = task.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Theme counts
        theme_counts = {theme: len(names) for theme, names in task_themes.items()}
        
        # Time range
        times = []
        for task in tasks:
            if task.get("updated_at"):
                try:
                    if isinstance(task["updated_at"], str):
                        times.append(datetime.fromisoformat(task["updated_at"].replace('Z', '+00:00')))
                    elif isinstance(task["updated_at"], datetime):
                        times.append(task["updated_at"])
                except:
                    continue
        
        time_range = {}
        if times:
            time_range = {
                "first": min(times).isoformat(),
                "last": max(times).isoformat()
            }
        
        return {
            "_type": "subtask_summary",
            "total_tasks": total_tasks,
            "status_distribution": status_counts,
            "task_themes": theme_counts,
            "time_range": time_range,
            "summary_created_at": datetime.now(timezone.utc).isoformat()
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple heuristic: ~4 characters per token (conservative estimate).
        For more accuracy, could use tiktoken library.
        """
        return len(text) // 4
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """
        Summarize arbitrary text content.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        if len(text) <= max_length:
            return text
        
        # Simple truncation approach
        return text[:max_length] + "..."
    
    def clean_expired_cache(self) -> int:
        """
        Clean expired entries from cache.
        
        Returns:
            Number of expired entries removed
        """
        expired_keys = []
        for key, cache_entry in self._cache.items():
            if cache_entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        total_entries = len(self._cache)
        expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_count,
            "expired_entries": expired_count,
            "cache_enabled": self.config.enable_summarization_cache
        }