"""
Configuration for memory summarization system.

This module defines comprehensive configuration for all memory layers
including short-term, session, and execution memory summarization settings.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SummarizationConfig:
    """Configuration for memory summarization."""
    
    # Enable/disable summarization
    enable_short_term_summarization: bool = True
    enable_session_compaction: bool = True
    # Note: Execution memory compaction disabled - it resets per task
    
    # Summarization strategy
    short_term_strategy: str = "async"  # sync, async, background
    session_strategy: str = "rule_based"  # rule_based, llm, hybrid
    
    # Model settings (can be overridden by AgentConfig)
    summarization_model: Optional[str] = None  # None = use from AgentConfig.llm.summarization_model
    summarization_temperature: Optional[float] = None  # None = use from AgentConfig.llm.summarization_temperature
    summarization_max_tokens: int = 500
    
    # Retention settings
    short_term_buffer_size: int = 10
    short_term_summary_token_limit: int = 2000
    
    # Session memory compaction settings
    session_tool_history_keep_recent: int = 20
    session_subtask_keep_active: bool = True
    session_subtask_keep_failed: int = 5
    session_active_files_limit: int = 10
    session_modified_files_limit: int = 20
    session_context_max_length: int = 2000
    session_accumulated_context_limit: int = 50
    
    # Compaction triggers - Token-based (Primary)
    # Note: These caps are conservative to leave room for prompts, tool outputs, and responses.
    # Model context windows are large (200K+), but memory should be a small fraction.
    session_compaction_auto: bool = True
    session_token_cap: int = 12000  # Trigger compaction when session memory exceeds this token count
    session_token_warning_threshold: int = 10000  # Warning level before cap
    
    # Compaction triggers - Count-based (Secondary/Backup)
    session_compaction_interval: int = 50  # Compact every N operations (backup trigger)
    session_compaction_threshold_tools: int = 30  # Compact when tool_history exceeds this
    session_compaction_threshold_subtasks: int = 20  # Compact when completed subtasks exceed this
    
    # Short-term memory token limits
    # Conservative limit to keep conversation history manageable
    short_term_token_cap: int = 4000  # Trigger summarization when conversation exceeds this
    short_term_token_warning_threshold: int = 3000  # Warning level before cap
    
    # Completed goals token limit
    # Keep completed goal summaries compact
    completed_goals_token_cap: int = 6000  # Compact completed_goals when summaries exceed this
    
    # Compression settings
    enable_progressive_compression: bool = True
    compression_age_days: List[int] = None  # Will default to [1, 7, 30]
    compression_levels: List[str] = None   # Will default to ["light", "medium", "heavy"]
    
    # Performance settings
    enable_summarization_cache: bool = True
    cache_ttl_hours: int = 24
    enable_batch_summarization: bool = True
    batch_size: int = 5
    async_queue_size: int = 100
    
    def __post_init__(self):
        """Set default values for list fields."""
        if self.compression_age_days is None:
            self.compression_age_days = [1, 7, 30]
        if self.compression_levels is None:
            self.compression_levels = ["light", "medium", "heavy"]