"""
Unified Agent Configuration System - Central configuration for all agent parameters.

This module provides a comprehensive configuration system that centralizes all agent
settings in one place, making it easy to configure and tune the entire system.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class AgentProfile(Enum):
    """Predefined configuration profiles for different environments."""
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"
    TESTING = "testing"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


@dataclass
class LLMConfig:
    """Large Language Model configuration section."""
    
    # === PRIMARY MODEL SELECTION ===
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-5-mini"  # Primary model for reasoning
    fallback_model: Optional[str] = "gpt-3.5-turbo"  # Fallback if primary fails
    
    # === COMPONENT-SPECIFIC MODEL SELECTION ===
    # All components use the primary model by default, but can be overridden
    goal_decomposer_model: Optional[str] = None  # None = use model_name
    tool_planner_model: Optional[str] = None     # None = use model_name
    validator_model: Optional[str] = None         # None = use model_name
    replanner_model: Optional[str] = None       # None = use model_name
    progress_evaluator_model: Optional[str] = None  # None = use model_name
    summarization_model: Optional[str] = None   # None = use model_name (for memory)
    
    # === MODEL PARAMETERS ===
    temperature: float = 0.1  # Lower = more deterministic (0.0-2.0)
    max_tokens: int = 8192    # Maximum tokens per response
    top_p: float = 1.0        # Nucleus sampling parameter
    frequency_penalty: float = 0.0  # Penalize frequent tokens
    presence_penalty: float = 0.0   # Penalize present tokens
    
    # === COMPONENT-SPECIFIC PARAMETERS ===
    # Component-specific temperature overrides (None = use default temperature)
    goal_decomposer_temperature: Optional[float] = None  # Default: 0.1
    tool_planner_temperature: Optional[float] = None    # Default: 0.1
    validator_temperature: Optional[float] = None        # Default: 0.1
    replanner_temperature: Optional[float] = None       # Default: 0.2 (slightly higher for creativity)
    progress_evaluator_temperature: Optional[float] = None  # Default: 0.1
    summarization_temperature: Optional[float] = None    # Default: 0.1
    
    # Component-specific max_tokens overrides
    goal_decomposer_max_tokens: int = 2000
    tool_planner_max_tokens: int = 1000
    validator_max_tokens: int = 1000
    replanner_max_tokens: int = 2000
    progress_evaluator_max_tokens: int = 1500
    summarization_max_tokens: int = 500
    
    # === CONTEXT MANAGEMENT ===
    context_window: int = 128000  # Model's context window size
    max_context_usage: float = 0.8  # Use max 80% of context window
    context_buffer: int = 1000    # Reserve tokens for response
    
    # === API CONFIGURATION ===
    api_timeout: float = 60.0     # Request timeout in seconds
    max_concurrent_calls: int = 5  # Max parallel LLM calls
    rate_limit_rpm: int = 3500    # Requests per minute limit
    
    # === RETRY CONFIGURATION ===
    max_retries: int = 3          # Max retry attempts per call
    retry_delay: float = 1.0      # Base delay between retries (seconds)
    exponential_backoff: bool = True  # Use exponential backoff
    timeout_per_call: float = 30.0   # Timeout per individual call
    
    # === HELPER METHODS ===
    def get_component_model(self, component: str) -> str:
        """Get the model name for a specific component."""
        model_map = {
            "goal_decomposer": self.goal_decomposer_model or self.model_name,
            "tool_planner": self.tool_planner_model or self.model_name,
            "validator": self.validator_model or self.model_name,
            "replanner": self.replanner_model or self.model_name,
            "progress_evaluator": self.progress_evaluator_model or self.model_name,
            "summarization": self.summarization_model or self.model_name,
        }
        return model_map.get(component, self.model_name)
    
    def get_component_temperature(self, component: str) -> float:
        """Get the temperature for a specific component."""
        temp_map = {
            "goal_decomposer": self.goal_decomposer_temperature if self.goal_decomposer_temperature is not None else self.temperature,
            "tool_planner": self.tool_planner_temperature if self.tool_planner_temperature is not None else self.temperature,
            "validator": self.validator_temperature if self.validator_temperature is not None else self.temperature,
            "replanner": self.replanner_temperature if self.replanner_temperature is not None else 0.2,  # Default 0.2 for creativity
            "progress_evaluator": self.progress_evaluator_temperature if self.progress_evaluator_temperature is not None else self.temperature,
            "summarization": self.summarization_temperature if self.summarization_temperature is not None else self.temperature,
        }
        return temp_map.get(component, self.temperature)
    
    def get_component_max_tokens(self, component: str) -> int:
        """Get the max_tokens for a specific component."""
        tokens_map = {
            "goal_decomposer": self.goal_decomposer_max_tokens,
            "tool_planner": self.tool_planner_max_tokens,
            "validator": self.validator_max_tokens,
            "replanner": self.replanner_max_tokens,
            "progress_evaluator": self.progress_evaluator_max_tokens,
            "summarization": self.summarization_max_tokens,
        }
        return tokens_map.get(component, self.max_tokens)


@dataclass
class ReasoningConfig:
    """Reasoning and execution configuration section."""
    
    # === REASONING LIMITS ===
    max_iterations: int = 20           # Max reasoning loop iterations
    max_execution_time: float = 300.0  # Max total execution time (seconds)
    max_task_depth: int = 5            # Max recursive task depth
    max_parallel_tasks: int = 3        # Max tasks executed in parallel
    
    # === GOAL DECOMPOSITION ===
    max_subtasks_per_goal: int = 10    # Max subtasks from goal decomposition
    goal_complexity_threshold: float = 0.7  # Threshold for complex goals
    enable_adaptive_planning: bool = True    # Allow plan modifications during execution
    
    # === TASK EXECUTION ===
    task_timeout: float = 60.0         # Timeout per individual task
    enable_task_retry: bool = True     # Retry failed tasks
    max_task_retries: int = 2          # Max retries per task
    task_retry_delay: float = 5.0      # Delay between task retries
    
    # === VALIDATION & MONITORING ===
    enable_progress_tracking: bool = True   # Track execution progress
    progress_check_interval: int = 5        # Check progress every N steps
    enable_result_validation: bool = True   # Validate task results
    validation_strictness: float = 0.8     # Validation threshold (0.0-1.0)


@dataclass
class GuardrailConfig:
    """External guardrails and safety limits configuration section."""
    
    # === LLM GUARDRAILS ===
    llm_max_retries: int = 3          # Max LLM call retries
    llm_retry_delay: float = 1.0      # Base retry delay (seconds)
    llm_timeout: float = 30.0         # Timeout per LLM call
    
    # === EXECUTION GUARDRAILS ===
    max_replanning_events: int = 5    # Max replanning events per session
    max_same_task_attempts: int = 3   # Max attempts for same task
    max_total_execution_steps: int = 50  # Max total execution steps
    
    # === MEMORY GUARDRAILS ===
    max_execution_steps_memory: int = 100   # Max steps stored in memory
    max_context_summary_length: int = 4000  # Max context summary chars
    enable_auto_cleanup: bool = True         # Enable automatic memory cleanup
    cleanup_threshold: float = 0.9          # Trigger cleanup at 90% capacity
    
    # === MONITORING & ALERTS ===
    enable_guardrail_monitoring: bool = True  # Enable guardrail monitoring
    alert_threshold: float = 0.8              # Alert when 80% of limits reached
    enable_proactive_alerts: bool = True      # Send alerts before limits hit


@dataclass
class MemoryConfig:
    """Memory and persistence configuration section."""
    
    # === MEMORY TYPES ===
    enable_short_term_memory: bool = True    # Enable working memory
    enable_long_term_memory: bool = True     # Enable persistent memory
    enable_episodic_memory: bool = True      # Enable session memory
    
    # === MEMORY LIMITS ===
    short_term_capacity: int = 1000          # Max items in short-term memory
    long_term_capacity: int = 10000          # Max items in long-term memory  
    episodic_session_limit: int = 100        # Max sessions in episodic memory
    
    # === MEMORY PERSISTENCE ===
    enable_persistence: bool = True          # Save memory to disk
    persistence_interval: int = 30           # Save every N seconds
    memory_file_path: Optional[str] = None   # Custom memory file path
    compression_enabled: bool = True         # Compress saved memory
    
    # === MEMORY OPTIMIZATION ===
    enable_memory_compression: bool = True   # Compress old memories
    compression_threshold_days: int = 7      # Compress memories older than N days
    enable_forgetting: bool = False          # Enable memory forgetting
    forgetting_rate: float = 0.01           # Forgetting rate per day


@dataclass
class ToolConfig:
    """Tool execution and management configuration section."""
    
    # === TOOL EXECUTION ===
    tool_timeout: float = 30.0               # Timeout per tool call
    max_parallel_tools: int = 3              # Max parallel tool executions
    enable_tool_retry: bool = True           # Retry failed tool calls
    tool_retry_attempts: int = 2             # Max retries per tool
    
    # === TOOL DISCOVERY ===
    auto_discover_tools: bool = True         # Automatically discover available tools
    tool_discovery_paths: List[str] = field( # Paths to search for tools
        default_factory=lambda: ["aec_agent/tools"]
    )
    
    # === TOOL VALIDATION ===
    validate_tool_inputs: bool = True        # Validate tool inputs before execution
    validate_tool_outputs: bool = True       # Validate tool outputs after execution
    tool_safety_checks: bool = True          # Enable tool safety checks
    
    # === BUILDING DATA TOOLS ===
    ifc_processing_timeout: float = 60.0     # Timeout for IFC file processing
    max_element_query_results: int = 1000    # Max elements returned per query
    enable_geometry_extraction: bool = True  # Extract geometry data from IFC
    cache_building_data: bool = True         # Cache loaded building data


@dataclass
class PerformanceConfig:
    """Performance tuning and optimization configuration section."""
    
    # === CONCURRENCY ===
    max_concurrent_operations: int = 5       # Max concurrent operations
    enable_parallel_execution: bool = True   # Enable parallel task execution
    worker_pool_size: int = 4               # Size of worker thread pool
    
    # === CACHING ===
    enable_response_caching: bool = True     # Cache LLM responses
    cache_ttl_seconds: int = 3600           # Cache time-to-live
    max_cache_size_mb: int = 100            # Max cache size in MB
    cache_compression: bool = True           # Compress cached data
    
    # === OPTIMIZATION ===
    enable_request_batching: bool = False    # Batch multiple requests
    batch_size: int = 5                     # Requests per batch
    enable_smart_routing: bool = True        # Route requests to optimal models
    
    # === RESOURCE LIMITS ===
    max_memory_usage_mb: int = 2048         # Max memory usage
    enable_memory_monitoring: bool = True   # Monitor memory usage
    gc_threshold: float = 0.8               # Trigger garbage collection at 80%


@dataclass
class LoggingConfig:
    """Logging and debugging configuration section."""
    
    # === LOG LEVELS ===
    log_level: str = "INFO"                 # Global log level (DEBUG/INFO/WARNING/ERROR)
    llm_log_level: str = "INFO"             # LLM-specific log level
    tool_log_level: str = "INFO"            # Tool-specific log level
    
    # === LOG OUTPUTS ===
    log_to_console: bool = True             # Log to console
    log_to_file: bool = True                # Log to file
    log_file_path: Optional[str] = None     # Custom log file path
    log_rotation: bool = True               # Rotate log files
    max_log_size_mb: int = 100             # Max log file size
    
    # === DEBUG OPTIONS ===
    debug_mode: bool = False                # Enable debug mode
    trace_llm_calls: bool = False           # Trace all LLM calls
    trace_tool_calls: bool = False          # Trace all tool calls
    save_debug_artifacts: bool = False      # Save debug files
    
    # === LANGSMITH TRACING ===
    enable_langsmith: bool = True           # Enable LangSmith tracing
    langsmith_project: str = "AEC-Reasoning-Agent"  # LangSmith project name
    trace_all_chains: bool = True           # Trace all LangChain operations


@dataclass
class SecurityConfig:
    """Security and safety configuration section."""
    
    # === API SECURITY ===
    require_api_keys: bool = True           # Require API keys for external services
    encrypt_api_keys: bool = False          # Encrypt stored API keys
    api_key_rotation_days: int = 90         # Rotate API keys every N days
    
    # === INPUT VALIDATION ===
    sanitize_inputs: bool = True            # Sanitize all user inputs
    max_input_length: int = 10000          # Max characters per input
    block_suspicious_patterns: bool = True  # Block known malicious patterns
    
    # === OUTPUT FILTERING ===
    filter_sensitive_data: bool = True     # Filter sensitive data from outputs
    enable_content_filtering: bool = True   # Filter inappropriate content
    
    # === AUDIT & COMPLIANCE ===
    enable_audit_logging: bool = True      # Log all operations for audit
    audit_log_retention_days: int = 365    # Keep audit logs for N days
    compliance_mode: bool = False          # Enable strict compliance mode


@dataclass
class AgentConfig:
    """
    Unified configuration for the entire AEC reasoning agent system.
    
    This is the central configuration class that contains all settings
    organized into logical sections for easy management.
    
    Example usage:
    
    # Use default configuration
    config = AgentConfig()
    
    # Create configuration for development
    config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
    
    # Load from environment variables
    config = AgentConfig.from_env()
    
    # Custom configuration
    config = AgentConfig(
        llm=LLMConfig(model_name="gpt-4", temperature=0.0),
        reasoning=ReasoningConfig(max_iterations=10),
        guardrails=GuardrailConfig(max_replanning_events=3)
    )
    """
    
    # === CORE CONFIGURATION SECTIONS ===
    llm: LLMConfig = field(default_factory=LLMConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # === GLOBAL SETTINGS ===
    profile: AgentProfile = AgentProfile.DEVELOPMENT
    session_id: Optional[str] = None
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=lambda: Path.cwd() / "data")
    
    # === API KEYS ===
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.langsmith_api_key:
            self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    
    @classmethod
    def for_profile(cls, profile: AgentProfile) -> 'AgentConfig':
        """
        Create configuration optimized for a specific profile.
        
        Examples:
        
        # Development profile - verbose logging, relaxed limits
        dev_config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
        
        # Production profile - minimal logging, strict limits  
        prod_config = AgentConfig.for_profile(AgentProfile.PRODUCTION)
        
        # Testing profile - fast execution, minimal retries
        test_config = AgentConfig.for_profile(AgentProfile.TESTING)
        """
        if profile == AgentProfile.DEVELOPMENT:
            return cls._development_config()
        elif profile == AgentProfile.STAGING:
            return cls._staging_config()
        elif profile == AgentProfile.PRODUCTION:
            return cls._production_config()
        elif profile == AgentProfile.TESTING:
            return cls._testing_config()
        else:
            return cls()
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """
        Create configuration from environment variables.
        
        Environment variables follow the pattern: AEC_<SECTION>_<PARAMETER>
        
        Examples:
        export AEC_LLM_MODEL_NAME="gpt-4"
        export AEC_LLM_TEMPERATURE="0.0"
        export AEC_REASONING_MAX_ITERATIONS="15"
        export AEC_GUARDRAILS_MAX_REPLANNING="3"
        export AEC_MEMORY_ENABLE_PERSISTENCE="false"
        export AEC_PERFORMANCE_MAX_CONCURRENT="10"
        """
        config = cls()
        
        # === LLM CONFIGURATION ===
        config.llm.model_name = os.getenv("AEC_LLM_MODEL_NAME", config.llm.model_name)
        config.llm.temperature = float(os.getenv("AEC_LLM_TEMPERATURE", str(config.llm.temperature)))
        config.llm.max_tokens = int(os.getenv("AEC_LLM_MAX_TOKENS", str(config.llm.max_tokens)))
        config.llm.max_retries = int(os.getenv("AEC_LLM_MAX_RETRIES", str(config.llm.max_retries)))
        config.llm.timeout_per_call = float(os.getenv("AEC_LLM_TIMEOUT", str(config.llm.timeout_per_call)))
        
        # === REASONING CONFIGURATION ===
        config.reasoning.max_iterations = int(os.getenv("AEC_REASONING_MAX_ITERATIONS", str(config.reasoning.max_iterations)))
        config.reasoning.max_execution_time = float(os.getenv("AEC_REASONING_MAX_EXECUTION_TIME", str(config.reasoning.max_execution_time)))
        config.reasoning.max_parallel_tasks = int(os.getenv("AEC_REASONING_MAX_PARALLEL", str(config.reasoning.max_parallel_tasks)))
        
        # === GUARDRAIL CONFIGURATION ===
        config.guardrails.max_replanning_events = int(os.getenv("AEC_GUARDRAILS_MAX_REPLANNING", str(config.guardrails.max_replanning_events)))
        config.guardrails.max_same_task_attempts = int(os.getenv("AEC_GUARDRAILS_MAX_TASK_ATTEMPTS", str(config.guardrails.max_same_task_attempts)))
        config.guardrails.max_total_execution_steps = int(os.getenv("AEC_GUARDRAILS_MAX_EXECUTION_STEPS", str(config.guardrails.max_total_execution_steps)))
        config.guardrails.max_execution_steps_memory = int(os.getenv("AEC_GUARDRAILS_MAX_MEMORY_STEPS", str(config.guardrails.max_execution_steps_memory)))
        
        # === MEMORY CONFIGURATION ===
        config.memory.enable_persistence = os.getenv("AEC_MEMORY_ENABLE_PERSISTENCE", "true").lower() == "true"
        config.memory.short_term_capacity = int(os.getenv("AEC_MEMORY_SHORT_TERM_CAPACITY", str(config.memory.short_term_capacity)))
        
        # === PERFORMANCE CONFIGURATION ===
        config.performance.max_concurrent_operations = int(os.getenv("AEC_PERFORMANCE_MAX_CONCURRENT", str(config.performance.max_concurrent_operations)))
        config.performance.enable_response_caching = os.getenv("AEC_PERFORMANCE_ENABLE_CACHING", "true").lower() == "true"
        
        # === LOGGING CONFIGURATION ===
        config.logging.log_level = os.getenv("AEC_LOG_LEVEL", config.logging.log_level)
        config.logging.debug_mode = os.getenv("AEC_DEBUG_MODE", "false").lower() == "true"
        config.logging.enable_langsmith = os.getenv("AEC_LANGSMITH_ENABLED", "true").lower() == "true"
        
        return config
    
    @classmethod
    def _development_config(cls) -> 'AgentConfig':
        """Development profile: Verbose logging, relaxed limits, debugging enabled."""
        return cls(
            profile=AgentProfile.DEVELOPMENT,
            llm=LLMConfig(
                model_name="gpt-5-mini",  # Faster, cheaper model for development
                temperature=0.1,
                max_retries=5,             # More retries for debugging
            ),
            reasoning=ReasoningConfig(
                max_iterations=30,         # Higher limits for experimentation
                max_execution_time=600.0,  # 10 minutes
                enable_adaptive_planning=True,
            ),
            guardrails=GuardrailConfig(
                max_replanning_events=10,  # More replanning allowed
                max_total_execution_steps=100,
            ),
            logging=LoggingConfig(
                log_level="DEBUG",         # Verbose logging
                debug_mode=True,
                trace_llm_calls=True,
                trace_tool_calls=True,
            ),
            performance=PerformanceConfig(
                enable_response_caching=True,  # Cache for faster development
                max_concurrent_operations=3,   # Conservative concurrency
            )
        )
    
    @classmethod
    def _production_config(cls) -> 'AgentConfig':
        """Production profile: Minimal logging, strict limits, optimized performance."""
        return cls(
            profile=AgentProfile.PRODUCTION,
            llm=LLMConfig(
                model_name="gpt-4",        # Most capable model
                temperature=0.05,          # Low temperature for consistency
                max_retries=3,
            ),
            reasoning=ReasoningConfig(
                max_iterations=15,         # Stricter limits
                max_execution_time=180.0,  # 3 minutes
                enable_adaptive_planning=True,
            ),
            guardrails=GuardrailConfig(
                max_replanning_events=3,   # Conservative replanning
                max_total_execution_steps=25,
                enable_guardrail_monitoring=True,
            ),
            logging=LoggingConfig(
                log_level="WARNING",       # Minimal logging
                debug_mode=False,
                trace_llm_calls=False,
            ),
            performance=PerformanceConfig(
                enable_response_caching=True,
                max_concurrent_operations=10,  # Higher concurrency
                enable_parallel_execution=True,
            ),
            security=SecurityConfig(
                audit_log_retention_days=365,  # Long retention for compliance
                compliance_mode=True,
            )
        )
    
    @classmethod
    def _staging_config(cls) -> 'AgentConfig':
        """Staging profile: Balanced settings for testing production scenarios."""
        return cls(
            profile=AgentProfile.STAGING,
            llm=LLMConfig(
                model_name="gpt-5-mini",
                temperature=0.1,
                max_retries=3,
            ),
            reasoning=ReasoningConfig(
                max_iterations=20,
                max_execution_time=240.0,  # 4 minutes
            ),
            guardrails=GuardrailConfig(
                max_replanning_events=5,
                max_total_execution_steps=50,
            ),
            logging=LoggingConfig(
                log_level="INFO",
                debug_mode=False,
                enable_langsmith=True,
            ),
            performance=PerformanceConfig(
                enable_response_caching=True,
                max_concurrent_operations=5,
            )
        )
    
    @classmethod
    def _testing_config(cls) -> 'AgentConfig':
        """Testing profile: Fast execution, minimal retries, comprehensive logging."""
        return cls(
            profile=AgentProfile.TESTING,
            llm=LLMConfig(
                model_name="gpt-3.5-turbo",  # Fast model for tests
                temperature=0.0,             # Deterministic responses
                max_retries=1,               # Quick failures
                timeout_per_call=10.0,       # Short timeouts
            ),
            reasoning=ReasoningConfig(
                max_iterations=5,            # Quick execution
                max_execution_time=30.0,     # 30 seconds
                task_timeout=10.0,
            ),
            guardrails=GuardrailConfig(
                max_replanning_events=2,
                max_total_execution_steps=10,
                max_execution_steps_memory=20,
            ),
            logging=LoggingConfig(
                log_level="DEBUG",           # Full logging for test debugging
                debug_mode=True,
                save_debug_artifacts=True,
            ),
            performance=PerformanceConfig(
                enable_response_caching=False,  # No caching in tests
                max_concurrent_operations=1,    # Sequential execution
            )
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'profile': self.profile.value,
            'llm': self.llm.__dict__,
            'reasoning': self.reasoning.__dict__,
            'guardrails': self.guardrails.__dict__,
            'memory': self.memory.__dict__,
            'tools': self.tools.__dict__,
            'performance': self.performance.__dict__,
            'logging': self.logging.__dict__,
            'security': self.security.__dict__,
            'session_id': self.session_id,
            'project_root': str(self.project_root),
            'data_dir': str(self.data_dir)
        }
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of validation errors.
        
        Returns:
            List of error messages. Empty list if configuration is valid.
        """
        errors = []
        
        # Validate LLM configuration
        if self.llm.temperature < 0.0 or self.llm.temperature > 2.0:
            errors.append("LLM temperature must be between 0.0 and 2.0")
        
        if self.llm.max_tokens < 1:
            errors.append("LLM max_tokens must be positive")
        
        # Validate reasoning configuration
        if self.reasoning.max_iterations < 1:
            errors.append("Reasoning max_iterations must be positive")
        
        if self.reasoning.max_execution_time <= 0:
            errors.append("Reasoning max_execution_time must be positive")
        
        # Validate guardrail configuration
        if self.guardrails.max_replanning_events < 1:
            errors.append("Guardrails max_replanning_events must be positive")
        
        # Validate API keys if required
        if self.security.require_api_keys:
            if not self.openai_api_key and self.llm.provider == LLMProvider.OPENAI:
                errors.append("OpenAI API key is required but not provided")
        
        return errors
    
    def get_effective_guardrail_config(self):
        """Get guardrail configuration compatible with existing GuardrailConfig."""
        from .core.llm_guardrails import GuardrailConfig as LegacyGuardrailConfig
        
        return LegacyGuardrailConfig(
            llm_max_retries=self.llm.max_retries,
            llm_retry_delay=self.llm.retry_delay,
            llm_timeout=self.llm.timeout_per_call,
            max_replanning_events=self.guardrails.max_replanning_events,
            max_same_task_attempts=self.guardrails.max_same_task_attempts,
            max_total_execution_steps=self.guardrails.max_total_execution_steps,
            max_execution_steps_memory=self.guardrails.max_execution_steps_memory,
            max_context_summary_length=self.guardrails.max_context_summary_length
        )


# === CONFIGURATION EXAMPLES ===

def get_example_configs() -> Dict[str, AgentConfig]:
    """
    Get example configurations for common use cases.
    
    Returns:
        Dictionary of example configurations with descriptive names.
    """
    return {
        
        # === BASIC EXAMPLES ===
        
        "minimal": AgentConfig(
            llm=LLMConfig(model_name="gpt-3.5-turbo", max_tokens=1024),
            reasoning=ReasoningConfig(max_iterations=5),
            guardrails=GuardrailConfig(max_total_execution_steps=10)
        ),
        
        "balanced": AgentConfig(
            llm=LLMConfig(model_name="gpt-5-mini", temperature=0.1),
            reasoning=ReasoningConfig(max_iterations=15, max_execution_time=180.0),
            guardrails=GuardrailConfig(max_replanning_events=5)
        ),
        
        "high_performance": AgentConfig(
            llm=LLMConfig(model_name="gpt-4", max_concurrent_calls=10),
            reasoning=ReasoningConfig(max_parallel_tasks=5),
            performance=PerformanceConfig(
                max_concurrent_operations=10,
                enable_parallel_execution=True,
                enable_response_caching=True
            )
        ),
        
        # === SPECIALIZED EXAMPLES ===
        
        "careful_execution": AgentConfig(
            reasoning=ReasoningConfig(
                validation_strictness=0.9,
                enable_result_validation=True,
                max_task_retries=3
            ),
            guardrails=GuardrailConfig(
                max_replanning_events=3,
                max_same_task_attempts=2,
                enable_guardrail_monitoring=True
            )
        ),
        
        "memory_optimized": AgentConfig(
            memory=MemoryConfig(
                enable_memory_compression=True,
                compression_threshold_days=1,
                short_term_capacity=500
            ),
            guardrails=GuardrailConfig(
                max_execution_steps_memory=50,
                enable_auto_cleanup=True
            )
        ),
        
        "debug_mode": AgentConfig(
            logging=LoggingConfig(
                log_level="DEBUG",
                debug_mode=True,
                trace_llm_calls=True,
                trace_tool_calls=True,
                save_debug_artifacts=True
            ),
            llm=LLMConfig(temperature=0.0),  # Deterministic for debugging
            guardrails=GuardrailConfig(max_total_execution_steps=100)
        )
    }


# === DEFAULT CONFIGURATION INSTANCE ===

# Global default configuration instance
default_config = AgentConfig()

# Environment-based configuration (preferred)
try:
    env_config = AgentConfig.from_env()
except Exception:
    env_config = default_config