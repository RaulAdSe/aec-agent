# Configuration & Deployment Guide

## Overview

The AEC Compliance Agent uses a comprehensive unified configuration system that centralizes all agent parameters in a single, easy-to-manage structure. This system supports environment variables, configuration profiles, and complete customization for any deployment scenario.

## Configuration Architecture

### Unified Configuration Structure

```python
@dataclass
class AgentConfig:
    llm: LLMConfig              # LLM model and API settings
    reasoning: ReasoningConfig  # Reasoning limits and behavior  
    guardrails: GuardrailConfig # Safety limits and monitoring
    memory: MemoryConfig        # Memory management settings
    tools: ToolConfig           # Tool execution configuration
    performance: PerformanceConfig  # Concurrency and optimization
    logging: LoggingConfig      # Logging and tracing settings
    security: SecurityConfig    # Security and audit configuration
```

### Configuration Areas

#### 1. LLM Configuration (`LLMConfig`)
Controls the language model behavior and API integration:

```python
@dataclass
class LLMConfig:
    # Model Selection
    model_name: str = "gpt-4o-mini"  # Primary model
    fallback_model: str = "gpt-3.5-turbo"  # Fallback if primary fails
    provider: str = "openai"  # LLM provider
    
    # Model Parameters
    temperature: float = 0.1  # Creativity (0.0-2.0, lower = more deterministic)
    max_tokens: int = 8192   # Maximum tokens per response
    top_p: float = 1.0       # Nucleus sampling parameter
    frequency_penalty: float = 0.0  # Reduce repetition
    
    # API Configuration
    timeout_per_call: float = 30.0  # Timeout per API call (seconds)
    max_retries: int = 3            # Maximum retry attempts
    retry_delay: float = 1.0        # Base delay between retries
    max_concurrent_calls: int = 5   # Concurrent API calls
    rate_limit_rpm: int = 3500      # Requests per minute limit
```

#### 2. Reasoning Configuration (`ReasoningConfig`)
Controls the agent's reasoning behavior and limits:

```python
@dataclass
class ReasoningConfig:
    # Execution Limits
    max_iterations: int = 20           # Maximum reasoning iterations
    max_execution_time: float = 300.0 # Maximum execution time (seconds)
    max_task_depth: int = 10          # Maximum subtask nesting
    
    # Goal Decomposition
    max_subtasks: int = 15            # Maximum subtasks per goal
    enable_adaptive_planning: bool = True  # Dynamic task adjustment
    complexity_threshold: float = 0.8     # Trigger task breakdown
    
    # Task Execution
    task_timeout: float = 60.0        # Individual task timeout
    max_task_retries: int = 2         # Retries per failed task
    enable_parallel_tasks: bool = True # Execute independent tasks in parallel
    max_parallel_tasks: int = 3       # Maximum concurrent tasks
    
    # Validation & Monitoring
    validation_strictness: float = 0.8    # Result validation threshold
    enable_progress_tracking: bool = True # Track goal progress
    progress_check_interval: int = 5      # Check progress every N iterations
```

#### 3. Guardrail Configuration (`GuardrailConfig`)
Defines safety limits and protective mechanisms:

```python
@dataclass
class GuardrailConfig:
    # LLM Guardrails
    max_llm_retries: int = 5               # Maximum LLM retry attempts
    llm_timeout: float = 45.0              # LLM call timeout
    enable_llm_fallback: bool = False      # No fallback (pure LLM)
    
    # Execution Guardrails
    max_replanning_events: int = 5         # Maximum replanning attempts
    max_task_attempts: int = 3             # Attempts per task
    max_total_execution_steps: int = 50    # Total execution steps limit
    execution_step_timeout: float = 30.0  # Individual step timeout
    
    # Memory Guardrails
    enable_memory_cleanup: bool = True     # Automatic memory management
    memory_cleanup_interval: int = 100    # Cleanup every N operations
    max_memory_context_length: int = 4000 # Context length limit
    
    # Monitoring & Alerts
    enable_guardrail_monitoring: bool = True     # Monitor guardrail hits
    alert_on_guardrail_hit: bool = False        # Alert on limit reached
    guardrail_hit_threshold: float = 0.8        # Alert threshold (80% of limit)
```

#### 4. Memory Configuration (`MemoryConfig`)
Controls memory management across all layers:

```python
@dataclass
class MemoryConfig:
    # Short-Term Memory (Conversation)
    short_term_buffer_size: int = 10           # Recent messages to keep
    short_term_token_cap: int = 4000          # Conversation token limit
    enable_short_term_summarization: bool = True  # Enable conversation summaries
    summarization_strategy: str = "async"     # sync, async, background
    
    # Session Memory (Structured State)  
    session_token_cap: int = 12000           # Session memory token limit
    session_token_warning_threshold: int = 10000  # Warning threshold
    enable_goal_based_archiving: bool = True # Goal lifecycle management
    tool_history_keep_recent: int = 20      # Recent tool executions to keep
    completed_goals_token_cap: int = 6000   # Archived goals token limit
    
    # Execution Memory (Per-Task)
    max_execution_steps: int = 100          # Maximum execution steps
    enable_context_deduplication: bool = True  # Remove duplicate context
    
    # Summarization Service
    summarization_model: str = "gpt-4o-mini"    # Model for summarization
    summarization_temperature: float = 0.1      # Low temperature for summaries
    enable_summarization_cache: bool = True     # Cache summaries
    cache_ttl_hours: int = 24                   # Cache time-to-live
    
    # Persistence
    enable_memory_persistence: bool = True     # Save memory to disk
    memory_save_interval: int = 50            # Save every N operations
    memory_file_path: str = "session_memory.json"  # Save file path
```

#### 5. Tool Configuration (`ToolConfig`)
Controls building analysis tool behavior:

```python
@dataclass 
class ToolConfig:
    # Tool Execution
    tool_timeout: float = 30.0              # Individual tool timeout
    max_tool_retries: int = 2               # Retry failed tools
    enable_parallel_tools: bool = False     # Parallel tool execution
    tool_retry_delay: float = 1.0          # Delay between retries
    
    # Tool Discovery
    enable_auto_discovery: bool = True      # Auto-discover available tools
    tool_search_paths: List[str] = None    # Additional tool search paths
    validate_tools_on_startup: bool = True # Validate tool availability
    
    # Building Data Tools
    ifc_processing_timeout: float = 60.0   # IFC file processing timeout
    max_element_query_limit: int = 10000   # Maximum elements per query
    enable_geometry_extraction: bool = True # Extract geometric data
    spatial_query_timeout: float = 30.0    # Spatial analysis timeout
```

#### 6. Performance Configuration (`PerformanceConfig`)
Controls performance optimization settings:

```python
@dataclass
class PerformanceConfig:
    # Concurrency
    max_concurrent_operations: int = 5      # Maximum concurrent operations
    enable_parallel_execution: bool = True # Enable parallel task execution
    worker_pool_size: int = 4              # Worker thread pool size
    
    # Caching
    enable_response_caching: bool = True    # Cache LLM responses
    cache_ttl_seconds: int = 3600          # Cache time-to-live
    max_cache_size_mb: int = 100           # Maximum cache size
    
    # Optimization
    enable_request_batching: bool = False   # Batch multiple requests
    batch_size: int = 5                    # Requests per batch
    enable_smart_routing: bool = True      # Intelligent request routing
    enable_resource_monitoring: bool = True # Monitor resource usage
```

#### 7. Logging Configuration (`LoggingConfig`)
Controls logging and observability:

```python
@dataclass
class LoggingConfig:
    # Log Levels
    log_level: str = "INFO"                # Global log level
    component_log_levels: Dict[str, str] = None  # Per-component levels
    
    # Log Outputs
    enable_console_logging: bool = True    # Log to console
    enable_file_logging: bool = False     # Log to file
    log_file_path: str = "agent.log"     # Log file path
    log_rotation_size_mb: int = 10        # Rotate log file size
    log_retention_days: int = 30          # Keep logs for N days
    
    # Debug & Tracing
    debug_mode: bool = False              # Enable debug mode
    trace_llm_calls: bool = False        # Trace LLM API calls
    trace_tool_calls: bool = False       # Trace tool executions
    save_debug_artifacts: bool = False    # Save debug information
    
    # LangSmith Integration
    langsmith_enabled: bool = True        # Enable LangSmith tracing
    langsmith_project: str = "AEC-Reasoning-Agent"  # LangSmith project
    langsmith_tags: List[str] = None     # Custom tags for traces
```

#### 8. Security Configuration (`SecurityConfig`)
Controls security and audit features:

```python
@dataclass
class SecurityConfig:
    # API Security
    api_key_rotation_days: int = 90       # Rotate API keys every N days
    enable_api_key_encryption: bool = False  # Encrypt stored API keys
    api_timeout_seconds: float = 30.0     # API call timeout
    
    # Input Validation
    max_input_length: int = 10000         # Maximum input character length
    enable_input_sanitization: bool = True # Sanitize user inputs
    blocked_patterns: List[str] = None    # Blocked input patterns
    
    # Output Filtering
    enable_output_filtering: bool = True   # Filter sensitive data
    sensitive_data_patterns: List[str] = None  # Patterns to filter
    enable_content_filtering: bool = False # Content filtering
    
    # Audit & Compliance
    enable_audit_logging: bool = False    # Enable audit logs
    audit_log_retention_days: int = 365  # Keep audit logs for N days
    compliance_mode: bool = False        # Enable compliance mode
    audit_sensitive_operations: bool = True  # Audit sensitive operations
```

## Configuration Profiles

### Pre-Configured Deployment Profiles

#### Development Profile
Optimized for local development and experimentation:

```python
DEVELOPMENT = AgentConfig(
    llm=LLMConfig(
        model_name="gpt-4o-mini",      # Cost-effective for development
        temperature=0.2,               # Slightly more creative
        max_retries=5,                 # More retries for debugging
        timeout_per_call=45.0          # Longer timeout for debugging
    ),
    reasoning=ReasoningConfig(
        max_iterations=50,             # Higher limits for experimentation
        max_execution_time=900.0,      # 15 minutes for complex analysis
        enable_adaptive_planning=True,  # Enable experimental features
        max_parallel_tasks=5           # Higher parallelism
    ),
    memory=MemoryConfig(
        short_term_token_cap=6000,     # Larger conversation buffer
        session_token_cap=15000,       # Larger session memory
        enable_summarization_cache=True # Cache for faster iteration
    ),
    logging=LoggingConfig(
        log_level="DEBUG",             # Detailed logging
        debug_mode=True,               # Debug mode enabled
        trace_llm_calls=True,          # Trace all LLM calls
        save_debug_artifacts=True      # Save debug information
    ),
    performance=PerformanceConfig(
        enable_response_caching=True,  # Cache for development speed
        max_concurrent_operations=10   # Higher concurrency for testing
    )
)
```

#### Production Profile  
Optimized for production deployment with strict limits:

```python
PRODUCTION = AgentConfig(
    llm=LLMConfig(
        model_name="gpt-4",           # Most capable model
        temperature=0.05,             # Very deterministic
        max_retries=3,                # Conservative retries
        timeout_per_call=20.0         # Shorter timeout for responsiveness
    ),
    reasoning=ReasoningConfig(
        max_iterations=15,            # Strict iteration limits
        max_execution_time=180.0,     # 3 minutes maximum
        validation_strictness=0.95,   # Very strict validation
        max_parallel_tasks=2          # Conservative parallelism
    ),
    guardrails=GuardrailConfig(
        enable_guardrail_monitoring=True,   # Monitor limits
        alert_on_guardrail_hit=True,        # Alert on limits
        max_replanning_events=3             # Strict replanning limit
    ),
    memory=MemoryConfig(
        session_token_cap=8000,       # Conservative memory limits
        enable_memory_persistence=True # Save state for recovery
    ),
    logging=LoggingConfig(
        log_level="WARNING",          # Minimal logging
        enable_file_logging=True,     # Persistent logs
        langsmith_enabled=True        # Production monitoring
    ),
    security=SecurityConfig(
        enable_audit_logging=True,    # Audit trail
        compliance_mode=True,         # Compliance features
        audit_log_retention_days=730  # 2-year retention
    )
)
```

#### Staging Profile
Balanced configuration for testing production scenarios:

```python
STAGING = AgentConfig(
    llm=LLMConfig(
        model_name="gpt-4o-mini",     # Balanced model choice
        temperature=0.1,              # Standard determinism
        max_retries=4                 # Moderate retries
    ),
    reasoning=ReasoningConfig(
        max_iterations=25,            # Moderate limits
        max_execution_time=240.0,     # 4 minutes
        max_parallel_tasks=3          # Balanced parallelism
    ),
    memory=MemoryConfig(
        session_token_cap=10000,      # Balanced memory limits
        enable_summarization_cache=True
    ),
    logging=LoggingConfig(
        log_level="INFO",             # Standard logging
        langsmith_enabled=True,       # Full tracing for testing
        debug_mode=False
    )
)
```

#### Testing Profile
Optimized for automated testing and CI/CD:

```python
TESTING = AgentConfig(
    llm=LLMConfig(
        model_name="gpt-3.5-turbo",   # Fast model for testing
        temperature=0.0,              # Maximum determinism
        max_retries=1,                # Minimal retries
        timeout_per_call=10.0         # Short timeout
    ),
    reasoning=ReasoningConfig(
        max_iterations=5,             # Minimal iterations
        max_execution_time=30.0,      # 30 seconds maximum
        enable_adaptive_planning=False, # Disable experimental features
        max_parallel_tasks=1          # No parallelism for determinism
    ),
    memory=MemoryConfig(
        enable_summarization_cache=False, # No cache for determinism
        enable_memory_persistence=False   # No disk persistence
    ),
    logging=LoggingConfig(
        log_level="DEBUG",            # Debug for test failures
        debug_mode=True,              # Debug mode for tests
        langsmith_enabled=False       # No external dependencies
    ),
    performance=PerformanceConfig(
        enable_response_caching=False, # No caching for determinism
        max_concurrent_operations=1    # Serial execution only
    )
)
```

## Environment Variable Configuration

### Complete Environment Variable Support

All configuration options can be set via environment variables using the `AEC_` prefix:

#### LLM Configuration
```bash
# Model selection
export AEC_LLM_MODEL_NAME="gpt-4"
export AEC_LLM_FALLBACK_MODEL="gpt-3.5-turbo"
export AEC_LLM_PROVIDER="openai"

# Model parameters  
export AEC_LLM_TEMPERATURE="0.05"
export AEC_LLM_MAX_TOKENS="4096"
export AEC_LLM_TOP_P="1.0"

# API settings
export AEC_LLM_TIMEOUT="30.0"
export AEC_LLM_MAX_RETRIES="5"
export AEC_LLM_MAX_CONCURRENT_CALLS="10"
export AEC_LLM_RATE_LIMIT_RPM="3000"
```

#### Reasoning Configuration
```bash
# Execution limits
export AEC_REASONING_MAX_ITERATIONS="30"
export AEC_REASONING_MAX_EXECUTION_TIME="600.0"
export AEC_REASONING_MAX_TASK_DEPTH="15"

# Task management
export AEC_REASONING_MAX_SUBTASKS="20"
export AEC_REASONING_ENABLE_ADAPTIVE_PLANNING="true"
export AEC_REASONING_TASK_TIMEOUT="90.0"
export AEC_REASONING_MAX_PARALLEL_TASKS="5"

# Validation
export AEC_REASONING_VALIDATION_STRICTNESS="0.9"
export AEC_REASONING_ENABLE_PROGRESS_TRACKING="true"
```

#### Memory Configuration
```bash
# Short-term memory
export AEC_MEMORY_SHORT_TERM_BUFFER_SIZE="15"
export AEC_MEMORY_SHORT_TERM_TOKEN_CAP="5000"
export AEC_MEMORY_ENABLE_SHORT_TERM_SUMMARIZATION="true"
export AEC_MEMORY_SUMMARIZATION_STRATEGY="async"

# Session memory
export AEC_MEMORY_SESSION_TOKEN_CAP="15000"
export AEC_MEMORY_ENABLE_GOAL_ARCHIVING="true"
export AEC_MEMORY_TOOL_HISTORY_KEEP_RECENT="25"

# Summarization
export AEC_MEMORY_SUMMARIZATION_MODEL="gpt-4o-mini"
export AEC_MEMORY_ENABLE_SUMMARIZATION_CACHE="true"
```

#### Performance Configuration
```bash
# Concurrency
export AEC_PERFORMANCE_MAX_CONCURRENT="10"
export AEC_PERFORMANCE_ENABLE_PARALLEL_EXECUTION="true"
export AEC_PERFORMANCE_WORKER_POOL_SIZE="6"

# Caching
export AEC_PERFORMANCE_ENABLE_CACHING="true"
export AEC_PERFORMANCE_CACHE_TTL_SECONDS="7200"
export AEC_PERFORMANCE_MAX_CACHE_SIZE_MB="200"
```

#### Logging Configuration
```bash
# Basic logging
export AEC_LOG_LEVEL="INFO"
export AEC_LOG_ENABLE_CONSOLE="true"
export AEC_LOG_ENABLE_FILE="true"
export AEC_LOG_FILE_PATH="/var/log/aec-agent.log"

# Debug settings
export AEC_DEBUG_MODE="false"
export AEC_TRACE_LLM_CALLS="false"
export AEC_TRACE_TOOL_CALLS="false"

# LangSmith
export AEC_LANGSMITH_ENABLED="true"
export AEC_LANGSMITH_PROJECT="AEC-Production"
```

### Loading Environment Configuration

```python
# Automatic environment loading
config = AgentConfig.from_env()
agent = ReasoningAgent(config=config)

# With profile base and environment overrides
config = AgentConfig.for_profile(AgentProfile.PRODUCTION)
config.apply_environment_overrides()  # Override with env vars
agent = ReasoningAgent(config=config)
```

## Usage Examples

### Basic Usage

```python
from aec_agent.config import AgentConfig, AgentProfile
from aec_agent.reasoning_agent import ReasoningAgent

# Use default configuration
agent = ReasoningAgent()

# Use predefined profile
config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
agent = ReasoningAgent(config=config)

# Load from environment variables
config = AgentConfig.from_env()
agent = ReasoningAgent(config=config)
```

### Custom Configuration

```python
from aec_agent.config import AgentConfig, LLMConfig, ReasoningConfig

# Create custom configuration
custom_config = AgentConfig(
    llm=LLMConfig(
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=4096,
        max_retries=5,
        timeout_per_call=45.0
    ),
    reasoning=ReasoningConfig(
        max_iterations=25,
        max_execution_time=300.0,
        validation_strictness=0.9,
        enable_adaptive_planning=True
    ),
    memory=MemoryConfig(
        session_token_cap=15000,
        enable_goal_based_archiving=True,
        summarization_strategy="async"
    )
)

agent = ReasoningAgent(config=custom_config)
```

### Environment-Specific Configurations

#### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11

# Set production environment variables
ENV AEC_LLM_MODEL_NAME="gpt-4"
ENV AEC_LLM_TEMPERATURE="0.05"
ENV AEC_REASONING_MAX_ITERATIONS="15"
ENV AEC_REASONING_MAX_EXECUTION_TIME="180.0"
ENV AEC_LOG_LEVEL="WARNING"
ENV AEC_SECURITY_ENABLE_AUDIT_LOGGING="true"
ENV AEC_SECURITY_COMPLIANCE_MODE="true"

# Application code
COPY . /app
WORKDIR /app
CMD ["python", "-m", "aec_agent.main"]
```

#### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aec-agent-config
data:
  # LLM Configuration
  AEC_LLM_MODEL_NAME: "gpt-4"
  AEC_LLM_TEMPERATURE: "0.05"
  AEC_LLM_MAX_RETRIES: "3"
  
  # Reasoning Configuration
  AEC_REASONING_MAX_ITERATIONS: "15"
  AEC_REASONING_MAX_EXECUTION_TIME: "180.0"
  AEC_REASONING_VALIDATION_STRICTNESS: "0.95"
  
  # Memory Configuration
  AEC_MEMORY_SESSION_TOKEN_CAP: "8000"
  AEC_MEMORY_ENABLE_PERSISTENCE: "true"
  
  # Logging Configuration
  AEC_LOG_LEVEL: "INFO"
  AEC_LANGSMITH_ENABLED: "true"
  AEC_LANGSMITH_PROJECT: "AEC-Production"
```

#### CI/CD Pipeline
```yaml
# .github/workflows/test.yml
name: Test AEC Agent
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    env:
      # Testing configuration
      AEC_LLM_MODEL_NAME: "gpt-3.5-turbo"
      AEC_LLM_TEMPERATURE: "0.0"
      AEC_REASONING_MAX_ITERATIONS: "5"
      AEC_REASONING_MAX_EXECUTION_TIME: "30.0"
      AEC_MEMORY_ENABLE_PERSISTENCE: "false"
      AEC_LOG_LEVEL: "DEBUG"
      AEC_LANGSMITH_ENABLED: "false"
      
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/ -v
```

## Configuration Tools

### Configuration Dashboard

View and validate current configuration:

```bash
# Show current configuration
python scripts/show_configuration.py

# Show specific sections
python scripts/show_configuration.py --section llm
python scripts/show_configuration.py --section memory
python scripts/show_configuration.py --section all

# Show environment variable mapping
python scripts/show_configuration.py --section environment

# Show available profiles
python scripts/show_configuration.py --section profiles
```

### Configuration Validation

```python
# Validate configuration
config = AgentConfig()
validation_errors = config.validate()

if validation_errors:
    print("Configuration errors found:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("Configuration is valid ✓")

# Validate specific sections
llm_errors = config.llm.validate()
memory_errors = config.memory.validate()
```

### Configuration Examples

```bash
# Run configuration examples
python examples/configuration_examples.py

# Examples include:
# - Development setup
# - Production deployment  
# - High-volume batch processing
# - Security-focused configuration
# - Performance optimization
```

## API Key Management

### Required API Keys

```bash
# OpenAI API key (required)
export OPENAI_API_KEY="sk-..."

# LangSmith API key (optional, for tracing)
export LANGSMITH_API_KEY="ls__..."

# Alternative LangChain API key format
export LANGCHAIN_API_KEY="ls__..."
```

### API Key Security

#### Production Best Practices
1. **Never commit API keys to code**
2. **Use environment variables or secure vaults**
3. **Rotate keys regularly** (every 90 days recommended)
4. **Monitor API usage** for anomalies
5. **Use least-privilege access** when possible

#### Key Rotation
```python
# Configuration supports key rotation
config = AgentConfig(
    security=SecurityConfig(
        api_key_rotation_days=90,
        enable_api_key_encryption=True
    )
)
```

## Deployment Patterns

### Single Instance Deployment

```python
# app.py
import os
from aec_agent.config import AgentConfig
from aec_agent.reasoning_agent import ReasoningAgent

def create_agent():
    """Create agent with environment-based configuration."""
    config = AgentConfig.from_env()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")
    
    return ReasoningAgent(config=config)

if __name__ == "__main__":
    agent = create_agent()
    # Start agent service...
```

### Multi-Instance Load Balancing

```python
# Load balancer configuration
class LoadBalancedAgent:
    def __init__(self, instance_configs: List[AgentConfig]):
        self.agents = [
            ReasoningAgent(config=config) 
            for config in instance_configs
        ]
        self.current = 0
    
    def get_agent(self) -> ReasoningAgent:
        """Round-robin load balancing."""
        agent = self.agents[self.current]
        self.current = (self.current + 1) % len(self.agents)
        return agent

# Create load-balanced setup
configs = [
    AgentConfig.for_profile(AgentProfile.PRODUCTION),
    AgentConfig.for_profile(AgentProfile.PRODUCTION),
    AgentConfig.for_profile(AgentProfile.PRODUCTION)
]

load_balancer = LoadBalancedAgent(configs)
```

### Auto-Scaling Configuration

```python
# Auto-scaling based on workload
class AutoScalingConfig:
    def __init__(self):
        self.base_config = AgentConfig.for_profile(AgentProfile.PRODUCTION)
    
    def get_config_for_load(self, load_factor: float) -> AgentConfig:
        """Adjust configuration based on current load."""
        config = copy.deepcopy(self.base_config)
        
        if load_factor > 0.8:
            # High load: reduce limits for faster turnaround
            config.reasoning.max_iterations = 10
            config.reasoning.max_execution_time = 120.0
            config.memory.session_token_cap = 6000
        elif load_factor < 0.3:
            # Low load: increase limits for better quality
            config.reasoning.max_iterations = 25
            config.reasoning.max_execution_time = 300.0
            config.memory.session_token_cap = 15000
        
        return config
```

## Monitoring & Observability

### Configuration-Based Monitoring

```python
# Enable comprehensive monitoring
monitoring_config = AgentConfig(
    guardrails=GuardrailConfig(
        enable_guardrail_monitoring=True,
        alert_on_guardrail_hit=True,
        guardrail_hit_threshold=0.75
    ),
    logging=LoggingConfig(
        log_level="INFO",
        langsmith_enabled=True,
        langsmith_project="AEC-Production",
        trace_llm_calls=True
    ),
    performance=PerformanceConfig(
        enable_resource_monitoring=True
    )
)
```

### Health Checks

```python
# Health check endpoint
def health_check(config: AgentConfig) -> Dict[str, Any]:
    """Comprehensive health check based on configuration."""
    checks = {}
    
    # API connectivity
    try:
        # Test LLM API
        test_llm = ChatOpenAI(model=config.llm.model_name)
        test_llm.invoke("test")
        checks["llm_api"] = "healthy"
    except Exception as e:
        checks["llm_api"] = f"unhealthy: {e}"
    
    # Configuration validation
    validation_errors = config.validate()
    checks["configuration"] = "valid" if not validation_errors else f"invalid: {validation_errors}"
    
    # Memory usage
    import psutil
    memory_percent = psutil.virtual_memory().percent
    checks["memory_usage"] = f"{memory_percent}%"
    
    # Overall status
    checks["status"] = "healthy" if all("healthy" in v or "valid" in v for v in checks.values() if isinstance(v, str)) else "unhealthy"
    
    return checks
```

## Troubleshooting

### Common Configuration Issues

#### 1. API Key Problems
```python
# Debug API key configuration
config = AgentConfig.from_env()

# Check if keys are loaded
print(f"OpenAI API Key present: {'OPENAI_API_KEY' in os.environ}")
print(f"LangSmith API Key present: {'LANGSMITH_API_KEY' in os.environ}")

# Test API connectivity
try:
    llm = ChatOpenAI(model=config.llm.model_name)
    result = llm.invoke("Hello")
    print("API connection successful ✓")
except Exception as e:
    print(f"API connection failed: {e}")
```

#### 2. Configuration Validation Errors
```python
# Comprehensive validation
config = AgentConfig()
all_errors = []

# Validate each section
sections = ['llm', 'reasoning', 'guardrails', 'memory', 'tools', 'performance', 'logging', 'security']
for section_name in sections:
    section = getattr(config, section_name)
    errors = section.validate() if hasattr(section, 'validate') else []
    if errors:
        all_errors.extend([f"{section_name}.{error}" for error in errors])

if all_errors:
    print("Configuration validation errors:")
    for error in all_errors:
        print(f"  - {error}")
```

#### 3. Environment Variable Loading Issues
```bash
# Check environment variable loading
python -c "
import os
from aec_agent.config import AgentConfig

print('Environment Variables:')
for key in sorted(os.environ.keys()):
    if key.startswith('AEC_'):
        print(f'  {key}={os.environ[key]}')

config = AgentConfig.from_env()
print(f'Loaded LLM model: {config.llm.model_name}')
print(f'Loaded max iterations: {config.reasoning.max_iterations}')
"
```

#### 4. Performance Issues
```python
# Performance debugging configuration
debug_config = AgentConfig(
    llm=LLMConfig(
        timeout_per_call=10.0,          # Short timeout
        max_retries=1,                  # Minimal retries
        max_concurrent_calls=1          # Serial execution
    ),
    reasoning=ReasoningConfig(
        max_iterations=5,               # Few iterations
        max_execution_time=30.0         # Short timeout
    ),
    logging=LoggingConfig(
        log_level="DEBUG",              # Detailed logging
        trace_llm_calls=True,          # Trace timing
        trace_tool_calls=True          # Trace tool performance
    ),
    performance=PerformanceConfig(
        enable_resource_monitoring=True, # Monitor resources
        max_concurrent_operations=1      # Serial execution
    )
)
```

### Debug Commands

```bash
# Configuration debugging commands

# 1. Show current configuration
python scripts/show_configuration.py --section all

# 2. Validate configuration
python -c "
from aec_agent.config import AgentConfig
config = AgentConfig.from_env()
errors = config.validate()
print('Valid ✓' if not errors else f'Errors: {errors}')
"

# 3. Test API connectivity
python -c "
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini')
result = llm.invoke('test')
print('API works ✓')
"

# 4. Show environment variables
env | grep AEC_

# 5. Test configuration loading
python examples/configuration_examples.py
```

## Best Practices

### Development Workflow
1. **Start with Development Profile**: Use generous limits for experimentation
2. **Enable Debug Logging**: Use detailed logging for troubleshooting
3. **Use Local Environment Variables**: Create `.env` file for local settings
4. **Test Configuration Changes**: Validate before committing

### Production Deployment
1. **Use Production Profile**: Start with production-optimized settings
2. **Set Environment Variables**: Configure via environment, not code
3. **Enable Monitoring**: Use LangSmith tracing and guardrail monitoring
4. **Implement Health Checks**: Monitor configuration and API connectivity
5. **Plan for Scaling**: Configure appropriate concurrency and resource limits

### Security Considerations
1. **Never Commit API Keys**: Use environment variables or secure vaults
2. **Enable Audit Logging**: Track sensitive operations in production
3. **Use Compliance Mode**: Enable additional security features if required
4. **Rotate Keys Regularly**: Implement key rotation strategy
5. **Monitor API Usage**: Watch for unusual patterns or excessive usage

## Summary

The unified configuration system provides:

✅ **Complete Control**: Every aspect of agent behavior is configurable  
✅ **Easy Management**: Single source of truth for all settings  
✅ **Environment Support**: Full environment variable configuration  
✅ **Profile System**: Pre-configured settings for common scenarios  
✅ **Validation**: Built-in validation and error checking  
✅ **Backward Compatibility**: Seamless migration from legacy configurations  
✅ **Tools & Examples**: Dashboard and examples for easy management  

This comprehensive configuration system enables the AEC Compliance Agent to be deployed in any environment from local development to enterprise production while maintaining optimal performance and security.