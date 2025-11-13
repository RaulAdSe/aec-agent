# Unified Agent Configuration System

## Overview

The AEC Reasoning Agent uses a comprehensive unified configuration system that centralizes all agent parameters in a single, easy-to-manage structure. This system replaces fragmented configuration across multiple files with a coherent, well-documented approach.

## Key Features

- **🎯 Centralized Configuration**: All settings in one place with clear organization
- **🌍 Environment Variable Support**: Complete configuration via environment variables  
- **📋 Configuration Profiles**: Pre-configured settings for different environments (dev/staging/prod/test)
- **✅ Validation & Compatibility**: Built-in validation and backward compatibility
- **📊 Configuration Dashboard**: Visual configuration inspection and debugging tools
- **🔧 Easy Customization**: Simple override system for any parameter

## Configuration Areas

The unified configuration system organizes settings into 8 logical areas:

### 1. **LLM Configuration** (`LLMConfig`)
- **Model Selection**: Provider, model name, fallback model
- **Model Parameters**: Temperature, max tokens, context window settings
- **API Configuration**: Timeouts, concurrency limits, rate limiting
- **Retry Configuration**: Max retries, backoff strategy, per-call timeouts

### 2. **Reasoning Configuration** (`ReasoningConfig`)
- **Reasoning Limits**: Max iterations, execution time, task depth
- **Goal Decomposition**: Subtask limits, complexity thresholds, adaptive planning
- **Task Execution**: Task timeouts, retry settings, parallel execution
- **Validation & Monitoring**: Progress tracking, result validation, strictness levels

### 3. **Guardrail Configuration** (`GuardrailConfig`)
- **LLM Guardrails**: Retry limits, timeouts, failure handling
- **Execution Guardrails**: Replanning limits, task attempt caps, execution step limits
- **Memory Guardrails**: Memory cleanup, context trimming, size limits
- **Monitoring & Alerts**: Guardrail monitoring, alert thresholds, proactive warnings

### 4. **Memory Configuration** (`MemoryConfig`)
- **Memory Types**: Short-term, long-term, episodic memory settings
- **Memory Limits**: Capacity limits for different memory types
- **Memory Persistence**: Disk persistence, compression, cleanup policies
- **Memory Optimization**: Compression thresholds, forgetting mechanisms

### 5. **Tool Configuration** (`ToolConfig`)
- **Tool Execution**: Timeouts, parallel execution, retry policies
- **Tool Discovery**: Auto-discovery, search paths, validation
- **Building Data Tools**: IFC processing settings, query limits, geometry extraction

### 6. **Performance Configuration** (`PerformanceConfig`)
- **Concurrency**: Max concurrent operations, parallel execution, worker pools
- **Caching**: Response caching, TTL settings, cache size limits
- **Optimization**: Request batching, smart routing, resource management

### 7. **Logging Configuration** (`LoggingConfig`)
- **Log Levels**: Global and component-specific log levels
- **Log Outputs**: Console, file, rotation settings
- **Debug Options**: Debug mode, tracing, artifact saving
- **LangSmith Tracing**: Integration with LangSmith for monitoring

### 8. **Security Configuration** (`SecurityConfig`)
- **API Security**: API key management, encryption, rotation
- **Input Validation**: Input sanitization, length limits, pattern blocking
- **Output Filtering**: Sensitive data filtering, content filtering
- **Audit & Compliance**: Audit logging, retention policies, compliance mode

## Usage Examples

### Basic Usage

```python
from aec_agent.core.agent_config import AgentConfig
from aec_agent.reasoning_agent import ReasoningAgent

# Use default configuration
agent = ReasoningAgent()

# Use environment-based configuration
config = AgentConfig.from_env()
agent = ReasoningAgent(config=config)

# Use profile-based configuration
dev_config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
agent = ReasoningAgent(config=dev_config)
```

### Custom Configuration

```python
from aec_agent.core.agent_config import (
    AgentConfig, LLMConfig, ReasoningConfig, GuardrailConfig
)

# Create custom configuration
custom_config = AgentConfig(
    llm=LLMConfig(
        model_name="gpt-4",
        temperature=0.1,
        max_tokens=4096,
        max_retries=5
    ),
    reasoning=ReasoningConfig(
        max_iterations=25,
        max_execution_time=300.0,
        validation_strictness=0.9
    ),
    guardrails=GuardrailConfig(
        max_replanning_events=5,
        max_total_execution_steps=50,
        enable_guardrail_monitoring=True
    )
)

agent = ReasoningAgent(config=custom_config)
```

### Environment Variable Configuration

```bash
# LLM Configuration
export AEC_LLM_MODEL_NAME="gpt-4"
export AEC_LLM_TEMPERATURE="0.05"
export AEC_LLM_MAX_TOKENS="4096"
export AEC_LLM_MAX_RETRIES="5"

# Reasoning Configuration
export AEC_REASONING_MAX_ITERATIONS="30"
export AEC_REASONING_MAX_EXECUTION_TIME="600.0"
export AEC_REASONING_MAX_PARALLEL="5"

# Guardrail Configuration
export AEC_GUARDRAILS_MAX_REPLANNING="8"
export AEC_GUARDRAILS_MAX_TASK_ATTEMPTS="4"
export AEC_GUARDRAILS_MAX_EXECUTION_STEPS="75"

# Performance Configuration
export AEC_PERFORMANCE_MAX_CONCURRENT="10"
export AEC_PERFORMANCE_ENABLE_CACHING="true"

# Logging Configuration
export AEC_LOG_LEVEL="INFO"
export AEC_DEBUG_MODE="false"
export AEC_LANGSMITH_ENABLED="true"
```

Then load automatically:

```python
# Automatically loads from environment variables
config = AgentConfig.from_env()
agent = ReasoningAgent(config=config)
```

## Configuration Profiles

Pre-configured profiles for common scenarios:

### Development Profile
- **Purpose**: Local development and experimentation
- **Model**: `gpt-4o-mini` (cost-effective)
- **Limits**: Generous limits for experimentation (30 iterations, 15 minutes)
- **Logging**: Debug mode with full tracing
- **Caching**: Enabled to reduce API calls

### Production Profile
- **Purpose**: Production deployment
- **Model**: `gpt-4` (most capable)
- **Limits**: Strict limits (15 iterations, 3 minutes)
- **Logging**: Warning level only
- **Security**: Audit logging and compliance mode enabled

### Staging Profile
- **Purpose**: Testing production scenarios
- **Model**: `gpt-4o-mini` (balanced)
- **Limits**: Moderate limits (20 iterations, 4 minutes)
- **Logging**: Info level with LangSmith integration

### Testing Profile
- **Purpose**: Automated testing and CI/CD
- **Model**: `gpt-3.5-turbo` (fast)
- **Limits**: Minimal limits (5 iterations, 30 seconds)
- **Logging**: Debug mode for test debugging
- **Caching**: Disabled for deterministic tests

## Configuration Tools

### Configuration Dashboard

View current configuration with the dashboard script:

```bash
# Show current configuration
python scripts/show_configuration.py --section current

# Show available profiles
python scripts/show_configuration.py --section profiles

# Show environment variables
python scripts/show_configuration.py --section environment

# Show example configurations
python scripts/show_configuration.py --section examples

# Show everything
python scripts/show_configuration.py --section all
```

### Configuration Examples

See comprehensive usage examples:

```bash
python examples/configuration_examples.py
```

### Configuration Validation

```python
config = AgentConfig()
errors = config.validate()

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid ✓")
```

## Real-World Scenarios

### Scenario 1: Local Development
```python
dev_config = AgentConfig(
    profile=AgentProfile.DEVELOPMENT,
    llm=LLMConfig(
        model_name="gpt-4o-mini",    # Cost-effective
        temperature=0.2,             # Slightly creative
        max_retries=5                # More retries for debugging
    ),
    reasoning=ReasoningConfig(
        max_iterations=50,           # Higher limits for experimentation
        max_execution_time=900.0,    # 15 minutes
        enable_adaptive_planning=True
    ),
    logging=LoggingConfig(
        log_level="DEBUG",
        debug_mode=True,
        trace_llm_calls=True,
        save_debug_artifacts=True
    )
)
```

### Scenario 2: Production Deployment
```python
prod_config = AgentConfig(
    profile=AgentProfile.PRODUCTION,
    llm=LLMConfig(
        model_name="gpt-4",          # Most capable
        temperature=0.05,            # Very deterministic
        max_retries=3,               # Conservative retries
        timeout_per_call=20.0        # Shorter timeouts
    ),
    reasoning=ReasoningConfig(
        max_iterations=10,           # Strict limits
        max_execution_time=120.0,    # 2 minutes max
        validation_strictness=0.95   # Very strict validation
    ),
    security=SecurityConfig(
        audit_log_retention_days=730,  # 2 years retention
        compliance_mode=True,
        enable_audit_logging=True
    )
)
```

### Scenario 3: High-Volume Batch Processing
```python
batch_config = AgentConfig(
    llm=LLMConfig(
        model_name="gpt-4o-mini",    # Fast and cost-effective
        temperature=0.0,             # Maximum determinism
        max_concurrent_calls=20,     # High concurrency
        rate_limit_rpm=3000          # High rate limit
    ),
    reasoning=ReasoningConfig(
        max_iterations=5,            # Quick processing
        max_execution_time=60.0,     # 1 minute per item
        max_parallel_tasks=10        # High parallelism
    ),
    performance=PerformanceConfig(
        enable_response_caching=True,
        cache_ttl_seconds=7200,      # 2 hour cache
        max_concurrent_operations=20,
        enable_request_batching=True,
        batch_size=10
    )
)
```

## Migration from Legacy Configuration

The unified configuration system maintains backward compatibility:

### Automatic Migration
```python
# Old way (still works)
agent = ReasoningAgent(
    model_name="gpt-4",
    temperature=0.1,
    max_iterations=15
)

# New way (recommended)
config = AgentConfig(
    llm=LLMConfig(model_name="gpt-4", temperature=0.1),
    reasoning=ReasoningConfig(max_iterations=15)
)
agent = ReasoningAgent(config=config)
```

### Configuration Conversion
```python
# Convert to legacy guardrail config for backward compatibility
config = AgentConfig()
legacy_guardrail_config = config.get_effective_guardrail_config()

# Use with existing components that expect legacy config
reasoning_controller = ReasoningController(
    ...,
    guardrail_config=legacy_guardrail_config
)
```

## Environment Variable Reference

Complete list of environment variables for configuration:

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `AEC_LLM_MODEL_NAME` | gpt-4o-mini | LLM model name | gpt-4 |
| `AEC_LLM_TEMPERATURE` | 0.1 | LLM temperature (0.0-2.0) | 0.05 |
| `AEC_LLM_MAX_TOKENS` | 8192 | Maximum tokens per response | 4096 |
| `AEC_LLM_MAX_RETRIES` | 3 | Maximum LLM retry attempts | 5 |
| `AEC_LLM_TIMEOUT` | 30.0 | LLM call timeout (seconds) | 45.0 |
| `AEC_REASONING_MAX_ITERATIONS` | 20 | Maximum reasoning iterations | 30 |
| `AEC_REASONING_MAX_EXECUTION_TIME` | 300.0 | Maximum execution time (seconds) | 600.0 |
| `AEC_REASONING_MAX_PARALLEL` | 3 | Maximum parallel tasks | 5 |
| `AEC_GUARDRAILS_MAX_REPLANNING` | 5 | Maximum replanning events | 8 |
| `AEC_GUARDRAILS_MAX_TASK_ATTEMPTS` | 3 | Maximum task attempts | 4 |
| `AEC_GUARDRAILS_MAX_EXECUTION_STEPS` | 50 | Maximum execution steps | 75 |
| `AEC_GUARDRAILS_MAX_MEMORY_STEPS` | 100 | Maximum memory steps | 150 |
| `AEC_MEMORY_ENABLE_PERSISTENCE` | true | Enable memory persistence | false |
| `AEC_MEMORY_SHORT_TERM_CAPACITY` | 1000 | Short-term memory capacity | 2000 |
| `AEC_PERFORMANCE_MAX_CONCURRENT` | 5 | Maximum concurrent operations | 10 |
| `AEC_PERFORMANCE_ENABLE_CACHING` | true | Enable response caching | false |
| `AEC_LOG_LEVEL` | INFO | Log level | DEBUG |
| `AEC_DEBUG_MODE` | false | Enable debug mode | true |
| `AEC_LANGSMITH_ENABLED` | true | Enable LangSmith tracing | false |

## Best Practices

### 1. **Use Environment Variables for Deployment**
- Set environment variables for production configuration
- Use different values for dev/staging/prod environments
- Keep API keys in environment variables, never in code

### 2. **Use Profiles for Common Scenarios**
- Start with predefined profiles (development, production, etc.)
- Customize specific parameters as needed
- Create custom profiles for specialized use cases

### 3. **Validate Configuration**
- Always validate configuration before use
- Check for common issues (missing API keys, invalid ranges)
- Use the configuration dashboard for debugging

### 4. **Monitor Configuration in Production**
- Enable guardrail monitoring in production
- Set appropriate alert thresholds
- Log configuration changes for audit trails

### 5. **Performance Tuning**
- Start with conservative settings
- Monitor actual usage patterns
- Adjust limits based on real-world performance
- Use caching for repeated operations

## Integration with External Guardrails

The unified configuration integrates seamlessly with the external guardrails system:

```python
# Configuration automatically provides guardrail settings
config = AgentConfig()
agent = ReasoningAgent(config=config)

# Guardrails are automatically configured from:
# - config.guardrails (GuardrailConfig)
# - config.llm (LLMConfig retry settings)
# - config.memory (MemoryConfig limits)
```

The external guardrails documentation is available in `docs/EXTERNAL_GUARDRAILS.md`.

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export LANGSMITH_API_KEY="your-langsmith-key-here"
   ```

2. **Configuration Validation Errors**
   ```python
   config = AgentConfig()
   errors = config.validate()
   print("Errors:", errors)
   ```

3. **Environment Variables Not Loading**
   ```bash
   # Check current environment
   python scripts/show_configuration.py --section environment
   ```

4. **Performance Issues**
   - Increase concurrent operations: `AEC_PERFORMANCE_MAX_CONCURRENT=10`
   - Enable caching: `AEC_PERFORMANCE_ENABLE_CACHING=true`
   - Reduce retry attempts: `AEC_LLM_MAX_RETRIES=1`

### Debug Configuration
```python
# Enable debug mode for detailed logging
debug_config = AgentConfig(
    logging=LoggingConfig(
        log_level="DEBUG",
        debug_mode=True,
        trace_llm_calls=True,
        trace_tool_calls=True,
        save_debug_artifacts=True
    )
)
```

## Future Enhancements

Planned improvements to the configuration system:

1. **Configuration Hot-Reloading**: Update configuration without restarting
2. **Configuration Profiles Storage**: Save and load custom profiles
3. **Advanced Validation**: More sophisticated validation rules
4. **Configuration History**: Track configuration changes over time
5. **Performance Recommendations**: AI-powered configuration optimization

## Summary

The unified configuration system provides:

✅ **Complete Control**: Every aspect of agent behavior is configurable
✅ **Easy Management**: Single source of truth for all settings  
✅ **Environment Support**: Full environment variable configuration
✅ **Profile System**: Pre-configured settings for common scenarios
✅ **Validation**: Built-in validation and error checking
✅ **Compatibility**: Backward compatibility with legacy systems
✅ **Tools**: Dashboard and examples for easy configuration management

This system centralizes all agent configuration in an organized, well-documented, and easy-to-use format, making it simple to configure the agent for any scenario from development to production deployment.