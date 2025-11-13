"""
Configuration Examples - Demonstrating how to use the unified AgentConfig system.

This file provides practical examples of how to configure the AEC reasoning agent
for different scenarios and use cases.
"""

import os
from pathlib import Path

# Import the unified configuration system
from aec_agent.config import (
    AgentConfig, AgentProfile, LLMProvider,
    LLMConfig, ReasoningConfig, GuardrailConfig, MemoryConfig,
    ToolConfig, PerformanceConfig, LoggingConfig, SecurityConfig,
    get_example_configs
)
from aec_agent import ReasoningAgent


def main():
    """Run configuration examples."""
    
    print("=== AEC Agent Configuration Examples ===\n")
    
    # === EXAMPLE 1: Using Default Configuration ===
    print("1. Default Configuration:")
    default_config = AgentConfig()
    print(f"   Model: {default_config.llm.model_name}")
    print(f"   Max Iterations: {default_config.reasoning.max_iterations}")
    print(f"   Max Execution Steps: {default_config.guardrails.max_total_execution_steps}")
    print()
    
    # === EXAMPLE 2: Environment-Based Configuration ===
    print("2. Environment-Based Configuration:")
    
    # Set some example environment variables
    os.environ["AEC_LLM_MODEL_NAME"] = "gpt-4"
    os.environ["AEC_LLM_TEMPERATURE"] = "0.0"
    os.environ["AEC_REASONING_MAX_ITERATIONS"] = "25"
    os.environ["AEC_GUARDRAILS_MAX_REPLANNING"] = "8"
    
    env_config = AgentConfig.from_env()
    print(f"   Model (from env): {env_config.llm.model_name}")
    print(f"   Temperature (from env): {env_config.llm.temperature}")
    print(f"   Max Iterations (from env): {env_config.reasoning.max_iterations}")
    print(f"   Max Replanning (from env): {env_config.guardrails.max_replanning_events}")
    print()
    
    # === EXAMPLE 3: Profile-Based Configuration ===
    print("3. Profile-Based Configurations:")
    
    # Development profile
    dev_config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
    print(f"   Development - Model: {dev_config.llm.model_name}, Log Level: {dev_config.logging.log_level}")
    
    # Production profile
    prod_config = AgentConfig.for_profile(AgentProfile.PRODUCTION)
    print(f"   Production - Model: {prod_config.llm.model_name}, Log Level: {prod_config.logging.log_level}")
    
    # Testing profile
    test_config = AgentConfig.for_profile(AgentProfile.TESTING)
    print(f"   Testing - Model: {test_config.llm.model_name}, Max Iterations: {test_config.reasoning.max_iterations}")
    print()
    
    # === EXAMPLE 4: Custom Configuration ===
    print("4. Custom Configuration:")
    
    custom_config = AgentConfig(
        llm=LLMConfig(
            model_name="gpt-4-turbo",
            temperature=0.2,
            max_tokens=4096,
            max_retries=5
        ),
        reasoning=ReasoningConfig(
            max_iterations=15,
            max_execution_time=240.0,  # 4 minutes
            max_parallel_tasks=3,
            validation_strictness=0.9
        ),
        guardrails=GuardrailConfig(
            max_replanning_events=3,
            max_total_execution_steps=30,
            enable_guardrail_monitoring=True
        ),
        memory=MemoryConfig(
            enable_persistence=True,
            short_term_capacity=500,
            enable_memory_compression=True
        ),
        performance=PerformanceConfig(
            enable_response_caching=True,
            max_concurrent_operations=8
        ),
        logging=LoggingConfig(
            log_level="DEBUG",
            enable_langsmith=True,
            trace_llm_calls=True
        )
    )
    
    print(f"   Custom - Model: {custom_config.llm.model_name}")
    print(f"   Custom - Temperature: {custom_config.llm.temperature}")
    print(f"   Custom - Max Iterations: {custom_config.reasoning.max_iterations}")
    print(f"   Custom - Validation Strictness: {custom_config.reasoning.validation_strictness}")
    print()
    
    # === EXAMPLE 5: Specialized Configurations ===
    print("5. Specialized Example Configurations:")
    
    examples = get_example_configs()
    
    print("   Available examples:")
    for name, config in examples.items():
        print(f"   - {name}: {config.llm.model_name}, {config.reasoning.max_iterations} iterations")
    print()
    
    # === EXAMPLE 6: Configuration Validation ===
    print("6. Configuration Validation:")
    
    # Valid configuration
    valid_config = AgentConfig()
    errors = valid_config.validate()
    print(f"   Valid config errors: {len(errors)}")
    
    # Invalid configuration
    invalid_config = AgentConfig(
        llm=LLMConfig(temperature=3.0),  # Invalid temperature > 2.0
        reasoning=ReasoningConfig(max_iterations=0)  # Invalid iterations <= 0
    )
    errors = invalid_config.validate()
    print(f"   Invalid config errors: {len(errors)}")
    for error in errors:
        print(f"     - {error}")
    print()
    
    # === EXAMPLE 7: Using Configuration with ReasoningAgent ===
    print("7. Using Configuration with ReasoningAgent:")
    
    # Method 1: Pass configuration directly
    high_perf_config = examples["high_performance"]
    # agent = ReasoningAgent(config=high_perf_config)
    print(f"   High performance config: {high_perf_config.llm.model_name}, {high_perf_config.performance.max_concurrent_operations} concurrent ops")
    
    # Method 2: Legacy parameters (backward compatibility)
    # agent = ReasoningAgent(model_name="gpt-4", temperature=0.1, max_iterations=10)
    print("   Legacy parameters supported for backward compatibility")
    
    # Method 3: Environment-based (automatic)
    # agent = ReasoningAgent()  # Uses AgentConfig.from_env() automatically
    print("   Environment-based config loaded automatically if no config provided")
    print()


def demonstrate_environment_configuration():
    """Demonstrate comprehensive environment variable configuration."""
    
    print("=== Environment Variable Configuration ===\n")
    
    # Set comprehensive environment variables
    env_vars = {
        # LLM Configuration
        "AEC_LLM_MODEL_NAME": "gpt-4",
        "AEC_LLM_TEMPERATURE": "0.05",
        "AEC_LLM_MAX_TOKENS": "4096",
        "AEC_LLM_MAX_RETRIES": "5",
        "AEC_LLM_TIMEOUT": "45.0",
        
        # Reasoning Configuration
        "AEC_REASONING_MAX_ITERATIONS": "30",
        "AEC_REASONING_MAX_EXECUTION_TIME": "600.0",
        "AEC_REASONING_MAX_PARALLEL": "5",
        
        # Guardrail Configuration
        "AEC_GUARDRAILS_MAX_REPLANNING": "8",
        "AEC_GUARDRAILS_MAX_TASK_ATTEMPTS": "4",
        "AEC_GUARDRAILS_MAX_EXECUTION_STEPS": "75",
        "AEC_GUARDRAILS_MAX_MEMORY_STEPS": "150",
        
        # Memory Configuration
        "AEC_MEMORY_ENABLE_PERSISTENCE": "true",
        "AEC_MEMORY_SHORT_TERM_CAPACITY": "2000",
        
        # Performance Configuration
        "AEC_PERFORMANCE_MAX_CONCURRENT": "10",
        "AEC_PERFORMANCE_ENABLE_CACHING": "true",
        
        # Logging Configuration
        "AEC_LOG_LEVEL": "INFO",
        "AEC_DEBUG_MODE": "false",
        "AEC_LANGSMITH_ENABLED": "true",
    }
    
    # Apply environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
        
    # Load configuration from environment
    config = AgentConfig.from_env()
    
    print("Environment configuration loaded:")
    print(f"  LLM Model: {config.llm.model_name}")
    print(f"  LLM Temperature: {config.llm.temperature}")
    print(f"  LLM Max Retries: {config.llm.max_retries}")
    print(f"  Reasoning Max Iterations: {config.reasoning.max_iterations}")
    print(f"  Reasoning Max Execution Time: {config.reasoning.max_execution_time}")
    print(f"  Guardrails Max Replanning: {config.guardrails.max_replanning_events}")
    print(f"  Guardrails Max Steps: {config.guardrails.max_total_execution_steps}")
    print(f"  Memory Persistence: {config.memory.enable_persistence}")
    print(f"  Performance Max Concurrent: {config.performance.max_concurrent_operations}")
    print(f"  Logging Level: {config.logging.log_level}")
    print()


def demonstrate_configuration_conversion():
    """Demonstrate configuration conversion and compatibility."""
    
    print("=== Configuration Conversion ===\n")
    
    # Create unified config
    config = AgentConfig(
        llm=LLMConfig(model_name="gpt-4", temperature=0.1),
        reasoning=ReasoningConfig(max_iterations=25),
        guardrails=GuardrailConfig(max_replanning_events=5)
    )
    
    # Convert to legacy guardrail config for backward compatibility
    legacy_guardrail_config = config.get_effective_guardrail_config()
    print("Legacy GuardrailConfig compatibility:")
    print(f"  Max Retries: {legacy_guardrail_config.llm_max_retries}")
    print(f"  Max Replanning: {legacy_guardrail_config.max_replanning_events}")
    print(f"  Max Execution Steps: {legacy_guardrail_config.max_total_execution_steps}")
    print()
    
    # Convert to dictionary for serialization
    config_dict = config.to_dict()
    print("Dictionary format (for JSON serialization):")
    print(f"  Profile: {config_dict['profile']}")
    print(f"  LLM Model: {config_dict['llm']['model_name']}")
    print(f"  Reasoning Max Iterations: {config_dict['reasoning']['max_iterations']}")
    print()


def demonstrate_real_world_scenarios():
    """Demonstrate real-world configuration scenarios."""
    
    print("=== Real-World Configuration Scenarios ===\n")
    
    # === Scenario 1: Development Environment ===
    print("Scenario 1: Local Development")
    dev_config = AgentConfig(
        profile=AgentProfile.DEVELOPMENT,
        llm=LLMConfig(
            model_name="gpt-4o-mini",  # Cheaper for development
            temperature=0.2,           # Slightly more creative for exploration
            max_retries=5              # More retries for debugging
        ),
        reasoning=ReasoningConfig(
            max_iterations=50,         # Higher limit for experimentation
            max_execution_time=900.0,  # 15 minutes for complex debugging
            enable_adaptive_planning=True
        ),
        guardrails=GuardrailConfig(
            max_replanning_events=15,  # Allow more replanning for testing
            max_total_execution_steps=200
        ),
        logging=LoggingConfig(
            log_level="DEBUG",
            debug_mode=True,
            trace_llm_calls=True,
            save_debug_artifacts=True
        ),
        performance=PerformanceConfig(
            enable_response_caching=True,  # Cache responses to save API calls
            max_concurrent_operations=2    # Conservative for development machine
        )
    )
    
    print(f"  Model: {dev_config.llm.model_name} (cost-effective)")
    print(f"  Debug Mode: {dev_config.logging.debug_mode} (full tracing)")
    print(f"  Max Iterations: {dev_config.reasoning.max_iterations} (generous limits)")
    print()
    
    # === Scenario 2: Production Deployment ===
    print("Scenario 2: Production Deployment")
    prod_config = AgentConfig(
        profile=AgentProfile.PRODUCTION,
        llm=LLMConfig(
            model_name="gpt-4",        # Most capable model
            temperature=0.05,          # Very deterministic
            max_retries=3,             # Conservative retries
            timeout_per_call=20.0      # Shorter timeouts
        ),
        reasoning=ReasoningConfig(
            max_iterations=10,         # Strict limits
            max_execution_time=120.0,  # 2 minutes max
            validation_strictness=0.95 # Very strict validation
        ),
        guardrails=GuardrailConfig(
            max_replanning_events=2,   # Minimal replanning
            max_total_execution_steps=15,
            enable_guardrail_monitoring=True,
            alert_threshold=0.7        # Early alerts
        ),
        logging=LoggingConfig(
            log_level="WARNING",       # Minimal logging
            debug_mode=False,
            enable_langsmith=True      # Production monitoring
        ),
        performance=PerformanceConfig(
            enable_response_caching=True,
            max_concurrent_operations=15,  # High concurrency
            enable_parallel_execution=True
        ),
        security=SecurityConfig(
            audit_log_retention_days=730,  # 2 years retention
            compliance_mode=True,
            enable_audit_logging=True
        )
    )
    
    print(f"  Model: {prod_config.llm.model_name} (most capable)")
    print(f"  Validation Strictness: {prod_config.reasoning.validation_strictness} (very strict)")
    print(f"  Max Iterations: {prod_config.reasoning.max_iterations} (strict limits)")
    print(f"  Compliance Mode: {prod_config.security.compliance_mode} (audit ready)")
    print()
    
    # === Scenario 3: High-Volume Processing ===
    print("Scenario 3: High-Volume Batch Processing")
    batch_config = AgentConfig(
        llm=LLMConfig(
            model_name="gpt-4o-mini",  # Fast and cost-effective
            temperature=0.0,           # Maximum determinism
            max_concurrent_calls=20,   # High concurrency
            rate_limit_rpm=3000        # High rate limit
        ),
        reasoning=ReasoningConfig(
            max_iterations=5,          # Quick processing
            max_execution_time=60.0,   # 1 minute per item
            max_parallel_tasks=10      # High parallelism
        ),
        guardrails=GuardrailConfig(
            max_replanning_events=1,   # Minimal replanning
            max_total_execution_steps=10
        ),
        memory=MemoryConfig(
            short_term_capacity=100,   # Smaller memory footprint
            enable_memory_compression=True,
            enable_forgetting=True     # Forget old data
        ),
        performance=PerformanceConfig(
            enable_response_caching=True,
            cache_ttl_seconds=7200,    # 2 hour cache
            max_concurrent_operations=20,
            enable_request_batching=True,
            batch_size=10
        ),
        logging=LoggingConfig(
            log_level="INFO",
            trace_llm_calls=False     # Reduce logging overhead
        )
    )
    
    print(f"  Parallelism: {batch_config.reasoning.max_parallel_tasks} tasks, {batch_config.llm.max_concurrent_calls} LLM calls")
    print(f"  Processing Speed: {batch_config.reasoning.max_execution_time}s per item")
    print(f"  Memory Optimization: {batch_config.memory.enable_memory_compression}, {batch_config.memory.enable_forgetting}")
    print()


if __name__ == "__main__":
    main()
    print("\\n" + "="*60 + "\\n")
    demonstrate_environment_configuration()
    print("\\n" + "="*60 + "\\n")
    demonstrate_configuration_conversion()
    print("\\n" + "="*60 + "\\n")
    demonstrate_real_world_scenarios()
    
    print("\\n=== Configuration Examples Complete ===")
    print("\\nTo use these configurations:")
    print("1. Set environment variables as shown in the examples")
    print("2. Use AgentConfig.from_env() to load them automatically")
    print("3. Pass custom AgentConfig to ReasoningAgent(config=...)")
    print("4. Use profile-based configs for common scenarios")