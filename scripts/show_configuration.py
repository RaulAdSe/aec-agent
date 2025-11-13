#!/usr/bin/env python3
"""
Configuration Dashboard - Display current agent configuration in a user-friendly format.

This script shows how the unified AgentConfig system works and displays
all configuration parameters in an organized dashboard format.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aec_agent.config import AgentConfig, AgentProfile, get_example_configs


def print_header(title: str, width: int = 80):
    """Print a formatted section header."""
    print("=" * width)
    print(f" {title.center(width-2)} ")
    print("=" * width)


def print_section(title: str, items: dict, width: int = 80):
    """Print a formatted configuration section."""
    print(f"\n{title}:")
    print("-" * len(title))
    
    for key, value in items.items():
        # Format boolean values
        if isinstance(value, bool):
            value = "✓" if value else "✗"
        # Format None values
        elif value is None:
            value = "Not Set"
        # Format enum values
        elif hasattr(value, 'value'):
            value = value.value
        
        print(f"  {key.replace('_', ' ').title():<30} {value}")


def show_current_configuration():
    """Display the current configuration loaded from environment."""
    
    print_header("AEC AGENT CONFIGURATION DASHBOARD")
    
    # Load current configuration
    config = AgentConfig.from_env()
    
    print(f"\nConfiguration Profile: {config.profile.value.upper()}")
    print(f"Session ID: {config.session_id or 'Auto-generated'}")
    print(f"Project Root: {config.project_root}")
    print(f"Data Directory: {config.data_dir}")
    
    # === LLM CONFIGURATION ===
    llm_config = {
        "provider": config.llm.provider,
        "model_name": config.llm.model_name,
        "fallback_model": config.llm.fallback_model,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens,
        "context_window": config.llm.context_window,
        "max_context_usage": f"{config.llm.max_context_usage:.0%}",
        "api_timeout": f"{config.llm.api_timeout}s",
        "max_retries": config.llm.max_retries,
        "retry_delay": f"{config.llm.retry_delay}s",
        "exponential_backoff": config.llm.exponential_backoff
    }
    print_section("LLM Configuration", llm_config)
    
    # === REASONING CONFIGURATION ===
    reasoning_config = {
        "max_iterations": config.reasoning.max_iterations,
        "max_execution_time": f"{config.reasoning.max_execution_time}s",
        "max_task_depth": config.reasoning.max_task_depth,
        "max_parallel_tasks": config.reasoning.max_parallel_tasks,
        "max_subtasks_per_goal": config.reasoning.max_subtasks_per_goal,
        "enable_adaptive_planning": config.reasoning.enable_adaptive_planning,
        "task_timeout": f"{config.reasoning.task_timeout}s",
        "enable_task_retry": config.reasoning.enable_task_retry,
        "max_task_retries": config.reasoning.max_task_retries,
        "validation_strictness": f"{config.reasoning.validation_strictness:.0%}"
    }
    print_section("Reasoning Configuration", reasoning_config)
    
    # === GUARDRAIL CONFIGURATION ===
    guardrail_config = {
        "max_replanning_events": config.guardrails.max_replanning_events,
        "max_same_task_attempts": config.guardrails.max_same_task_attempts,
        "max_total_execution_steps": config.guardrails.max_total_execution_steps,
        "max_execution_steps_memory": config.guardrails.max_execution_steps_memory,
        "max_context_summary_length": f"{config.guardrails.max_context_summary_length} chars",
        "enable_auto_cleanup": config.guardrails.enable_auto_cleanup,
        "cleanup_threshold": f"{config.guardrails.cleanup_threshold:.0%}",
        "enable_guardrail_monitoring": config.guardrails.enable_guardrail_monitoring,
        "alert_threshold": f"{config.guardrails.alert_threshold:.0%}"
    }
    print_section("Guardrail Configuration", guardrail_config)
    
    # === MEMORY CONFIGURATION ===
    memory_config = {
        "enable_short_term_memory": config.memory.enable_short_term_memory,
        "enable_long_term_memory": config.memory.enable_long_term_memory,
        "enable_episodic_memory": config.memory.enable_episodic_memory,
        "short_term_capacity": config.memory.short_term_capacity,
        "long_term_capacity": config.memory.long_term_capacity,
        "enable_persistence": config.memory.enable_persistence,
        "persistence_interval": f"{config.memory.persistence_interval}s",
        "enable_memory_compression": config.memory.enable_memory_compression,
        "compression_threshold_days": f"{config.memory.compression_threshold_days} days"
    }
    print_section("Memory Configuration", memory_config)
    
    # === PERFORMANCE CONFIGURATION ===
    performance_config = {
        "max_concurrent_operations": config.performance.max_concurrent_operations,
        "enable_parallel_execution": config.performance.enable_parallel_execution,
        "worker_pool_size": config.performance.worker_pool_size,
        "enable_response_caching": config.performance.enable_response_caching,
        "cache_ttl_seconds": f"{config.performance.cache_ttl_seconds}s",
        "max_cache_size_mb": f"{config.performance.max_cache_size_mb}MB",
        "enable_request_batching": config.performance.enable_request_batching,
        "batch_size": config.performance.batch_size,
        "max_memory_usage_mb": f"{config.performance.max_memory_usage_mb}MB"
    }
    print_section("Performance Configuration", performance_config)
    
    # === LOGGING CONFIGURATION ===
    logging_config = {
        "log_level": config.logging.log_level,
        "llm_log_level": config.logging.llm_log_level,
        "tool_log_level": config.logging.tool_log_level,
        "log_to_console": config.logging.log_to_console,
        "log_to_file": config.logging.log_to_file,
        "debug_mode": config.logging.debug_mode,
        "trace_llm_calls": config.logging.trace_llm_calls,
        "trace_tool_calls": config.logging.trace_tool_calls,
        "enable_langsmith": config.logging.enable_langsmith,
        "langsmith_project": config.logging.langsmith_project
    }
    print_section("Logging Configuration", logging_config)
    
    # === SECURITY CONFIGURATION ===
    security_config = {
        "require_api_keys": config.security.require_api_keys,
        "sanitize_inputs": config.security.sanitize_inputs,
        "max_input_length": config.security.max_input_length,
        "filter_sensitive_data": config.security.filter_sensitive_data,
        "enable_audit_logging": config.security.enable_audit_logging,
        "audit_log_retention_days": f"{config.security.audit_log_retention_days} days",
        "compliance_mode": config.security.compliance_mode
    }
    print_section("Security Configuration", security_config)
    
    # === API KEYS STATUS ===
    api_keys_status = {
        "openai_api_key": "✓ Set" if config.openai_api_key else "✗ Not Set",
        "anthropic_api_key": "✓ Set" if config.anthropic_api_key else "✗ Not Set", 
        "langsmith_api_key": "✓ Set" if config.langsmith_api_key else "✗ Not Set"
    }
    print_section("API Keys Status", api_keys_status)
    
    # === CONFIGURATION VALIDATION ===
    validation_errors = config.validate()
    print(f"\nConfiguration Validation:")
    print("-" * 25)
    if validation_errors:
        print("  ✗ Configuration has errors:")
        for error in validation_errors:
            print(f"    - {error}")
    else:
        print("  ✓ Configuration is valid")
    
    print("\n" + "=" * 80)


def show_available_profiles():
    """Show available configuration profiles."""
    
    print_header("AVAILABLE CONFIGURATION PROFILES")
    
    profiles = [
        (AgentProfile.DEVELOPMENT, "Development - Verbose logging, relaxed limits, debugging enabled"),
        (AgentProfile.STAGING, "Staging - Balanced settings for testing production scenarios"),  
        (AgentProfile.PRODUCTION, "Production - Minimal logging, strict limits, optimized performance"),
        (AgentProfile.TESTING, "Testing - Fast execution, minimal retries, comprehensive logging")
    ]
    
    for profile, description in profiles:
        print(f"\n{profile.value.upper()}:")
        print(f"  {description}")
        
        # Show key settings for this profile
        config = AgentConfig.for_profile(profile)
        print(f"  Model: {config.llm.model_name}")
        print(f"  Max Iterations: {config.reasoning.max_iterations}")
        print(f"  Log Level: {config.logging.log_level}")
        print(f"  Debug Mode: {'✓' if config.logging.debug_mode else '✗'}")


def show_environment_variables():
    """Show environment variable usage."""
    
    print_header("ENVIRONMENT VARIABLE CONFIGURATION")
    
    env_vars = [
        # LLM Configuration
        ("AEC_LLM_MODEL_NAME", "gpt-4", "LLM model name"),
        ("AEC_LLM_TEMPERATURE", "0.1", "LLM temperature (0.0-2.0)"),
        ("AEC_LLM_MAX_TOKENS", "8192", "Maximum tokens per response"),
        ("AEC_LLM_MAX_RETRIES", "3", "Maximum LLM retry attempts"),
        ("AEC_LLM_TIMEOUT", "30.0", "LLM call timeout (seconds)"),
        
        # Reasoning Configuration  
        ("AEC_REASONING_MAX_ITERATIONS", "20", "Maximum reasoning iterations"),
        ("AEC_REASONING_MAX_EXECUTION_TIME", "300.0", "Maximum execution time (seconds)"),
        ("AEC_REASONING_MAX_PARALLEL", "3", "Maximum parallel tasks"),
        
        # Guardrail Configuration
        ("AEC_GUARDRAILS_MAX_REPLANNING", "5", "Maximum replanning events"),
        ("AEC_GUARDRAILS_MAX_TASK_ATTEMPTS", "3", "Maximum task attempts"),
        ("AEC_GUARDRAILS_MAX_EXECUTION_STEPS", "50", "Maximum execution steps"),
        ("AEC_GUARDRAILS_MAX_MEMORY_STEPS", "100", "Maximum memory steps"),
        
        # Performance Configuration
        ("AEC_PERFORMANCE_MAX_CONCURRENT", "5", "Maximum concurrent operations"),
        ("AEC_PERFORMANCE_ENABLE_CACHING", "true", "Enable response caching"),
        
        # Logging Configuration
        ("AEC_LOG_LEVEL", "INFO", "Log level (DEBUG/INFO/WARNING/ERROR)"),
        ("AEC_DEBUG_MODE", "false", "Enable debug mode"),
        ("AEC_LANGSMITH_ENABLED", "true", "Enable LangSmith tracing")
    ]
    
    print("\nSet these environment variables to customize configuration:")
    print("\n# LLM Configuration")
    for var, default, desc in env_vars:
        if "LLM" in var:
            current = os.getenv(var, "Not Set")
            status = "✓" if current != "Not Set" else "✗"
            print(f'export {var}="{default}"  # {desc} (Current: {current}) {status}')
    
    print("\n# Reasoning Configuration")
    for var, default, desc in env_vars:
        if "REASONING" in var:
            current = os.getenv(var, "Not Set")
            status = "✓" if current != "Not Set" else "✗"
            print(f'export {var}="{default}"  # {desc} (Current: {current}) {status}')
    
    print("\n# Guardrail Configuration")
    for var, default, desc in env_vars:
        if "GUARDRAILS" in var:
            current = os.getenv(var, "Not Set")
            status = "✓" if current != "Not Set" else "✗"
            print(f'export {var}="{default}"  # {desc} (Current: {current}) {status}')
    
    print("\n# Other Configuration")
    for var, default, desc in env_vars:
        if not any(section in var for section in ["LLM", "REASONING", "GUARDRAILS"]):
            current = os.getenv(var, "Not Set")
            status = "✓" if current != "Not Set" else "✗"
            print(f'export {var}="{default}"  # {desc} (Current: {current}) {status}')


def show_example_configurations():
    """Show example configurations."""
    
    print_header("EXAMPLE CONFIGURATIONS")
    
    examples = get_example_configs()
    
    for name, config in examples.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  Model: {config.llm.model_name}")
        print(f"  Temperature: {config.llm.temperature}")
        print(f"  Max Iterations: {config.reasoning.max_iterations}")
        print(f"  Max Execution Steps: {config.guardrails.max_total_execution_steps}")
        
        if hasattr(config, 'memory') and hasattr(config.memory, 'enable_memory_compression'):
            print(f"  Memory Compression: {'✓' if config.memory.enable_memory_compression else '✗'}")
        if hasattr(config, 'logging') and hasattr(config.logging, 'debug_mode'):
            print(f"  Debug Mode: {'✓' if config.logging.debug_mode else '✗'}")


def main():
    """Main entry point."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Display AEC agent configuration")
    parser.add_argument(
        "--section", 
        choices=["current", "profiles", "environment", "examples", "all"],
        default="current",
        help="Configuration section to display"
    )
    
    args = parser.parse_args()
    
    if args.section == "current":
        show_current_configuration()
    elif args.section == "profiles":
        show_available_profiles()
    elif args.section == "environment":
        show_environment_variables()
    elif args.section == "examples":
        show_example_configurations()
    elif args.section == "all":
        show_current_configuration()
        show_available_profiles()
        show_environment_variables()
        show_example_configurations()
    
    print(f"\n{'='*80}")
    print("Configuration Dashboard Complete")
    print("\nUsage Examples:")
    print("  python scripts/show_configuration.py --section current     # Show current config")
    print("  python scripts/show_configuration.py --section profiles    # Show available profiles")
    print("  python scripts/show_configuration.py --section environment # Show environment vars")
    print("  python scripts/show_configuration.py --section examples    # Show example configs")
    print("  python scripts/show_configuration.py --section all         # Show everything")


if __name__ == "__main__":
    main()