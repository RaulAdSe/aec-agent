#!/usr/bin/env python3
"""
Test Agent with LangSmith Tracing

This script creates and tests the AEC reasoning agent with LangSmith tracing enabled.
It will generate traces that you can view in your LangSmith dashboard.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
from pathlib import Path
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('"\'')
                os.environ[key] = value

# Ensure we can import from the package
sys.path.insert(0, str(Path(__file__).parent))

from aec_agent import ReasoningAgent, AgentConfig, AgentProfile


def test_agent_with_simple_query():
    """Test the agent with a simple query to generate LangSmith traces."""
    
    print("=== AEC Agent Test with LangSmith Tracing ===\n")
    
    # Create development configuration for testing
    config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
    
    # Enable LangSmith tracing explicitly
    config.logging.enable_langsmith = True
    config.logging.langsmith_project = "AEC-Agent-Testing"
    
    print(f"Configuration:")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Max Iterations: {config.reasoning.max_iterations}")
    print(f"  LangSmith Enabled: {config.logging.enable_langsmith}")
    print(f"  LangSmith Project: {config.logging.langsmith_project}")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    
    print(f"\nAPI Keys:")
    print(f"  OpenAI API Key: {'✓ Set' if openai_key else '✗ Not Set'}")
    print(f"  LangSmith API Key: {'✓ Set' if langsmith_key else '✗ Not Set'}")
    
    if not openai_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return False
        
    if not langsmith_key:
        print("\n⚠️  WARNING: LANGSMITH_API_KEY not found")
        print("LangSmith tracing will be disabled")
        print("To enable tracing, set:")
        print("export LANGSMITH_API_KEY='your-langsmith-key-here'")
    
    try:
        print("\n🚀 Creating AEC Reasoning Agent...")
        agent = ReasoningAgent(config=config)
        
        print("✅ Agent created successfully!")
        print(f"\nAgent Status: {agent.get_status()}")
        
        # Test with a simple query
        print("\n🧠 Testing agent with a simple building analysis query...")
        
        test_goal = "Explain what building elements I need to analyze for fire safety compliance in a residential building"
        
        print(f"Query: {test_goal}")
        print("\n⏳ Processing...")
        
        # This should generate LangSmith traces
        result = agent.process_goal(test_goal)
        
        print("\n📊 Result:")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        
        if result.get('reasoning_result'):
            reasoning = result['reasoning_result']
            if reasoning.get('summary'):
                summary = reasoning['summary']
                print(f"\nSummary:")
                print(f"  Completed Tasks: {summary.get('completed_tasks', 0)}")
                print(f"  Total Tasks: {summary.get('total_tasks', 0)}")
                print(f"  Execution Time: {summary.get('total_execution_time', 0):.2f}s")
        
        print("\n✅ Test completed successfully!")
        
        if langsmith_key:
            print(f"\n🔍 Check your LangSmith dashboard for traces:")
            print(f"   Project: {config.logging.langsmith_project}")
            print(f"   URL: https://smith.langchain.com/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during agent test: {e}")
        return False


def test_configuration_dashboard():
    """Show the configuration being used."""
    
    print("\n" + "="*60)
    print("Configuration Dashboard")
    print("="*60)
    
    config = AgentConfig.for_profile(AgentProfile.DEVELOPMENT)
    
    print(f"\nLLM Configuration:")
    print(f"  Provider: {config.llm.provider.value}")
    print(f"  Model: {config.llm.model_name}")
    print(f"  Temperature: {config.llm.temperature}")
    print(f"  Max Tokens: {config.llm.max_tokens}")
    print(f"  Max Retries: {config.llm.max_retries}")
    
    print(f"\nReasoning Configuration:")
    print(f"  Max Iterations: {config.reasoning.max_iterations}")
    print(f"  Max Execution Time: {config.reasoning.max_execution_time}s")
    print(f"  Max Parallel Tasks: {config.reasoning.max_parallel_tasks}")
    print(f"  Adaptive Planning: {config.reasoning.enable_adaptive_planning}")
    
    print(f"\nGuardrail Configuration:")
    print(f"  Max Replanning Events: {config.guardrails.max_replanning_events}")
    print(f"  Max Task Attempts: {config.guardrails.max_same_task_attempts}")
    print(f"  Max Execution Steps: {config.guardrails.max_total_execution_steps}")
    print(f"  Monitoring Enabled: {config.guardrails.enable_guardrail_monitoring}")
    
    print(f"\nLogging Configuration:")
    print(f"  Log Level: {config.logging.log_level}")
    print(f"  Debug Mode: {config.logging.debug_mode}")
    print(f"  LangSmith Enabled: {config.logging.enable_langsmith}")
    print(f"  LangSmith Project: {config.logging.langsmith_project}")
    
    # Validate configuration
    errors = config.validate()
    print(f"\nConfiguration Validation:")
    if errors:
        print(f"  ❌ {len(errors)} error(s) found:")
        for error in errors:
            print(f"    - {error}")
    else:
        print(f"  ✅ Configuration is valid")


def main():
    """Main test function."""
    
    test_configuration_dashboard()
    
    print("\n" + "="*60)
    print("Agent Test")
    print("="*60)
    
    success = test_agent_with_simple_query()
    
    if success:
        print("\n🎉 All tests passed! The agent is working correctly.")
        print("\nNext steps:")
        print("1. Check LangSmith for execution traces")
        print("2. Try more complex queries")
        print("3. Load building data and run compliance analysis")
    else:
        print("\n💥 Test failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()