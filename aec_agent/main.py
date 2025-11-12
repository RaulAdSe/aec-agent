"""
Main entry point for the AEC Compliance Agent.

Command-line interface for running compliance analysis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

import click

# Load environment variables
load_dotenv()

from .core.config import config
from .core.logger import get_logger


# Initialize logging
logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config-file', type=click.Path(exists=True), help='Configuration file path')
def cli(verbose: bool, config_file: Optional[str]):
    """AEC Compliance Agent - Building code compliance verification."""
    if verbose:
        config.log_level = "DEBUG"
        logger.info("Verbose logging enabled")
    
    if config_file:
        logger.info(f"Loading configuration from {config_file}")
        # TODO: Implement config file loading


@cli.command()
@click.argument('building_data_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--analysis-type', 
              type=click.Choice(['general', 'fire_safety', 'accessibility']),
              default='general',
              help='Type of compliance analysis to perform')
def analyze(building_data_file: str, output: Optional[str], analysis_type: str):
    """Analyze building data for compliance using LangChain agent."""
    
    if not config.openai_api_key:
        click.echo("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return
    
    logger.info(f"Starting {analysis_type} analysis of {building_data_file}")
    
    try:
        from .reasoning_agent import create_reasoning_agent
        
        # Create LangChain agent
        agent = create_reasoning_agent(verbose=config.log_level == "DEBUG", temperature=0.1)
        
        # Prepare query based on analysis type and file
        if analysis_type == 'fire_safety':
            query = f"Load building data from {building_data_file} and perform a comprehensive fire safety compliance analysis"
        elif analysis_type == 'accessibility':
            query = f"Load building data from {building_data_file} and analyze accessibility compliance requirements"
        else:
            query = f"Load building data from {building_data_file} and perform a general compliance analysis"
        
        # Perform analysis
        result = agent.process({"query": query})
        
        if result['status'] == 'success':
            # Prepare results for output
            results = {
                "analysis_type": analysis_type,
                "building_data_file": building_data_file,
                "status": "success",
                "analysis": result['response'],
                "tools_used": [],
                "timestamp": "2024-01-01T00:00:00"  # TODO: Use actual timestamp
            }
            
            # Output results
            output_results(results, output)
            logger.info("Analysis completed successfully")
        else:
            click.echo(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
        
    except ImportError as e:
        click.echo(f"❌ Missing LangChain dependencies: {e}")
        click.echo("Install with: pip install langchain langchain-openai")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        click.echo(f"❌ Analysis failed: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show agent status and configuration."""
    
    click.echo("AEC Compliance Agent Status")
    click.echo("=" * 30)
    click.echo(f"Environment: {config.environment}")
    click.echo(f"Default Model: {config.default_model}")
    click.echo(f"TOON Enabled: {config.use_toon} (disabled)")
    click.echo(f"Data Directory: {config.data_dir}")
    click.echo(f"Log Level: {config.log_level}")
    
    # Check API key availability
    openai_status = "✓ Available" if config.openai_api_key else "✗ Not configured"
    click.echo(f"OpenAI API Key: {openai_status}")
    
    # Check LangSmith tracing
    langsmith_status = "✓ Enabled" if config.langchain_tracing_v2 and config.langchain_api_key else "✗ Disabled"
    click.echo(f"LangSmith Tracing: {langsmith_status}")
    click.echo(f"LangSmith Project: {config.langchain_project}")


# API server functionality removed - use reasoning agent directly


@cli.command()
@click.argument('query')
@click.option('--building-data', help='Path to building data JSON file')
@click.option('--verbose', is_flag=True, help='Show detailed output')
def query(query: str, building_data: Optional[str], verbose: bool):
    """Run a natural language query against the LangChain agent."""
    
    if not config.openai_api_key:
        click.echo("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return
    
    try:
        from .reasoning_agent import create_reasoning_agent
        
        if verbose:
            click.echo("🔧 Creating LangChain agent...")
        
        agent = create_reasoning_agent(verbose=verbose, temperature=0.1)
        
        # Prepare query with optional building data
        query_text = query
        if building_data:
            query_text = f"First load building data from {building_data}, then: {query}"
        
        # Execute query
        click.echo(f"🔍 Query: {query}")
        if building_data:
            click.echo(f"📊 Building data: {building_data}")
        
        result = agent.process(query_text)
        
        if result['status'] == 'success':
            click.echo(f"\n✅ Result:")
            click.echo(result['response'])
            
            # Note: tools_used not available in simplified version
        else:
            click.echo(f"\n❌ Error: {result.get('message', 'Unknown error')}")
            
    except ImportError as e:
        click.echo(f"❌ Missing LangChain dependencies: {e}")
        click.echo("Install with: pip install langchain langchain-openai")
    except Exception as e:
        click.echo(f"❌ Query failed: {e}")


@cli.command()
def chat():
    """Start an interactive chat session with the LangChain agent."""
    
    if not config.openai_api_key:
        click.echo("❌ OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return
        
    try:
        from .reasoning_agent import create_reasoning_agent
        
        click.echo("🏗️ AEC Compliance Agent - Interactive Chat")
        click.echo("Enter natural language queries about building compliance")
        click.echo("Type 'exit' to quit")
        
        agent = create_reasoning_agent(verbose=False, temperature=0.1)
        click.echo(f"✅ Agent ready: {agent.get_status()['name']}")
        
        while True:
            try:
                user_input = click.prompt("\n🔍 Query", type=str).strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'help':
                    click.echo("""
Available commands:
- Ask natural language questions about building compliance
- 'status' - Show agent status  
- 'exit' - Quit

Example queries:
- "What is the status of the compliance knowledge base?"
- "Load building data from data/out/FM-ARC_v2.json and analyze it"
- "What are the fire safety requirements for emergency exits?"
                    """)
                elif user_input.lower() == 'status':
                    status = agent.get_status()
                    click.echo(f"Status: {status['status']}")
                    click.echo(f"Tools: {len(status['available_tools'])}")
                    for tool in status['available_tools']:
                        click.echo(f"  - {tool}")
                else:
                    # Process with LangChain agent
                    click.echo("🔄 Processing...")
                    result = agent.process(user_input)
                    
                    if result['status'] == 'success':
                        click.echo(f"\n💬 Response:")
                        click.echo(result['response'])
                        # Note: tools_used not available in simplified version
                    else:
                        click.echo(f"❌ Error: {result.get('message', 'Unknown error')}")
                        
            except click.Abort:
                click.echo("\nGoodbye! 👋")
                break
            except Exception as e:
                click.echo(f"❌ Error: {e}")
                
    except ImportError as e:
        click.echo(f"❌ Missing LangChain dependencies: {e}")
        click.echo("Install with: pip install langchain langchain-openai")
    except Exception as e:
        click.echo(f"❌ Failed to create agent: {e}")


@cli.command()
@click.argument('project_name')
def init_project(project_name: str):
    """Initialize a new compliance analysis project."""
    
    project_dir = Path(config.data_dir) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create project structure
    (project_dir / "blueprints").mkdir(exist_ok=True)
    (project_dir / "analysis_results").mkdir(exist_ok=True)
    (project_dir / "reports").mkdir(exist_ok=True)
    
    # Create project config
    project_config = {
        "project_name": project_name,
        "created_at": "2024-01-01T00:00:00",  # TODO: Use actual timestamp
        "compliance_codes": ["CTE DB-SI", "CTE DB-SUA"],
        "analysis_settings": {
            "use_toon": False,
            "confidence_threshold": 0.8
        }
    }
    
    config_file = project_dir / "project_config.json"
    with open(config_file, 'w') as f:
        json.dump(project_config, f, indent=2)
    
    click.echo(f"Project '{project_name}' initialized at {project_dir}")
    click.echo("Project structure:")
    click.echo("  blueprints/     - Place IFC/CAD files here")
    click.echo("  analysis_results/ - Analysis outputs")
    click.echo("  reports/        - Generated compliance reports")


def load_building_data(file_path: str) -> Dict[str, Any]:
    """Load building data from file."""
    
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Building data file not found: {file_path}")
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # TODO: Add support for other formats (IFC, DXF, etc.)
        raise ValueError(f"Unsupported file format: {path.suffix}")


def output_results(results: Dict[str, Any], output_file: Optional[str] = None):
    """Output analysis results."""
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        click.echo(f"Results saved to {output_path}")
    else:
        # Pretty print to console
        click.echo("Analysis Results:")
        click.echo("=" * 40)
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))


@cli.command()
@click.argument('goal')
@click.option('--session-id', help='Session ID for memory continuity')
@click.option('--max-iterations', default=20, help='Maximum reasoning iterations')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
def reason(goal: str, session_id: Optional[str] = None, max_iterations: int = 20, verbose: bool = False):
    """
    Process a goal using autonomous reasoning.
    
    Example: aec_agent reason "Analyze fire safety compliance"
    """
    try:
        from .reasoning_agent import create_reasoning_agent
        
        click.echo(f"🤖 Starting autonomous reasoning for goal: {goal}")
        click.echo("=" * 60)
        
        # Create reasoning agent
        agent = create_reasoning_agent(
            model_name=config.default_model,
            temperature=config.temperature,
            verbose=verbose,
            enable_memory=True,
            session_id=session_id,
            max_iterations=max_iterations
        )
        
        # Process the goal
        result = agent.process_goal(goal)
        
        # Display results
        click.echo("\n📊 Reasoning Results:")
        click.echo("=" * 40)
        click.echo(f"Status: {result.get('status')}")
        click.echo(f"Message: {result.get('message')}")
        
        reasoning_result = result.get('reasoning_result', {})
        summary = reasoning_result.get('summary', {})
        
        if summary:
            click.echo(f"\n📈 Summary:")
            click.echo(f"  Goal Achieved: {summary.get('goal_achieved', False)}")
            click.echo(f"  Tasks Completed: {summary.get('completed_tasks', 0)}/{summary.get('total_tasks', 0)}")
            click.echo(f"  Execution Time: {summary.get('execution_time', 0):.2f}s")
            click.echo(f"  Iterations: {summary.get('iterations', 0)}")
        
        # Display task breakdown
        tasks = reasoning_result.get('tasks', [])
        if tasks:
            click.echo(f"\n📋 Task Breakdown:")
            for i, task in enumerate(tasks, 1):
                status_icon = "✅" if task['status'] == 'completed' else "❌" if task['status'] == 'failed' else "⏳"
                click.echo(f"  {i}. {status_icon} {task['name']} ({task['status']})")
                if task.get('tools'):
                    click.echo(f"     Tools: {', '.join(task['tools'])}")
        
        # Display session info if available
        session_summary = result.get('session_summary')
        if session_summary:
            click.echo(f"\n💾 Session Info:")
            click.echo(f"  Session ID: {session_summary.get('session_id')}")
            click.echo(f"  Memory Stats: {session_summary['memory_stats']['total_turns']} conversation turns")
        
        if verbose and reasoning_result.get('outputs'):
            click.echo(f"\n🔍 Detailed Outputs:")
            for output in reasoning_result['outputs']:
                click.echo(f"  Tool: {output['tool']}")
                click.echo(f"  Time: {output['execution_time']:.2f}s")
                click.echo(f"  Output: {str(output['output'])[:200]}...")
                click.echo()
        
        click.echo(f"\n🎉 Reasoning completed!")
        
    except ImportError as e:
        click.echo(f"❌ Error: Reasoning agent not available. {e}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    cli()