"""
Main entry point for the AEC Compliance Agent.

Command-line interface for running compliance analysis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

import click

from .core.config import config
from .core.logger import get_logger
from .core.registry import registry
from .agents.compliance_agent import ComplianceAgent, ComplianceAgentConfig


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
@click.option('--use-toon/--no-toon', default=True, help='Use TOON format for data processing')
def analyze(building_data_file: str, output: Optional[str], analysis_type: str, use_toon: bool):
    """Analyze building data for compliance."""
    
    logger.info(f"Starting {analysis_type} analysis of {building_data_file}")
    
    try:
        # Load building data
        building_data = load_building_data(building_data_file)
        
        # Create and configure agent
        agent_config = ComplianceAgentConfig(
            use_toon=use_toon,
            verbose=config.log_level == "DEBUG"
        )
        agent = ComplianceAgent(agent_config)
        
        # Perform analysis
        if analysis_type == 'general':
            results = agent.process(building_data)
        else:
            results = agent.analyze_specific_compliance(building_data, analysis_type)
        
        # Output results
        output_results(results, output)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


@cli.command()
def status():
    """Show agent status and configuration."""
    
    click.echo("AEC Compliance Agent Status")
    click.echo("=" * 30)
    click.echo(f"Environment: {config.environment}")
    click.echo(f"Default Model: {config.default_model}")
    click.echo(f"TOON Enabled: {config.use_toon}")
    click.echo(f"Data Directory: {config.data_dir}")
    click.echo(f"Log Level: {config.log_level}")
    
    # Check API key availability
    api_status = "✓ Available" if config.google_api_key else "✗ Not configured"
    click.echo(f"Google API Key: {api_status}")


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
            "use_toon": True,
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


if __name__ == '__main__':
    cli()