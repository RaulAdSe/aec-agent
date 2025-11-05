"""
Web application entry point for the AEC Compliance Agent.

Provides REST API and optional web UI for compliance analysis.
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string
from werkzeug.exceptions import BadRequest

from .core.config import config
from .core.logger import get_logger
from .agents.compliance_agent import ComplianceAgent, ComplianceAgentConfig


# Initialize Flask app
app = Flask(__name__)
logger = get_logger(__name__)

# Global agent instance
agent: Optional[ComplianceAgent] = None


def create_app() -> Flask:
    """Create and configure the Flask application."""
    
    global agent
    
    # Initialize agent
    agent_config = ComplianceAgentConfig(
        use_toon=config.use_toon,
        verbose=config.debug
    )
    agent = ComplianceAgent(agent_config)
    
    logger.info("AEC Compliance Agent web application initialized")
    return app


@app.route('/')
def index():
    """Home page with basic UI."""
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AEC Compliance Agent</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 4px; }
            .method { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AEC Compliance Agent API</h1>
            <p>Building code compliance verification service</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> /status - Agent status and configuration
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> /analyze - Analyze building data for compliance
                <br><small>Body: {"building_data": {...}, "analysis_type": "general|fire_safety|accessibility"}</small>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> /memory - Get agent memory summary
            </div>
            
            <div class="endpoint">
                <span class="method">DELETE</span> /memory - Clear agent memory
            </div>
            
            <h2>Configuration:</h2>
            <ul>
                <li>Environment: {{ environment }}</li>
                <li>Model: {{ model }}</li>
                <li>TOON Enabled: {{ use_toon }}</li>
                <li>Debug Mode: {{ debug }}</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(
        html_template,
        environment=config.environment,
        model=config.default_model,
        use_toon=config.use_toon,
        debug=config.debug
    )


@app.route('/status', methods=['GET'])
def get_status():
    """Get agent status and configuration."""
    
    status_info = {
        "agent_status": agent.get_status() if agent else {"status": "not_initialized"},
        "configuration": {
            "environment": config.environment,
            "model": config.default_model,
            "use_toon": config.use_toon,
            "debug": config.debug
        },
        "api_keys": {
            "google_api_key": "configured" if config.google_api_key else "missing"
        }
    }
    
    return jsonify(status_info)


@app.route('/analyze', methods=['POST'])
def analyze_building():
    """Analyze building data for compliance."""
    
    if not agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            raise BadRequest("No JSON data provided")
        
        building_data = data.get('building_data')
        analysis_type = data.get('analysis_type', 'general')
        
        if not building_data:
            raise BadRequest("No building_data provided")
        
        # Perform analysis
        if analysis_type == 'general':
            results = agent.process(building_data)
        else:
            results = agent.analyze_specific_compliance(building_data, analysis_type)
        
        return jsonify({
            "status": "success",
            "analysis_type": analysis_type,
            "results": results
        })
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/memory', methods=['GET'])
def get_memory():
    """Get agent memory summary."""
    
    if not agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    memory_summary = agent.get_memory_summary()
    return jsonify(memory_summary)


@app.route('/memory', methods=['DELETE'])
def clear_memory():
    """Clear agent memory."""
    
    if not agent:
        return jsonify({"error": "Agent not initialized"}), 500
    
    agent.clear_memory()
    return jsonify({"status": "success", "message": "Memory cleared"})


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    
    return jsonify({
        "status": "healthy",
        "agent_initialized": agent is not None,
        "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Create and run the app
    app = create_app()
    
    # Run with debug mode based on config
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=config.debug
    )