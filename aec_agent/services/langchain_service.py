"""
LangChain service for the AEC Compliance Agent.

Provides FastAPI application with LangServe integration for the LangChain agent.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI
from langserve import add_routes
from langchain.schema.runnable import Runnable
from pydantic import BaseModel

from ..agent import create_agent


class ComplianceQuery(BaseModel):
    """Input schema for compliance queries."""
    query: str
    building_data_path: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ComplianceResponse(BaseModel):
    """Output schema for compliance responses."""
    status: str
    message: str
    response: str
    tools_used: list
    raw_result: Optional[Dict[str, Any]] = None


class ComplianceAgentRunnable(Runnable):
    """
    LangChain Runnable wrapper for the compliance agent.
    
    This allows the agent to be used with LangServe and LangChain Studio.
    """
    
    def __init__(self):
        """Initialize the runnable with the compliance agent."""
        self.agent = create_agent(verbose=True, temperature=0.1)
    
    def invoke(self, input: ComplianceQuery, config=None) -> ComplianceResponse:
        """
        Invoke the compliance agent with the given query.
        
        Args:
            input: ComplianceQuery with user query and optional context
            config: Optional run configuration
            
        Returns:
            ComplianceResponse with agent results
        """
        # Prepare input data for the agent
        input_data = {
            "query": input.query,
            "context": input.context or {}
        }
        
        # If building data path is provided, include it in the context
        if input.building_data_path:
            input_data["context"]["building_data_path"] = input.building_data_path
        
        # Process with the agent
        result = self.agent.process(input.query)
        
        # Return structured response
        return ComplianceResponse(
            status=result.get("status", "unknown"),
            message=result.get("message", ""),
            response=result.get("response", ""),
            tools_used=[],
            raw_result=result.get("raw_result")
        )
    
    async def ainvoke(self, input: ComplianceQuery, config=None) -> ComplianceResponse:
        """Async version of invoke."""
        # For now, just call the sync version
        # In production, you might want to implement true async processing
        return self.invoke(input, config)


def create_app() -> FastAPI:
    """Create the FastAPI application with LangChain agent."""
    
    # Create the FastAPI app
    app = FastAPI(
        title="AEC Compliance Agent API",
        description="LangChain-powered agent for AEC building compliance analysis",
        version="1.0.0"
    )
    
    # Create the compliance agent runnable
    compliance_agent = ComplianceAgentRunnable()
    
    # Add routes for LangServe
    add_routes(
        app,
        compliance_agent,
        path="/compliance",
        input_type=ComplianceQuery,
        output_type=ComplianceResponse
    )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "aec-compliance-agent"}
    
    # Agent status endpoint
    @app.get("/agent/status")
    async def agent_status():
        """Get agent status and available tools."""
        return compliance_agent.agent.get_status()
    
    # Example usage endpoint for testing
    @app.post("/example")
    async def example_query():
        """Example query to test the agent."""
        example_input = ComplianceQuery(
            query="What is the status of the compliance knowledge base?",
            context={"source": "api_test"}
        )
        
        return compliance_agent.invoke(example_input)
    
    return app