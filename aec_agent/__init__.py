"""
AEC Compliance Agent - Autonomous reasoning agent for building compliance analysis.
"""

from .core.reasoning_agent import ReasoningAgent, create_reasoning_agent
from .config import AgentConfig, AgentProfile

# Keep legacy alias for backwards compatibility
ComplianceAgent = ReasoningAgent
create_agent = create_reasoning_agent

__all__ = [
    "ReasoningAgent",
    "create_reasoning_agent", 
    "AgentConfig",
    "AgentProfile",
    "ComplianceAgent",  # Legacy alias
    "create_agent"      # Legacy alias
]