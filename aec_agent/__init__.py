"""
AEC Compliance Agent - Autonomous reasoning agent for building compliance analysis.
"""

from .reasoning_agent import ReasoningAgent, create_reasoning_agent

# Keep legacy alias for backwards compatibility
ComplianceAgent = ReasoningAgent
create_agent = create_reasoning_agent

__all__ = [
    "ReasoningAgent",
    "create_reasoning_agent",
    "ComplianceAgent",  # Legacy alias
    "create_agent"      # Legacy alias
]