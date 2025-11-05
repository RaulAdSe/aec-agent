"""Configuration for the AEC Compliance Agent."""

from typing import List, Optional
from pydantic import Field

from ...agents.base_agent import BaseAgentConfig


class ComplianceAgentConfig(BaseAgentConfig):
    """Configuration specific to the AEC Compliance Agent."""
    
    name: str = Field(default="compliance_agent", description="Agent name")
    description: str = Field(
        default="AEC compliance verification agent for Spanish building codes",
        description="Agent description"
    )
    
    # Compliance-specific settings
    building_codes: List[str] = Field(
        default=["CTE DB-SI", "CTE DB-SUA"],
        description="Building codes to check compliance against"
    )
    
    use_toon: bool = Field(
        default=True,
        description="Whether to use TOON format for data exchange"
    )
    
    # Analysis settings
    max_analysis_iterations: int = Field(
        default=10,
        description="Maximum iterations for analysis"
    )
    
    confidence_threshold: float = Field(
        default=0.8,
        description="Minimum confidence for compliance decisions"
    )