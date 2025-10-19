"""
Agent state management for AEC compliance verification.

This module defines the state structure for the ReAct agent, including
message history, iteration tracking, and compliance results.
"""

from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
import operator


class ComplianceResult(TypedDict):
    """Result of a compliance check."""
    check_type: str
    element_id: str
    is_compliant: bool
    message: str
    regulation_reference: Optional[str]
    details: Dict[str, Any]


class AgentState(TypedDict):
    """
    State of the ReAct agent during execution.
    
    This state tracks:
    - Message history for conversation flow
    - Iteration count to prevent infinite loops
    - Compliance results from checks performed
    - Current focus area for the agent
    """
    # Core conversation state
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
    
    # Compliance tracking
    compliance_results: Annotated[List[ComplianceResult], operator.add]
    current_focus: Optional[str]  # Current area of focus (e.g., "door_widths", "egress_distances")
    
    # Agent metadata
    agent_mode: str  # "verification", "analysis", "explanation"
    max_iterations: int
    is_complete: bool


def create_initial_state(
    query: str,
    agent_mode: str = "verification",
    max_iterations: int = 15
) -> AgentState:
    """
    Create initial agent state for a new compliance verification session.
    
    Args:
        query: Initial user query or task
        agent_mode: Mode of operation ("verification", "analysis", "explanation")
        max_iterations: Maximum number of iterations before stopping
        
    Returns:
        Initial agent state
    """
    from langchain_core.messages import HumanMessage
    
    return AgentState(
        messages=[HumanMessage(content=query)],
        iterations=0,
        compliance_results=[],
        current_focus=None,
        agent_mode=agent_mode,
        max_iterations=max_iterations,
        is_complete=False
    )


def add_compliance_result(
    state: AgentState,
    check_type: str,
    element_id: str,
    is_compliant: bool,
    message: str,
    regulation_reference: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Add a compliance result to the agent state.
    
    Args:
        state: Current agent state
        check_type: Type of compliance check (e.g., "door_width", "egress_distance")
        element_id: ID of the element checked (e.g., "D001", "R001")
        is_compliant: Whether the element is compliant
        message: Human-readable compliance message
        regulation_reference: Reference to relevant regulation
        details: Additional details about the check
        
    Returns:
        Updated agent state
    """
    result = ComplianceResult(
        check_type=check_type,
        element_id=element_id,
        is_compliant=is_compliant,
        message=message,
        regulation_reference=regulation_reference,
        details=details or {}
    )
    
    # Create new state with added result
    new_state = state.copy()
    new_state["compliance_results"] = state["compliance_results"] + [result]
    
    return new_state


def get_compliance_summary(state: AgentState) -> Dict[str, Any]:
    """
    Get a summary of all compliance results.
    
    Args:
        state: Current agent state
        
    Returns:
        Dictionary with compliance summary
    """
    results = state["compliance_results"]
    
    if not results:
        return {
            "total_checks": 0,
            "compliant": 0,
            "non_compliant": 0,
            "compliance_rate": 0.0,
            "issues": []
        }
    
    compliant_count = sum(1 for r in results if r["is_compliant"])
    non_compliant_count = len(results) - compliant_count
    compliance_rate = compliant_count / len(results) if results else 0.0
    
    # Group issues by type
    issues_by_type = {}
    for result in results:
        if not result["is_compliant"]:
            check_type = result["check_type"]
            if check_type not in issues_by_type:
                issues_by_type[check_type] = []
            issues_by_type[check_type].append({
                "element_id": result["element_id"],
                "message": result["message"],
                "regulation": result["regulation_reference"]
            })
    
    return {
        "total_checks": len(results),
        "compliant": compliant_count,
        "non_compliant": non_compliant_count,
        "compliance_rate": compliance_rate,
        "issues_by_type": issues_by_type,
        "overall_status": "COMPLIANT" if non_compliant_count == 0 else "NON_COMPLIANT"
    }


def should_continue_execution(state: AgentState) -> bool:
    """
    Determine if the agent should continue execution.
    
    Args:
        state: Current agent state
        
    Returns:
        True if agent should continue, False otherwise
    """
    # Check iteration limit
    if state["iterations"] >= state["max_iterations"]:
        return False
    
    # Check if explicitly marked as complete
    if state["is_complete"]:
        return False
    
    return True


def mark_complete(state: AgentState) -> AgentState:
    """
    Mark the agent execution as complete.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state
    """
    new_state = state.copy()
    new_state["is_complete"] = True
    return new_state


def update_focus(state: AgentState, focus_area: str) -> AgentState:
    """
    Update the current focus area of the agent.
    
    Args:
        state: Current agent state
        focus_area: New focus area (e.g., "door_widths", "egress_distances")
        
    Returns:
        Updated agent state
    """
    new_state = state.copy()
    new_state["current_focus"] = focus_area
    return new_state
