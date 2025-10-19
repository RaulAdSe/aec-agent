"""
LangGraph orchestration for the AEC compliance verification agent.

This module implements the ReAct (Reasoning and Acting) pattern using LangGraph
to create an autonomous agent that can verify building code compliance.
"""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from .state import AgentState, should_continue_execution, mark_complete
from .tools import (
    get_room_info,
    get_door_info, 
    list_all_doors,
    check_door_width_compliance,
    query_normativa,
    calculate_egress_distance
)
from .prompts import get_system_prompt


class ComplianceAgent:
    """
    ReAct agent for building code compliance verification.
    
    This agent uses the ReAct pattern to autonomously verify building designs
    against Spanish building codes (CTE) by reasoning about what information
    it needs and taking actions to gather that information.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-pro",
        temperature: float = 0.1,
        max_iterations: int = 15
    ):
        """
        Initialize the compliance agent.
        
        Args:
            model_name: Google model name to use
            temperature: LLM temperature for response generation
            max_iterations: Maximum number of reasoning iterations
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        # Define available tools
        self.tools = [
            get_room_info,
            get_door_info,
            list_all_doors,
            check_door_width_compliance,
            query_normativa,
            calculate_egress_distance
        ]
        
        # Initialize LLM with tools
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        ).bind_tools(self.tools)
        
        # Build the agent graph
        self.agent = self._build_agent_graph()
    
    def _build_agent_graph(self) -> StateGraph:
        """
        Build the LangGraph agent workflow.
        
        Returns:
            Compiled StateGraph agent
        """
        # Create the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """
        Agent reasoning node - the core ReAct reasoning step.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with agent's response
        """
        messages = state["messages"]
        
        # Add system prompt if this is the first iteration
        if state["iterations"] == 0:
            system_prompt = get_system_prompt(state.get("agent_mode", "verification"))
            system_message = SystemMessage(content=system_prompt)
            messages = [system_message] + list(messages)
        
        # Call the LLM
        response = self.llm.invoke(messages)
        
        # Update state
        new_state = state.copy()
        new_state["messages"] = list(messages) + [response]
        new_state["iterations"] = state["iterations"] + 1
        
        return new_state
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Decide whether to continue execution or end.
        
        Args:
            state: Current agent state
            
        Returns:
            "continue" to use tools, "end" to finish
        """
        # Check if we should continue execution
        if not should_continue_execution(state):
            return "end"
        
        # Check if the last message has tool calls
        messages = state["messages"]
        if not messages:
            return "end"
        
        last_message = messages[-1]
        
        # If the agent made tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        # If the agent didn't make tool calls, we're done
        return "end"
    
    def verify_compliance(
        self,
        query: str,
        agent_mode: str = "verification"
    ) -> Dict[str, Any]:
        """
        Run compliance verification for a given query.
        
        Args:
            query: User query or verification task
            agent_mode: Mode of operation
            
        Returns:
            Dictionary with verification results
        """
        from .state import create_initial_state
        
        # Create initial state
        initial_state = create_initial_state(
            query=query,
            agent_mode=agent_mode,
            max_iterations=self.max_iterations
        )
        
        # Run the agent
        final_state = self.agent.invoke(initial_state)
        
        # Extract results
        return self._extract_results(final_state)
    
    def _extract_results(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract and format results from agent execution.
        
        Args:
            state: Final agent state
            
        Returns:
            Formatted results dictionary
        """
        # Get the final agent response
        messages = state["messages"]
        agent_responses = [msg for msg in messages if isinstance(msg, AIMessage)]
        final_response = agent_responses[-1] if agent_responses else None
        
        # Get compliance results
        compliance_results = state.get("compliance_results", [])
        
        # Calculate summary statistics
        total_checks = len(compliance_results)
        compliant_count = sum(1 for r in compliance_results if r.get("is_compliant", False))
        non_compliant_count = total_checks - compliant_count
        compliance_rate = compliant_count / total_checks if total_checks > 0 else 0.0
        
        # Group issues by type
        issues_by_type = {}
        for result in compliance_results:
            if not result.get("is_compliant", True):
                check_type = result.get("check_type", "unknown")
                if check_type not in issues_by_type:
                    issues_by_type[check_type] = []
                issues_by_type[check_type].append({
                    "element_id": result.get("element_id", "unknown"),
                    "message": result.get("message", ""),
                    "regulation": result.get("regulation_reference", "")
                })
        
        return {
            "query": state["messages"][0].content if state["messages"] else "",
            "agent_response": final_response.content if final_response else "",
            "iterations": state["iterations"],
            "compliance_summary": {
                "total_checks": total_checks,
                "compliant": compliant_count,
                "non_compliant": non_compliant_count,
                "compliance_rate": compliance_rate,
                "overall_status": "COMPLIANT" if non_compliant_count == 0 else "NON_COMPLIANT"
            },
            "issues_by_type": issues_by_type,
            "detailed_results": compliance_results,
            "execution_successful": state["iterations"] < state["max_iterations"]
        }


def create_compliance_agent(
    model_name: str = "gemini-pro",
    temperature: float = 0.1,
    max_iterations: int = 15
) -> ComplianceAgent:
    """
    Create a new compliance verification agent.
    
    Args:
        model_name: Google model name to use
        temperature: LLM temperature for response generation
        max_iterations: Maximum number of reasoning iterations
        
    Returns:
        Configured ComplianceAgent instance
    """
    return ComplianceAgent(
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations
    )


# Convenience functions for common verification tasks

def verify_door_widths(agent: ComplianceAgent) -> Dict[str, Any]:
    """
    Verify door width compliance for all doors in the project.
    
    Args:
        agent: ComplianceAgent instance
        
    Returns:
        Verification results
    """
    query = """Please verify that all doors in the project meet minimum width requirements according to CTE DB-SI regulations. 
    
    For each door:
    1. Check the door width against minimum requirements
    2. Pay special attention to emergency exit doors (minimum 900mm)
    3. Verify accessibility requirements for accessible doors
    4. Provide specific measurements and regulation references
    
    Be thorough and systematic in your verification."""
    
    return agent.verify_compliance(query, agent_mode="door_widths")


def verify_egress_distances(agent: ComplianceAgent) -> Dict[str, Any]:
    """
    Verify egress distance compliance for all rooms in the project.
    
    Args:
        agent: ComplianceAgent instance
        
    Returns:
        Verification results
    """
    query = """Please verify that all rooms in the project have adequate egress distances according to CTE DB-SI regulations.
    
    For each room:
    1. Calculate the egress distance to the nearest exit
    2. Verify against maximum allowed distances for the room use
    3. Check that egress paths are accessible
    4. Consider occupancy load requirements
    
    Provide specific distances and regulation references for each room."""
    
    return agent.verify_compliance(query, agent_mode="egress_distances")


def comprehensive_verification(agent: ComplianceAgent) -> Dict[str, Any]:
    """
    Perform comprehensive building code compliance verification.
    
    Args:
        agent: ComplianceAgent instance
        
    Returns:
        Verification results
    """
    query = """Please perform a comprehensive building code compliance verification according to Spanish CTE regulations.
    
    This should include:
    1. Door width compliance verification for all doors
    2. Egress distance compliance verification for all rooms
    3. Accessibility compliance where applicable
    4. Fire safety compliance for fire-rated elements
    
    Provide a detailed report with:
    - Executive summary of overall compliance status
    - Detailed findings for each element checked
    - Specific recommendations for any non-compliant elements
    - Regulation references for all requirements
    
    Be systematic and thorough in your verification process."""
    
    return agent.verify_compliance(query, agent_mode="comprehensive")


# Example usage and testing functions

def run_agent_demo():
    """
    Run a demonstration of the compliance agent.
    
    This function shows how to use the agent for compliance verification.
    """
    print("=" * 70)
    print("AEC COMPLIANCE AGENT DEMO")
    print("=" * 70)
    
    # Create agent
    agent = create_compliance_agent()
    
    # Example verification query
    query = """Please verify door width compliance for all doors in the project. 
    Check each door against CTE DB-SI requirements and provide a detailed report."""
    
    print(f"\nQuery: {query}")
    print("\nRunning verification...")
    
    # Run verification
    results = agent.verify_compliance(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    
    print(f"\nAgent Response:")
    print("-" * 50)
    print(results["agent_response"])
    
    print(f"\nCompliance Summary:")
    print("-" * 50)
    summary = results["compliance_summary"]
    print(f"Total checks: {summary['total_checks']}")
    print(f"Compliant: {summary['compliant']}")
    print(f"Non-compliant: {summary['non_compliant']}")
    print(f"Compliance rate: {summary['compliance_rate']:.1%}")
    print(f"Overall status: {summary['overall_status']}")
    
    if results["issues_by_type"]:
        print(f"\nIssues by type:")
        print("-" * 50)
        for issue_type, issues in results["issues_by_type"].items():
            print(f"\n{issue_type.upper()}:")
            for issue in issues:
                print(f"  - {issue['element_id']}: {issue['message']}")
    
    print(f"\nCompleted in {results['iterations']} iterations")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Run demo if script is executed directly
    run_agent_demo()
