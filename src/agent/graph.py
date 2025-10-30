"""
ReAct Agent Implementation using LangGraph

This module implements the ReAct (Reasoning + Acting) agent for autonomous
AEC compliance verification using LangGraph for state management and workflow.
"""

import json
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from .tools import (
    load_json_data,
    list_all_doors,
    create_circulation_graph_tool,
    find_nearest_door_tool,
    calculate_clearance_tool,
    check_door_width_compliance,
    query_normativa
)
from .memory import SlidingWindowMemory


class AgentState(TypedDict):
    """State for the ReAct agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    iterations: int
    max_iterations: int
    current_task: Optional[str]
    compliance_results: List[Dict[str, Any]]
    final_answer: Optional[str]
    memory: Optional[SlidingWindowMemory]


class ReActAgent:
    """ReAct Agent for AEC compliance verification."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_iterations: int = 10,
        memory_window_size: int = 5
    ):
        """
        Initialize the ReAct Agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Model temperature
            max_iterations: Maximum number of reasoning iterations
            memory_window_size: Size of the sliding window memory
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=2000
        )
        
        self.max_iterations = max_iterations
        self.memory_window_size = memory_window_size
        
        # Define available tools (simplified to 7 essential tools - JSON only)
        self.tools = [
            load_json_data,
            list_all_doors,
            create_circulation_graph_tool,
            find_nearest_door_tool,
            calculate_clearance_tool,
            check_door_width_compliance,
            query_normativa
        ]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the LangGraph workflow for the ReAct agent."""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "tools": "tools",
                "final_summary": "agent",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent reasoning node."""
        messages = state["messages"]
        iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        memory = state.get("memory")
        
        # Initialize memory if not present
        if memory is None:
            memory = SlidingWindowMemory(window_size=self.memory_window_size)
            # Add existing messages to memory
            for msg in messages:
                memory.add_message(msg)
        
        # Check if we've exceeded max iterations
        if iterations >= max_iterations:
            final_message = AIMessage(
                content=f"I've reached the maximum number of iterations ({max_iterations}). "
                       f"Here's what I've found so far: {self._summarize_results(state)}"
            )
            memory.add_message(final_message)
            return {
                **state,
                "messages": messages + [final_message],
                "iterations": iterations + 1,
                "final_answer": final_message.content,
                "memory": memory
            }
        
        # Get the task from the first human message
        if not messages:
            return state
        
        # Find the first human message for the task
        task_message = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                task_message = msg
                break
        
        current_task = task_message.content if task_message else "Continue compliance verification"
        
        # Check if this is a final summary request
        tool_message_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))
        
        # Count unique tool types used
        tool_names_used = set()
        for msg in messages:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'name'):
                tool_names_used.add(msg.name)
        
        unique_tool_count = len(tool_names_used)
        
        # Determine if this should be final summary or continued analysis
        is_final_summary = (tool_message_count >= 5 and unique_tool_count >= 3 and iterations > 0)
        needs_more_analysis = (tool_message_count > 0 and 
                              (unique_tool_count < 3 or tool_message_count < 5) and 
                              iterations < max_iterations - 2)
        
        # Create system prompt for reasoning
        system_prompt = self._create_system_prompt(current_task, iterations, is_final_summary, needs_more_analysis, unique_tool_count, tool_message_count)
        
        # Use the existing messages directly (this is key for LangGraph)
        # LangGraph expects the messages to flow naturally through the graph
        if is_final_summary:
            # For final summary, use LLM without tools to force text response
            response = self.llm.invoke([
                HumanMessage(content=system_prompt)
            ] + messages)
        else:
            response = self.llm_with_tools.invoke([
                HumanMessage(content=system_prompt)
            ] + messages)
        
        # Add response to memory
        memory.add_message(response)
        
        return {
            **state,
            "messages": messages + [response],
            "iterations": iterations + 1,
            "current_task": current_task,
            "memory": memory
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end the workflow."""
        messages = state["messages"]
        iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        
        # End if max iterations reached
        if iterations >= max_iterations:
            return "end"
        
        # Check if the last message has tool calls
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if we've used any tools and should provide a final summary
        # Count how many ToolMessage instances we have
        tool_message_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))
        ai_message_count = sum(1 for msg in messages if isinstance(msg, AIMessage))
        
        # Count unique tool types used (to ensure comprehensive analysis)
        tool_names_used = set()
        for msg in messages:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'name'):
                tool_names_used.add(msg.name)
        
        unique_tool_count = len(tool_names_used)
        
        # Require at least 3 different tools and 5 total tool calls before allowing final summary
        # unless we're near max iterations
        minimum_tools_required = 3
        minimum_calls_required = 5
        
        # If we haven't used enough tools yet, continue with tools (don't go to final summary yet)
        if (unique_tool_count < minimum_tools_required or 
            tool_message_count < minimum_calls_required) and iterations < max_iterations - 2:
            # If the last message has no tool calls, the agent thinks it's done but we want more analysis
            return "final_summary"  # This will prompt the agent to continue analysis, not end
        
        # If we've used enough tools and the last AI message doesn't contain a comprehensive summary,
        # give the agent one more chance to provide a final synthesis
        if tool_message_count >= minimum_calls_required and unique_tool_count >= minimum_tools_required:
            # Check if the last AI message looks like a final summary
            if isinstance(last_message, AIMessage):
                content = str(last_message.content).lower() if last_message.content else ""
                # If it's empty, very short, or doesn't contain summary keywords, continue
                if len(content) < 200 or not any(keyword in content for keyword in 
                    ['summary', 'conclusion', 'findings', 'recommendations', 'compliance assessment', 'analysis']):
                    # Only allow one more iteration for final summary
                    if iterations < max_iterations - 1:
                        return "final_summary"
        
        # If no tool calls, end the workflow
        return "end"
    
    def _create_system_prompt(self, task: str, iteration: int, is_final_summary: bool = False, needs_more_analysis: bool = False, unique_tool_count: int = 0, tool_message_count: int = 0) -> str:
        """Create system prompt for the agent."""
        if is_final_summary:
            return f"""You are an expert AEC compliance verification agent providing a FINAL SUMMARY.

Your original task: {task}

You have completed {iteration} iterations and used multiple tools to gather information. 

**IMPORTANT: DO NOT USE ANY MORE TOOLS - PROVIDE A COMPREHENSIVE FINAL SUMMARY**

Based on the tool results you've gathered, provide a comprehensive final analysis that includes:

1. **Summary of findings** - What did you discover about the building?
2. **Compliance assessment** - What meets/doesn't meet requirements?
3. **Key risks or issues** - What should be prioritized?
4. **Actionable recommendations** - What specific steps should be taken?

Structure your response as a professional compliance report. Be thorough but concise.
This is your final message - make it count!"""
        
        elif needs_more_analysis:
            return f"""You are an expert AEC compliance verification agent continuing your analysis.

Your original task: {task}

**ANALYSIS PROGRESS CHECK:**
- You have used {unique_tool_count} different tools (need at least 3)
- You have made {tool_message_count} tool calls (need at least 5) 
- Current iteration: {iteration}

**YOU MUST CONTINUE ANALYSIS - DO NOT PROVIDE A FINAL SUMMARY YET**

Based on the Tool Usage Strategy, you still need to use more tools. Continue with:

1. If you haven't done comprehensive door width compliance checking, use `check_door_width_compliance(door_id)` on multiple doors (at least 5 different doors)
2. If you haven't checked building connectivity, use `create_circulation_graph_tool()` 
3. If you haven't consulted regulations, use `query_normativa(question)` for DB-SUA and DB-SI requirements
4. If you need proximity analysis, use `find_nearest_door_tool(point_x, point_y)`
5. If you need clearance checks, use `calculate_clearance_tool(elem1_type, elem1_id, elem2_type, elem2_id)`

**Continue your systematic analysis. Use tools to gather more evidence before concluding.**

Reason about what additional information you need, then use the appropriate tools to gather it."""
        
        return f"""You are an expert AEC (Architecture, Engineering, Construction) compliance verification agent.

Your task: {task}

You have access to 7 essential tools for building compliance verification:

**Data & Building Information:**
1. **load_json_data(json_file_path)** - Load building data from JSON files
2. **list_all_doors()** - List all doors in the project

**Core Geometric Analysis:**
3. **create_circulation_graph_tool()** - Create circulation graph for pathfinding and connectivity
4. **find_nearest_door_tool(point_x, point_y)** - Find nearest door to any point
5. **calculate_clearance_tool(elem1_type, elem1_id, elem2_type, elem2_id)** - Calculate clearance between elements

**Compliance Verification:**
6. **check_door_width_compliance(door_id)** - Check if a door meets width requirements
7. **query_normativa(question)** - Query Spanish building codes (CTE) for regulations

**ReAct Process:**
1. **Reason** about what information you need
2. **Act** by using the appropriate tools
3. **Observe** the results and reason about next steps
4. **Repeat** until you have a complete answer

**Current iteration:** {iteration}

**Tool Usage Strategy (follow unless the user explicitly requests a single check):**
1. Start by calling `list_all_doors()` to understand the dataset.
2. Call `query_normativa()` to retrieve current thresholds/clauses before judging compliance (DB-SUA, DB-SI).
3. Sample multiple doors with `check_door_width_compliance(door_id)` — at least 5 calls across different doors unless the dataset is smaller.
4. Construct a connectivity basis with `create_circulation_graph_tool()` to prepare for egress assessment; if needed, use `find_nearest_door_tool()` and `calculate_clearance_tool()` to investigate potential bottlenecks.
5. When uncertain, consult `query_normativa()` again for clarification.

Aim to use at least 3 different tools and make several tool calls before concluding, unless the user asks you to stop early or the evidence is conclusive.

**Instructions:**
- Use tools to gather information systematically and iteratively
- Query building codes proactively for regulatory grounding
- Check compliance for each relevant element you inspect
- Provide clear, actionable conclusions
- When you have enough information, provide a final comprehensive answer

**Important:** Always use tools to verify compliance rather than making assumptions. Query the CTE regulations when checking requirements."""

    def _summarize_results(self, state: AgentState) -> str:
        """Summarize the compliance results."""
        results = state.get("compliance_results", [])
        if not results:
            return "No compliance results available."
        
        summary = "Compliance Summary:\n"
        for result in results:
            summary += f"- {result.get('element', 'Unknown')}: {result.get('status', 'Unknown')}\n"
        
        return summary
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent with a task.
        
        Args:
            input_data: Dictionary containing 'messages' and optional 'max_iterations'
            
        Returns:
            Dictionary with agent response and metadata
        """
        # Initialize memory
        memory = SlidingWindowMemory(window_size=self.memory_window_size)
        
        # Initialize state
        initial_state = {
            "messages": input_data.get("messages", []),
            "iterations": 0,
            "max_iterations": input_data.get("max_iterations", self.max_iterations),
            "current_task": None,
            "compliance_results": [],
            "final_answer": None,
            "memory": memory
        }
        
        # Run the agent
        result = self.graph.invoke(initial_state)
        
        return {
            "messages": result["messages"],
            "iterations": result["iterations"],
            "final_answer": result.get("final_answer"),
            "compliance_results": result.get("compliance_results", []),
            "memory_stats": result.get("memory", memory).get_memory_stats()
        }


def create_compliance_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_iterations: int = 10,
    memory_window_size: int = 5
) -> ReActAgent:
    """
    Create a ReAct compliance verification agent with memory.
    
    Args:
        model_name: OpenAI model to use
        temperature: Model temperature
        max_iterations: Maximum number of reasoning iterations
        memory_window_size: Size of the sliding window memory (default: 5)
        
    Returns:
        Configured ReActAgent instance with memory
    """
    return ReActAgent(
        model_name=model_name,
        temperature=temperature,
        max_iterations=max_iterations,
        memory_window_size=memory_window_size
    )


# Example usage and testing
if __name__ == "__main__":
    # Create agent
    agent = create_compliance_agent()
    
    # Test with a simple query
    result = agent.invoke({
        "messages": [HumanMessage(content="List all doors in the project and check their compliance")]
    })
    
    print("Agent Response:")
    for message in result["messages"]:
        if hasattr(message, 'content'):
            print(f"- {message.__class__.__name__}: {message.content}")
    
    print(f"\nCompleted in {result['iterations']} iterations")