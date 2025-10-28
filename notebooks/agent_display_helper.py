"""
Agent Display Helper - Clean Tool Usage Summary

This module provides clean, summarized display of agent tool usage
instead of verbose logs.
"""

import json
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage


def summarize_tool_result(tool_result: str) -> str:
    """
    Summarize a tool result to MAX 2 lines, then "..."
    
    Args:
        tool_result: The tool result string (usually JSON)
        
    Returns:
        Summarized result string (max 2 lines)
    """
    try:
        # Try to parse as JSON (both objects and arrays)
        if isinstance(tool_result, str) and (tool_result.strip().startswith('{') or tool_result.strip().startswith('[')):
            data = json.loads(tool_result)
            
            # Handle different tool result types - KEEP IT SHORT!
            if isinstance(data, list) and len(data) > 0:
                # Handle list results (like door lists) - SHORT!
                if isinstance(data[0], dict) and 'id' in data[0]:
                    return f"📋 {len(data)} doors ({data[0].get('id', 'item')})"
                else:
                    return f"📋 {len(data)} items"
            
            elif isinstance(data, dict):
                if 'error' in data:
                    return f"❌ {data['error'][:40]}..."
                
                elif 'success' in data and 'message' in data:
                    return f"✅ {data['message'][:40]}..."
                
                elif 'compliance_status' in data:
                    status = data['compliance_status']
                    door_id = data.get('door_id', 'Unknown')
                    return f"🔍 {door_id}: {status}"
                
                elif 'answer' in data:
                    return f"📋 {data['answer'][:40]}..."
                
                else:
                    # Generic JSON summary - SHORT!
                    return f"📋 {str(data)[:30]}..."
            
            else:
                # Generic JSON summary - SHORT!
                return f"📋 {str(data)[:30]}..."
        
        else:
            # Non-JSON result - SHORT!
            return f"📋 {tool_result[:30]}..."
            
    except (json.JSONDecodeError, KeyError, IndexError):
        # Fallback for non-JSON or malformed results - SHORT!
        return f"📋 {str(tool_result)[:30]}..."


def display_agent_conversation(result: Dict[str, Any], show_tool_details: bool = False) -> None:
    """
    Display agent conversation with clean tool usage summary.
    
    Args:
        result: Agent invoke result
        show_tool_details: If True, show full tool details (default: False)
    """
    messages = result.get("messages", [])
    iterations = result.get("iterations", 0)
    
    print("🤖 AGENT CONVERSATION:")
    print("-" * 50)
    
    tool_usage_summary = []
    
    for i, msg in enumerate(messages):
        if hasattr(msg, 'content') and msg.content and not msg.content.startswith("You are"):
            
            if isinstance(msg, HumanMessage):
                print(f"👤 USER: {msg.content}")
                print()
                
            elif isinstance(msg, AIMessage):
                print(f"🤖 AGENT: {msg.content}")
                print()
                
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, 'name', 'unknown_tool')
                
                if show_tool_details:
                    # Show full tool details
                    print(f"🔧 TOOL ({tool_name}): {msg.content}")
                    print()
                else:
                    # Show summarized tool usage
                    summary = summarize_tool_result(msg.content)
                    tool_usage_summary.append(f"🔧 {tool_name}: {summary}")
    
    # Show tool usage summary
    if tool_usage_summary and not show_tool_details:
        print("🔧 TOOLS USED:")
        for tool_summary in tool_usage_summary:
            print(f"   {tool_summary}")
        print()
    
    print(f"✅ Completed in {iterations} iterations")


def display_agent_analysis(result: Dict[str, Any], title: str = "AGENT ANALYSIS") -> None:
    """
    Display agent analysis with clean formatting.
    
    Args:
        result: Agent invoke result
        title: Title for the analysis
    """
    print("=" * 70)
    print(f"🤖 {title}")
    print("=" * 70 + "\n")
    
    display_agent_conversation(result, show_tool_details=False)
    
    print("✅ Agent completed analysis successfully!")


def display_tool_summary_only(result: Dict[str, Any]) -> None:
    """
    Display only the tool usage summary, no conversation.
    
    Args:
        result: Agent invoke result
    """
    messages = result.get("messages", [])
    iterations = result.get("iterations", 0)
    
    tool_usage_summary = []
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = getattr(msg, 'name', 'unknown_tool')
            summary = summarize_tool_result(msg.content)
            tool_usage_summary.append(f"🔧 {tool_name}: {summary}")
    
    print("🔧 TOOLS USED:")
    for tool_summary in tool_usage_summary:
        print(f"   {tool_summary}")
    
    print(f"\n✅ Completed in {iterations} iterations")


# Example usage functions for common scenarios

def run_agent_with_clean_display(agent, prompt: str, max_iterations: int = 10, title: str = "AGENT ANALYSIS"):
    """
    Run agent and display results with clean formatting.
    
    Args:
        agent: The agent instance
        prompt: The prompt to send
        max_iterations: Maximum iterations
        title: Title for display
    """
    from langchain_core.messages import HumanMessage
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=prompt)],
            "max_iterations": max_iterations
        })
        
        display_agent_analysis(result, title)
        return result
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return None


def run_agent_tools_only(agent, prompt: str, max_iterations: int = 10):
    """
    Run agent and show tool usage summary + final agent message.
    
    Args:
        agent: The agent instance
        prompt: The prompt to send
        max_iterations: Maximum iterations
    """
    from langchain_core.messages import HumanMessage
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=prompt)],
            "max_iterations": max_iterations
        })
        
        # Show tool summary
        display_tool_summary_only(result)
        
        # Show final agent message
        messages = result.get("messages", [])
        final_agent_message = None
        
        # Find the last AI message (agent's final response)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("You are"):
                final_agent_message = msg.content
                break
        
        if final_agent_message:
            print(f"\n🤖 AGENT RESPONSE:")
            print(f"{final_agent_message}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return None
