"""
Agent Display Helper - Clean Tool Usage Summary

This module provides clean, summarized display of agent tool usage
instead of verbose logs.
"""

import json
from typing import List, Dict, Any
try:
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
except Exception:  # Fallback if langchain_core is not installed in the environment
    class BaseMessage:  # type: ignore
        pass
    class HumanMessage(BaseMessage):  # type: ignore
        pass
    class AIMessage(BaseMessage):  # type: ignore
        pass
    class ToolMessage(BaseMessage):  # type: ignore
        pass

# Display length controls (tunable)
MAX_MESSAGE_SNIPPET = 120  # for 'error', 'message', 'answer', etc.
MAX_TEXT_SNIPPET = 90      # for generic JSON/text fallbacks
MAX_VERBOSE_TOOL_SNIPPET = 300  # hard cap for verbose tool outputs
MAX_REASONING_SNIPPET = 600
MAX_FINAL_OUTPUT_SNIPPET = 1500


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
                    return f"❌ {data['error'][:MAX_MESSAGE_SNIPPET]}..."
                
                elif 'success' in data and 'message' in data:
                    return f"✅ {data['message'][:MAX_MESSAGE_SNIPPET]}..."
                
                elif 'compliance_status' in data:
                    status = data['compliance_status']
                    door_id = data.get('door_id', 'Unknown')
                    return f"🔍 {door_id}: {status}"
                
                elif 'answer' in data:
                    return f"📋 {data['answer'][:MAX_MESSAGE_SNIPPET]}..."
                
                else:
                    # Generic JSON summary - SHORT!
                    return f"📋 {str(data)[:MAX_TEXT_SNIPPET]}..."
            
            else:
                # Generic JSON summary - SHORT!
                return f"📋 {str(data)[:MAX_TEXT_SNIPPET]}..."
        
        else:
            # Non-JSON result - SHORT!
            return f"📋 {tool_result[:MAX_TEXT_SNIPPET]}..."
            
    except (json.JSONDecodeError, KeyError, IndexError):
        # Fallback for non-JSON or malformed results - SHORT!
        return f"📋 {str(tool_result)[:MAX_TEXT_SNIPPET]}..."


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
    
    def _message_kind(m: Any) -> str:
        if isinstance(m, HumanMessage) or m.__class__.__name__ == 'HumanMessage':
            return 'human'
        if isinstance(m, AIMessage) or m.__class__.__name__ == 'AIMessage':
            return 'ai'
        if isinstance(m, ToolMessage) or m.__class__.__name__ == 'ToolMessage' or hasattr(m, 'name'):
            return 'tool'
        return 'other'

    for i, msg in enumerate(messages):
        if hasattr(msg, 'content') and msg.content and not str(msg.content).startswith("You are"):
            kind = _message_kind(msg)
            if kind == 'human':
                print(f"👤 USER: {msg.content}")
                print()
            elif kind == 'ai':
                print(f"🤖 AGENT: {msg.content}")
                print()
            elif kind == 'tool':
                tool_name = getattr(msg, 'name', 'unknown_tool')
                if show_tool_details:
                    print(f"🔧 TOOL ({tool_name}): {msg.content}")
                    print()
                else:
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
        is_tool = isinstance(msg, ToolMessage) or msg.__class__.__name__ == 'ToolMessage' or hasattr(msg, 'name')
        if is_tool:
            tool_name = getattr(msg, 'name', 'unknown_tool')
            summary = summarize_tool_result(getattr(msg, 'content', ''))
            tool_usage_summary.append(f"🔧 {tool_name}: {summary}")
    
    print("🔧 TOOLS USED:")
    for tool_summary in tool_usage_summary:
        print(f"   {tool_summary}")
    
    print(f"\n✅ Completed in {iterations} iterations")


# Example usage functions for common scenarios

def display_full_conversation(result: Dict[str, Any], title: str = "FULL CONVERSATION", show_tool_outputs: bool = False) -> None:
    """
    Display the entire conversation chronologically with explicit roles and tools.
    Shows: User messages, Agent messages, Tool names (and outputs if enabled).
    """
    messages = result.get("messages", [])
    iterations = result.get("iterations", 0)
    tools_used: List[str] = []

    print("=" * 70)
    print(f"🗣️ {title}")
    print("=" * 70 + "\n")

    def to_text(value: Any, limit: int | None = None) -> str:
        try:
            if isinstance(value, str):
                s = value
            else:
                s = json.dumps(value, ensure_ascii=False)
        except Exception:
            s = str(value)
        if limit is not None and len(s) > limit:
            return s[:limit] + "..."
        return s

    def extract_reasoning(msg: AIMessage) -> str | None:
        # Common fields models may use for chain-of-thought style summaries
        candidate_keys = ["reasoning", "analysis", "thoughts", "plan"]
        kw = getattr(msg, 'additional_kwargs', None)
        if isinstance(kw, dict):
            for key in candidate_keys:
                if key in kw and kw[key]:
                    return to_text(kw[key], MAX_REASONING_SNIPPET)
        # Sometimes reasoning is embedded in content as a dict/list
        if not isinstance(msg.content, str) and msg.content:
            try:
                if isinstance(msg.content, dict):
                    for key in candidate_keys:
                        if key in msg.content and msg.content[key]:
                            return to_text(msg.content[key], MAX_REASONING_SNIPPET)
            except Exception:
                pass
        return None

    for msg in messages:
        if hasattr(msg, 'content') and msg.content and not str(msg.content).startswith("You are"):
            if isinstance(msg, HumanMessage) or msg.__class__.__name__ == 'HumanMessage':
                print(f"- User: {msg.content}")
            elif isinstance(msg, AIMessage) or msg.__class__.__name__ == 'AIMessage':
                # Optional reasoning block
                reasoning_snippet = extract_reasoning(msg)
                if reasoning_snippet:
                    print(f"- Agent reasoning: {reasoning_snippet}")
                # Agent surface message (handles structured content too)
                print(f"- Agent: {to_text(msg.content, MAX_MESSAGE_SNIPPET)}")
                # Show explicit tool call decisions if present on the AI message
                tool_calls = []
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls = msg.tool_calls
                elif hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
                    maybe_calls = msg.additional_kwargs.get('tool_calls')
                    if maybe_calls:
                        tool_calls = maybe_calls
                for call in tool_calls:
                    name = call.get('name') if isinstance(call, dict) else getattr(call, 'name', 'unknown_tool')
                    args = call.get('args') if isinstance(call, dict) else getattr(call, 'args', {})
                    print(f"- Agent tool call → {name} args={args}")
            elif isinstance(msg, ToolMessage) or msg.__class__.__name__ == 'ToolMessage' or hasattr(msg, 'name'):
                tool_name = getattr(msg, 'name', 'unknown_tool')
                tools_used.append(tool_name)
                if show_tool_outputs:
                    content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if len(content_str) > MAX_VERBOSE_TOOL_SNIPPET:
                        content_str = content_str[:MAX_VERBOSE_TOOL_SNIPPET] + "..."
                    print(f"- Tool ({tool_name}): {content_str}")
                else:
                    print(f"- Tool ({tool_name}): {summarize_tool_result(msg.content)}")

    if tools_used:
        unique_tools = ", ".join(sorted(set(tools_used)))
        print(f"\n🔧 Tools used: {unique_tools}")

    # Print the final agent message explicitly
    final_agent_message = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.content.startswith("You are"):
            final_agent_message = m.content
            break
    if final_agent_message:
        print("\n- Agent output to the user:")
        print(to_text(final_agent_message, MAX_FINAL_OUTPUT_SNIPPET))
    else:
        print("\n- Agent output to the user: <no final agent message found>")

    print(f"\n✅ Completed in {iterations} iterations")


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


def run_agent_with_verbose_display(agent, prompt: str, max_iterations: int = 10, title: str = "FULL CONVERSATION", show_tool_outputs: bool = True):
    """
    Run agent and display the full conversation (User, Agent, Tool steps) in order.
    
    Args:
        agent: The agent instance
        prompt: The prompt to send
        max_iterations: Maximum iterations
        title: Title for display
        show_tool_outputs: If True, prints full tool outputs; otherwise summaries
    """
    from langchain_core.messages import HumanMessage
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=prompt)],
            "max_iterations": max_iterations
        })
        
        display_full_conversation(result, title=title, show_tool_outputs=show_tool_outputs)
        return result
        
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return None
