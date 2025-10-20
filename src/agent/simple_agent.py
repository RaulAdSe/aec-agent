"""
Simple ReAct Agent Implementation without LangGraph.

This is a simplified version that handles tool calls correctly.
"""

import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from .tools import (
    get_room_info,
    get_door_info, 
    list_all_doors,
    check_door_width_compliance,
    query_normativa,
    calculate_egress_distance
)


class SimpleReActAgent:
    """Simple ReAct Agent for AEC compliance verification."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_iterations: int = 10
    ):
        """
        Initialize the Simple ReAct Agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for response generation
            max_iterations: Maximum number of reasoning iterations
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=2000
        )
        
        self.max_iterations = max_iterations
        
        # Define available tools
        self.tools = {
            "get_room_info": get_room_info,
            "get_door_info": get_door_info,
            "list_all_doors": list_all_doors,
            "check_door_width_compliance": check_door_width_compliance,
            "query_normativa": query_normativa,
            "calculate_egress_distance": calculate_egress_distance
        }
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values()))
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent with a query.
        
        Args:
            input_data: Dictionary containing 'messages' and 'iterations'
            
        Returns:
            Dictionary with agent response and metadata
        """
        messages = input_data.get("messages", [])
        iterations = input_data.get("iterations", 0)
        
        if not messages:
            return {
                "messages": [],
                "iterations": 0,
                "final_answer": "No messages provided"
            }
        
        # Get the user query
        user_message = messages[0]
        if not isinstance(user_message, HumanMessage):
            return {
                "messages": [],
                "iterations": 0,
                "final_answer": "Invalid message type"
            }
        
        query = user_message.content
        
        # Create system prompt
        system_prompt = self._create_system_prompt(query)
        
        # Start conversation
        conversation = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        # Run ReAct loop
        for iteration in range(self.max_iterations):
            try:
                # Get agent response
                response = self.llm_with_tools.invoke(conversation)
                conversation.append(response)
                
                # Check if agent made tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    # Execute tool calls
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call['args']
                        tool_id = tool_call['id']
                        
                        # Execute the tool
                        if tool_name in self.tools:
                            try:
                                result = self.tools[tool_name].invoke(tool_args)
                                tool_message = ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_id
                                )
                                conversation.append(tool_message)
                            except Exception as e:
                                tool_message = ToolMessage(
                                    content=f"Error executing {tool_name}: {str(e)}",
                                    tool_call_id=tool_id
                                )
                                conversation.append(tool_message)
                        else:
                            tool_message = ToolMessage(
                                content=f"Unknown tool: {tool_name}",
                                tool_call_id=tool_id
                            )
                            conversation.append(tool_message)
                else:
                    # No tool calls, agent provided final answer
                    break
                    
            except Exception as e:
                # Handle errors gracefully
                error_message = AIMessage(content=f"Error: {str(e)}")
                conversation.append(error_message)
                break
        
        # Extract final answer
        final_answer = conversation[-1].content if conversation else "No response generated"
        
        return {
            "messages": conversation,
            "iterations": len(conversation),
            "final_answer": final_answer
        }
    
    def run(self, question: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Run the agent with a simple question string.
        
        Args:
            question: The question to ask the agent
            max_iterations: Maximum number of iterations (optional)
            
        Returns:
            Dictionary with agent response and metadata
        """
        if max_iterations is not None:
            original_max = self.max_iterations
            self.max_iterations = max_iterations
        
        try:
            # Create input data for invoke method
            input_data = {
                "messages": [HumanMessage(content=question)],
                "iterations": 0
            }
            
            result = self.invoke(input_data)
            return result
        finally:
            if max_iterations is not None:
                self.max_iterations = original_max
    
    def run_with_streaming(self, question: str, max_iterations: int = None) -> Dict[str, Any]:
        """
        Run the agent with streaming output showing reasoning steps.
        
        Args:
            question: The question to ask the agent
            max_iterations: Maximum number of iterations (optional)
            
        Returns:
            Dictionary with agent response and metadata
        """
        if max_iterations is not None:
            original_max = self.max_iterations
            self.max_iterations = max_iterations
        
        try:
            print(f"\n💬 Question: {question}")
            print("\n" + "="*80)
            print("🔍 AGENT REASONING PROCESS (LIVE)")
            print("="*80)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(question)
            
            # Start conversation
            conversation = [
                HumanMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            
            # Run ReAct loop with live updates
            for iteration in range(self.max_iterations):
                print(f"\n🔄 ITERATION {iteration + 1}")
                print("-" * 50)
                
                try:
                    # Get agent response
                    print("🧠 Agent reasoning...")
                    response = self.llm_with_tools.invoke(conversation)
                    conversation.append(response)
                    
                    print(f"💭 Agent thought: {response.content[:150]}...")
                    
                    # Check if agent made tool calls
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        print(f"\n🔧 TOOL CALLS ({len(response.tool_calls)}):")
                        
                        # Execute tool calls
                        for i, tool_call in enumerate(response.tool_calls, 1):
                            tool_name = tool_call['name']
                            tool_args = tool_call['args']
                            tool_id = tool_call['id']
                            
                            print(f"\n  {i}. 🔨 Executing: {tool_name}")
                            print(f"     📝 Arguments: {tool_args}")
                            
                            # Execute the tool
                            if tool_name in self.tools:
                                try:
                                    result = self.tools[tool_name].invoke(tool_args)
                                    print(f"     ✅ Result: {str(result)[:100]}...")
                                    
                                    # Create tool message
                                    from langchain_core.messages import ToolMessage
                                    tool_message = ToolMessage(
                                        content=str(result),
                                        tool_call_id=tool_id
                                    )
                                    conversation.append(tool_message)
                                    
                                except Exception as e:
                                    print(f"     ❌ Error: {str(e)}")
                                    tool_message = ToolMessage(
                                        content=f"Error: {str(e)}",
                                        tool_call_id=tool_id
                                    )
                                    conversation.append(tool_message)
                            else:
                                print(f"     ❌ Unknown tool: {tool_name}")
                                print(f"     Available tools: {list(self.tools.keys())}")
                                tool_message = ToolMessage(
                                    content=f"Unknown tool: {tool_name}",
                                    tool_call_id=tool_id
                                )
                                conversation.append(tool_message)
                    else:
                        # No tool calls, agent provided final answer
                        print(f"\n✅ FINAL ANSWER:")
                        print(f"📝 {response.content}")
                        break
                        
                except Exception as e:
                    print(f"\n❌ Error in iteration {iteration + 1}: {str(e)}")
                    break
            
            print(f"\n📊 Completed in {len(conversation)} steps")
            
            # Extract final answer
            final_answer = conversation[-1].content if conversation else "No response generated"
            
            return {
                "messages": conversation,
                "iterations": len(conversation),
                "final_answer": final_answer
            }
        finally:
            if max_iterations is not None:
                self.max_iterations = original_max
    
    def _create_system_prompt(self, task: str) -> str:
        """Create system prompt for the agent."""
        return f"""You are an expert AEC (Architecture, Engineering, Construction) compliance verification agent.

Your task: {task}

You have access to the following tools:
- get_room_info: Get detailed information about a room
- get_door_info: Get detailed information about a door  
- list_all_doors: List all doors in the project
- check_door_width_compliance: Check if a door meets width requirements
- query_normativa: Query building codes and regulations
- calculate_egress_distance: Calculate evacuation distances

Use these tools to gather information and provide comprehensive answers about building compliance.

When you have enough information, provide a clear, detailed response without making additional tool calls."""


def create_simple_compliance_agent() -> SimpleReActAgent:
    """Create a simple compliance agent instance."""
    return SimpleReActAgent()
