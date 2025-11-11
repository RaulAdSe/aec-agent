#!/usr/bin/env python3
"""
Test script for the AEC Compliance Agent.

Tests all tools and workflows to identify errors.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aec_agent.agent import create_agent


def test_agent_creation():
    """Test basic agent creation."""
    print("🔧 Testing agent creation...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY not found. Please set your OpenAI API key.")
        return None
    
    try:
        agent = create_agent(verbose=False, temperature=1.0)
        status = agent.get_status()
        print(f"✅ Agent created: {status['name']}")
        print(f"📋 Tools available: {status['tools']}")
        return agent
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return None


def test_individual_tools():
    """Test each tool individually without the agent."""
    print("\n🛠️ Testing individual tools...")
    
    # Test building data tools
    try:
        from aec_agent.tools.building_data_toolkit import (
            load_building_data, get_all_elements, get_all_properties
        )
        
        print("✅ Building data toolkit imported")
        
        # Test with sample data if available
        sample_file = "data/out/FM-ARC_v2.json"
        if os.path.exists(sample_file):
            print(f"📊 Testing with {sample_file}...")
            
            # Test load_building_data
            result = load_building_data(sample_file)
            if result.get("status") == "success":
                print("✅ load_building_data works")
                
                # Test get_all_elements
                spaces_result = get_all_elements("spaces")
                if spaces_result.get("status") == "success":
                    spaces = spaces_result.get("data", [])
                    print(f"✅ get_all_elements works ({len(spaces)} spaces found)")
                    
                    # Test get_all_properties on first space
                    if spaces:
                        props_result = get_all_properties(spaces[0]["id"])
                        if props_result.get("status") == "success":
                            print("✅ get_all_properties works")
                        else:
                            print(f"❌ get_all_properties failed: {props_result.get('logs')}")
                else:
                    print(f"❌ get_all_elements failed: {spaces_result.get('logs')}")
            else:
                print(f"❌ load_building_data failed: {result.get('logs')}")
        else:
            print(f"⚠️ Sample file not found: {sample_file}")
            
    except Exception as e:
        print(f"❌ Building data tools error: {e}")
    
    # Test compliance search tools
    try:
        from aec_agent.tools.compliance_search import check_knowledge_base_status
        
        print("✅ Compliance search imported")
        
        # Test knowledge base status
        kb_result = check_knowledge_base_status()
        print(f"✅ Knowledge base status: {kb_result.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Compliance search error: {e}")


def test_simple_queries(agent):
    """Test simple queries that don't require complex tool usage."""
    print("\n💬 Testing simple queries...")
    
    test_queries = [
        "Hello, what can you help me with?",
        "What tools do you have available?",
        "Explain what you can do for building compliance analysis"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = agent.process(query)
            if result['status'] == 'success':
                print(f"✅ Response: {result['response'][:100]}...")
            else:
                print(f"❌ Error: {result['message']}")
        except Exception as e:
            print(f"❌ Exception: {e}")


def test_knowledge_base_query(agent):
    """Test knowledge base related queries."""
    print("\n📚 Testing knowledge base queries...")
    
    queries = [
        "Check the knowledge base status",
        "How many compliance documents are available?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = agent.process(query)
            if result['status'] == 'success':
                print(f"✅ Response: {result['response']}")
            else:
                print(f"❌ Error: {result['message']}")
                print(f"   Details: {result}")
        except Exception as e:
            print(f"❌ Exception: {e}")


def test_building_data_workflow(agent):
    """Test building data analysis workflow."""
    print("\n🏗️ Testing building data workflow...")
    
    sample_file = "data/out/FM-ARC_v2.json"
    if not os.path.exists(sample_file):
        print(f"⚠️ Sample file not found: {sample_file}")
        return
    
    queries = [
        f"Load building data from {sample_file}",
        "How many spaces are in the building?",
        "What types of elements are available?",
        "Show me the properties of the first space"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = agent.process(query)
            if result['status'] == 'success':
                response = result['response']
                print(f"✅ Response: {response[:200]}...")
            else:
                print(f"❌ Error: {result['message']}")
                print(f"   Raw result: {result}")
        except Exception as e:
            print(f"❌ Exception: {e}")


def main():
    """Run all tests."""
    print("🚀 Testing AEC Compliance Agent\n")
    
    # Test 1: Agent creation
    agent = test_agent_creation()
    if not agent:
        print("\n❌ Cannot continue - agent creation failed")
        return
    
    # Test 2: Individual tools
    test_individual_tools()
    
    # Test 3: Simple queries
    test_simple_queries(agent)
    
    # Test 4: Knowledge base queries
    test_knowledge_base_query(agent)
    
    # Test 5: Building data workflow
    test_building_data_workflow(agent)
    
    print("\n🎉 Testing completed!")
    print("\nIf you see errors above, they need to be fixed.")


if __name__ == "__main__":
    main()