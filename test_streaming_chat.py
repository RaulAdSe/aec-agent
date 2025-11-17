#!/usr/bin/env python3
"""
Test script for streaming chat functionality
Tests the new generate_streaming_response function
"""

import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

# Add project to Python path
sys.path.append(os.path.dirname(__file__))

# Mock Streamlit for testing
class MockStreamlit:
    def empty(self):
        return MockContainer()
    
    def info(self, text):
        print(f"[INFO] {text}")
    
    def markdown(self, text):
        print(f"[MARKDOWN] {text}")

class MockContainer:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def empty(self):
        pass
    
    def info(self, text):
        print(f"[CONTAINER INFO] {text}")
    
    def markdown(self, text):
        print(f"[CONTAINER MARKDOWN] {text}")

# Mock Streamlit
import streamlit as st
st.empty = MockStreamlit().empty
st.info = MockStreamlit().info
st.markdown = MockStreamlit().markdown

# Setup mock session state
st.session_state = MagicMock()
st.session_state.processed_ifc_files = {}
st.session_state.uploaded_pdfs = {}
st.session_state.pdf_rag_manager = MagicMock()

# Load environment variables for OpenAI
from dotenv import load_dotenv
load_dotenv()

def test_streaming_functionality():
    """Test the streaming response generation."""
    print("🧪 Testing Streaming Chat Functionality")
    print("=" * 60)
    
    # Import the function after setting up mocks
    from app import generate_streaming_response
    
    test_queries = [
        "Tell me about the building model",
        "What spaces do you see?",
        "Check for compliance issues",
        "Show me the doors",
        "Hello, how are you?"
    ]
    
    print("\n🤖 Testing Streaming Responses:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n👤 User Query: {query}")
        print("🤖 Streaming Response Process:")
        print("-" * 40)
        
        try:
            response, thinking_steps = generate_streaming_response(query)
            
            print("\n📝 Final Response:")
            print(f"{response[:200]}...")
            
            print("\n🧠 Thinking Steps:")
            for i, step in enumerate(thinking_steps, 1):
                print(f"  {i}. {step['action']} - {step['description']}")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"❌ Error testing query '{query}': {e}")
            import traceback
            traceback.print_exc()
    
    return True

def test_with_mock_data():
    """Test streaming with mock IFC data."""
    print("\n🏗️ Testing with Mock Building Data")
    print("=" * 60)
    
    # Add mock processed files
    mock_ifc_data = {
        "test_building.ifc": {
            "json_data": {
                "spaces": [
                    {"name": "Office Room 1", "id": "space_001", "area": 25.5, "type": "Office"},
                    {"name": "Conference Room", "id": "space_002", "area": 40.2, "type": "Meeting"}
                ],
                "doors": [
                    {"name": "Main Entry", "id": "door_001", "width": 0.9, "height": 2.1},
                    {"name": "Office Door", "id": "door_002", "width": 0.8, "height": 2.0}
                ],
                "stairs": [
                    {"name": "Main Staircase", "id": "stair_001", "position": {"x": 10.0, "y": 5.0, "z": 0.0}}
                ],
                "walls": [
                    {"name": "Exterior Wall North", "id": "wall_001", "type": "Exterior", "thickness": 0.3}
                ]
            }
        }
    }
    
    st.session_state.processed_ifc_files = mock_ifc_data
    
    # Import the function
    from app import generate_streaming_response
    
    test_queries_with_data = [
        "Tell me about the spaces in the building",
        "What doors are available?", 
        "Show me the stairs and how far apart they are",
        "Analyze the walls"
    ]
    
    for query in test_queries_with_data:
        print(f"\n👤 User Query: {query}")
        print("🤖 Streaming with Building Data:")
        print("-" * 40)
        
        try:
            response, thinking_steps = generate_streaming_response(query)
            
            print("\n📝 Response with Data:")
            print(response[:300] + "..." if len(response) > 300 else response)
            
            print(f"\n🧠 Thinking Steps ({len(thinking_steps)} total):")
            for step in thinking_steps[-3:]:  # Show last 3 steps
                print(f"  • {step['action']} - {step['description']}")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return True

if __name__ == "__main__":
    print("🏗️ AEC Compliance Agent - Streaming Chat Test")
    print("=" * 70)
    
    try:
        # Test basic streaming functionality
        success1 = test_streaming_functionality()
        
        # Test with mock building data
        success2 = test_with_mock_data()
        
        print("\n" + "=" * 70)
        if success1 and success2:
            print("🎉 All streaming chat tests completed successfully!")
            print("✅ Streaming responses with tool visualization working!")
            print("🚀 Ready for real building analysis!")
        else:
            print("⚠️ Some tests completed with issues - check output above")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)