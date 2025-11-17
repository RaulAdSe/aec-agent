#!/usr/bin/env python3
"""
Test improved chat responses with actual building data analysis
"""

import os
import sys
import json
from pathlib import Path

# Add project to Python path  
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Mock Streamlit session state for testing
class MockSessionState:
    def __init__(self):
        # Load actual processed IFC data if available
        processed_dir = Path("data/processed_ifc")
        self.processed_ifc_files = {}
        self.uploaded_pdfs = {}
        
        if processed_dir.exists():
            for json_file in processed_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    self.processed_ifc_files[json_file.stem] = {
                        "json_data": data,
                        "file_size": json_file.stat().st_size,
                        "processed_at": "2023-11-17"
                    }
                except:
                    continue

# Import after setting up mock
sys.path.append('.')

def test_improved_chat_responses():
    """Test the improved chat responses with actual building data."""
    print("🧪 Testing Improved Chat Responses")
    print("=" * 60)
    
    # Mock session state
    mock_session = MockSessionState()
    
    if not mock_session.processed_ifc_files:
        print("⚠️ No processed IFC files found in data/processed_ifc/")
        print("   Run the IFC processing first or use test data")
        return False
    
    print(f"✅ Found {len(mock_session.processed_ifc_files)} processed IFC files")
    
    # Import the generate_response function by mocking st.session_state
    import streamlit as st
    from unittest.mock import MagicMock
    
    # Mock Streamlit session state
    st.session_state = MagicMock()
    st.session_state.processed_ifc_files = mock_session.processed_ifc_files
    st.session_state.uploaded_pdfs = mock_session.uploaded_pdfs
    st.session_state.pdf_rag_manager = MagicMock()
    
    # Import generate_response function
    from app import generate_response
    
    # Test queries that should now work better
    test_queries = [
        "Tell me about the stairs",
        "How far away are the stairs from each other?", 
        "Show me the doors",
        "What spaces do you see?",
        "Tell me about the walls",
        "What doors are available?",
        "Show me the building elements",
        "Where are the stairs located?"
    ]
    
    print("\n🤖 Testing Chat Responses:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        print("🤖 Assistant:")
        
        try:
            response = generate_response(query)
            
            # Check if response is meaningful (not just generic)
            if "I have access to" in response and "Ask me about" in response:
                print("❌ Generic response (not using building data)")
                print(f"   Response: {response[:100]}...")
            else:
                print("✅ Specific response with building data")
                # Show first 200 chars of response
                preview = response[:200] + "..." if len(response) > 200 else response
                print(f"   Response: {preview}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test with actual building data summary
    print("\n📊 Building Data Summary:")
    print("=" * 40)
    
    total_spaces = 0
    total_doors = 0
    total_walls = 0
    total_stairs = 0
    
    for file_name, file_data in mock_session.processed_ifc_files.items():
        json_data = file_data['json_data']
        spaces = len(json_data.get('spaces', []))
        doors = len(json_data.get('doors', []))
        walls = len(json_data.get('walls', []))
        stairs = len(json_data.get('stairs', []))
        
        total_spaces += spaces
        total_doors += doors
        total_walls += walls
        total_stairs += stairs
        
        print(f"📄 {file_name}:")
        print(f"   Spaces: {spaces}, Doors: {doors}, Walls: {walls}, Stairs: {stairs}")
    
    print(f"\n🏗️ Total Elements:")
    print(f"   Spaces: {total_spaces}")
    print(f"   Doors: {total_doors}")
    print(f"   Walls: {total_walls}")
    print(f"   Stairs: {total_stairs}")
    
    # Test specific element analysis
    if total_stairs > 0:
        print(f"\n🪜 Stairs Analysis:")
        for file_name, file_data in mock_session.processed_ifc_files.items():
            stairs = file_data['json_data'].get('stairs', [])
            if stairs:
                print(f"   {file_name}: {len(stairs)} stairs")
                for i, stair in enumerate(stairs[:3], 1):  # Show first 3
                    name = stair.get('name', f'Stair {i}')
                    position = stair.get('position', {})
                    print(f"     - {name}: {position}")
    
    return True


if __name__ == "__main__":
    print("🏗️ AEC Compliance Agent - Improved Chat Test")
    print("=" * 70)
    
    try:
        success = test_improved_chat_responses()
        
        print("\n" + "=" * 70)
        if success:
            print("🎉 Chat improvement test completed!")
            print("🚀 Chat should now provide detailed building analysis!")
        else:
            print("⚠️ Test completed with warnings - check output above")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)