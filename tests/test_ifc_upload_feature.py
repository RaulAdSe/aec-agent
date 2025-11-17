#!/usr/bin/env python3
"""
Test script for IFC upload functionality
Simulates the workflow without Streamlit UI
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Import our modules
from aec_agent.utils.ifc_to_json import IFCToJSONConverter
from services.ifc_data_manager import IFCDataManager

def test_ifc_processing():
    """Test the complete IFC processing workflow."""
    print("🧪 Testing IFC Upload Functionality")
    print("=" * 50)
    
    # Check if we have any sample IFC files
    print("1. Looking for sample IFC files...")
    
    # Common places to look for IFC files
    search_paths = [
        "data/blueprints",
        "data/in", 
        "data",
        "examples",
        "."
    ]
    
    ifc_files = []
    for path in search_paths:
        if Path(path).exists():
            ifc_files.extend(list(Path(path).glob("*.ifc")))
    
    if not ifc_files:
        print("❌ No IFC files found for testing")
        print("   Expected locations:", search_paths)
        print("   You can add a sample .ifc file to any of these directories")
        return False
    
    print(f"✅ Found {len(ifc_files)} IFC file(s):")
    for f in ifc_files:
        print(f"   📄 {f}")
    
    # Test with the first IFC file found
    test_file = ifc_files[0]
    print(f"\n2. Testing processing with: {test_file.name}")
    
    try:
        # Simulate the Streamlit upload process
        converter = IFCToJSONConverter(str(test_file))
        json_data = converter.extract_to_dict()
        
        print("✅ IFC conversion successful!")
        
        # Show summary like the Streamlit app does
        summary = {
            "project": json_data["file_info"]["project_name"],
            "total_elements": json_data["file_info"]["total_elements"],
            "spaces": len(json_data.get("spaces", [])),
            "walls": len(json_data.get("walls", [])),
            "doors": len(json_data.get("doors", [])),
            "slabs": len(json_data.get("slabs", [])),
            "stairs": len(json_data.get("stairs", []))
        }
        
        print("📊 Extracted elements:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error processing IFC file: {e}")
        return False
    
    # Test storage system
    print("\n3. Testing storage system...")
    
    # Create processed files directory
    processed_dir = Path("data/processed_ifc")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON file (simulate Streamlit storage)
    json_file_path = processed_dir / f"{test_file.name}.json"
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved to: {json_file_path}")
    print(f"   File size: {json_file_path.stat().st_size} bytes")
    
    # Test agent data access
    print("\n4. Testing agent data access...")
    
    manager = IFCDataManager()
    
    # Test all access methods
    all_files = manager.get_all_processed_files()
    print(f"✅ Processed files found: {all_files}")
    
    building_summary = manager.get_building_summary()
    print("📋 Building summary:")
    for key, value in building_summary.items():
        print(f"   {key}: {value}")
    
    # Test element access
    all_spaces = manager.get_all_spaces()
    all_doors = manager.get_all_doors()
    all_walls = manager.get_all_walls()
    
    print(f"🏠 Elements accessible to agent:")
    print(f"   Spaces: {len(all_spaces)}")
    print(f"   Doors: {len(all_doors)}")  
    print(f"   Walls: {len(all_walls)}")
    
    # Show some example data
    if all_spaces:
        print(f"\n📍 Example space: {all_spaces[0].get('name', 'Unnamed')}")
    if all_doors:
        print(f"🚪 Example door: {all_doors[0].get('name', 'Unnamed')}")
    
    # Test agent-ready data format
    agent_data = manager.get_compliance_data_for_agent()
    print(f"\n🤖 Agent data structure ready:")
    print(f"   Keys: {list(agent_data.keys())}")
    print(f"   Total elements for agent: {len(agent_data.get('spaces', [])) + len(agent_data.get('doors', [])) + len(agent_data.get('walls', []))}")
    
    return True

def test_session_state_simulation():
    """Simulate how Streamlit session state would work."""
    print("\n" + "=" * 50)
    print("🔄 Testing Session State Simulation")
    print("=" * 50)
    
    # Simulate session state
    session_state = {
        "processed_ifc_files": {},
        "messages": []
    }
    
    # Simulate file upload
    manager = IFCDataManager()
    files = manager.get_all_processed_files()
    
    if not files:
        print("❌ No processed files to simulate with")
        return
    
    # Simulate processing (already done, just load)
    for filename in files:
        json_data = manager.get_building_data(filename)
        if json_data:
            session_state["processed_ifc_files"][filename] = {
                "json_data": json_data,
                "file_size": Path(f"data/processed_ifc/{filename}.json").stat().st_size,
                "processed_at": str(datetime.now())
            }
    
    print(f"✅ Simulated session state with {len(session_state['processed_ifc_files'])} files")
    
    # Test chat responses  
    def simulate_chat_response(prompt, processed_files):
        """Simulate the generate_response function."""
        if "space" in prompt.lower() or "room" in prompt.lower():
            if processed_files:
                all_spaces = []
                for data in processed_files.values():
                    all_spaces.extend(data['json_data'].get('spaces', []))
                
                if all_spaces:
                    space_names = [space.get('name', 'Unnamed') for space in all_spaces[:5]]
                    response = f"I found {len(all_spaces)} spaces in your building models. "
                    response += f"Some examples: {', '.join(space_names)}. "
                    return response
        return "Default response"
    
    # Test various prompts
    test_prompts = [
        "Tell me about the spaces",
        "How many rooms are there?", 
        "Show me the doors",
        "What buildings do you have?"
    ]
    
    print("\n💬 Testing chat responses:")
    for prompt in test_prompts:
        response = simulate_chat_response(prompt, session_state["processed_ifc_files"])
        print(f"   Q: {prompt}")
        print(f"   A: {response[:100]}...")
        print()

def show_data_storage_structure():
    """Show where and how data is stored."""
    print("\n" + "=" * 50) 
    print("💾 Data Storage Structure")
    print("=" * 50)
    
    # Show directory structure
    dirs_to_check = [
        "data/processed_ifc",
        "services", 
        "."
    ]
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            print(f"\n📁 {dir_path}/")
            for item in path.iterdir():
                if item.is_file():
                    size = item.stat().st_size
                    print(f"   📄 {item.name} ({size} bytes)")
                elif item.is_dir():
                    print(f"   📁 {item.name}/")
    
    # Show data flow
    print(f"\n🔄 Data Flow:")
    print(f"   1. IFC file uploaded → Streamlit file uploader")
    print(f"   2. File saved temporarily → /tmp/...")
    print(f"   3. IFCToJSONConverter processes → JSON data")
    print(f"   4. Data stored in session → st.session_state.processed_ifc_files")
    print(f"   5. Data saved persistently → data/processed_ifc/*.json")
    print(f"   6. Agent accesses via → IFCDataManager")

if __name__ == "__main__":
    print("🏗️ AEC Compliance Agent - IFC Upload Test")
    print("=" * 60)
    
    success = test_ifc_processing()
    if success:
        test_session_state_simulation()
    
    show_data_storage_structure()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    if success:
        print("🎉 IFC upload functionality is working correctly!")
    else:
        print("❌ Some issues found - check the output above")