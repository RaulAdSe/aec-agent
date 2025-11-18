#!/usr/bin/env python3
"""
Programmatic test for Streamlit IFC upload and agent questioning flow.
This simulates the user experience without requiring the actual Streamlit UI.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aec_agent.utils.ifc_to_json import IFCToJSONConverter
from aec_agent.agent import create_agent
from services.session_manager import SessionManager
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging to see agent reasoning
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

def test_ifc_upload_flow():
    """Test the complete IFC upload and questioning flow."""
    
    print("🏗️ AEC COMPLIANCE AGENT - STREAMLIT IFC FLOW TEST")
    print("=" * 60)
    
    # 1. Simulate finding available IFC files
    ifc_dir = Path("data/in")
    available_files = list(ifc_dir.glob("*.ifc")) if ifc_dir.exists() else []
    
    if not available_files:
        print("❌ No IFC files found in data/in/ directory")
        print("💡 Please add some IFC files to test with")
        return
    
    print(f"📁 Found {len(available_files)} IFC files:")
    for file in available_files:
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    # Use the first available file
    test_file = available_files[0]
    print(f"\n🔄 Testing with: {test_file.name}")
    
    # 2. Simulate IFC processing (like Streamlit upload)
    print("\n📊 STEP 1: Processing IFC file...")
    try:
        converter = IFCToJSONConverter(str(test_file))
        json_data = converter.extract_to_dict()
        
        # Save processed JSON (like Streamlit does)
        processed_dir = Path("data/processed_ifc")
        processed_dir.mkdir(parents=True, exist_ok=True)
        json_file_path = processed_dir / f"{test_file.name}.json"
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Processed {test_file.name} → {json_file_path}")
        print(f"📈 Building data summary:")
        print(f"  - Project: {json_data['file_info']['project_name']}")
        print(f"  - Total elements: {json_data['file_info']['total_elements']}")
        print(f"  - Spaces: {len(json_data.get('spaces', []))}")
        print(f"  - Walls: {len(json_data.get('walls', []))}")
        print(f"  - Doors: {len(json_data.get('doors', []))}")
        print(f"  - Slabs: {len(json_data.get('slabs', []))}")
        
    except Exception as e:
        print(f"❌ Error processing IFC file: {e}")
        return
    
    # 3. Initialize agent (like Streamlit does)
    print(f"\n🤖 STEP 2: Initializing reasoning agent...")
    try:
        # Create a test session
        session_manager = SessionManager()
        session_id = session_manager.create_new_session()
        
        # Initialize agent with recovery system
        agent = create_agent(
            model_name="gpt-4o-mini",
            temperature=0.1,
            verbose=True,
            enable_memory=True,
            session_id=session_id,
            max_iterations=20
        )
        
        # Track the processed file in agent memory
        if hasattr(agent, 'memory_manager') and agent.memory_manager:
            agent.memory_manager.track_active_file(str(json_file_path))
        
        print("✅ Agent initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        return
    
    # 4. Test series of questions
    test_questions = [
        "Load the building data",
        "How many doors are in the building?",
        "What is the distance between the first two doors?",
        "Show me all the spaces in the building",
        "Calculate the area of the largest space"
    ]
    
    print(f"\n💬 STEP 3: Testing agent responses...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Create enhanced prompt with building context (like Streamlit does)
            enhanced_prompt = f"{question}\n\nNote: Available building models: {test_file.name}"
            
            # Process the question
            result = agent.process_goal(enhanced_prompt)
            
            # Extract response
            if isinstance(result, dict) and 'reasoning_result' in result:
                reasoning = result['reasoning_result']
                response = reasoning.get('message', 'No response generated')
                
                print(f"🤖 Agent Response: {response}")
                
                # Show execution summary
                if 'summary' in reasoning:
                    summary = reasoning['summary']
                    completed = summary.get('completed_tasks', 0)
                    total = summary.get('total_tasks', 0)
                    progress = summary.get('progress_percentage', 0)
                    
                    print(f"📊 Execution: {completed}/{total} tasks ({progress:.1f}%)")
                    
                    if reasoning.get('status') == 'partial':
                        print("⚠️  Partial completion detected - recovery system may have been used")
                
            else:
                print(f"🤖 Agent Response: {str(result)}")
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            print("🔄 Recovery system should handle this...")
    
    print(f"\n🎉 STREAMLIT IFC FLOW TEST COMPLETE!")
    print("=" * 60)
    
    # 5. Test recovery scenarios
    print(f"\n🛡️ STEP 4: Testing recovery scenarios...")
    
    recovery_questions = [
        "Calculate the distance between two doors that don't exist",  # Should gracefully degrade
        "Use a tool that doesn't exist",  # Should recover with alternative
        "Analyze fire safety compliance"  # Complex multi-step task
    ]
    
    for question in recovery_questions:
        print(f"\n🧪 Recovery Test: {question}")
        try:
            enhanced_prompt = f"{question}\n\nNote: Available building models: {test_file.name}"
            result = agent.process_goal(enhanced_prompt)
            
            if isinstance(result, dict) and 'reasoning_result' in result:
                reasoning = result['reasoning_result']
                response = reasoning.get('message', 'No response generated')
                status = reasoning.get('status', 'unknown')
                
                print(f"🤖 Response: {response}")
                print(f"📊 Status: {status}")
                
                if status == 'partial':
                    print("✅ Recovery system successfully handled the error!")
            else:
                print(f"🤖 Response: {str(result)}")
                
        except Exception as e:
            print(f"❌ Recovery test failed: {e}")

if __name__ == "__main__":
    test_ifc_upload_flow()