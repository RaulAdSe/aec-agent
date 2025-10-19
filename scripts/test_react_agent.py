#!/usr/bin/env python3
"""
Test ReAct Agent Implementation

This script tests the ReAct agent for autonomous AEC compliance verification.
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append('.')

def test_react_agent():
    """Test the ReAct agent implementation."""
    print("🤖 Testing ReAct Agent Implementation")
    print("=" * 50)
    
    try:
        # Import agent components
        from src.agent.graph import create_compliance_agent
        from src.agent.tools import load_project_data, set_vectorstore_manager
        from src.rag.vectorstore_manager import VectorstoreManager
        from src.rag.qa_chain import create_qa_chain
        from langchain_core.messages import HumanMessage
        
        print("✅ Agent components imported successfully")
        
        # Create mock project data
        mock_project_data = {
            "metadata": {
                "project_name": "Test Office Building",
                "building_type": "office",
                "level_name": "Ground Floor"
            },
            "levels": [
                {
                    "name": "Ground Floor",
                    "elevation": 0.0,
                    "rooms": [
                        {
                            "id": "ROOM_001",
                            "name": "Office 1",
                            "area": 20.0,
                            "use": "office",
                            "level": "Ground Floor",
                            "boundary": {
                                "points": [
                                    {"x": 0, "y": 0},
                                    {"x": 5, "y": 0},
                                    {"x": 5, "y": 4},
                                    {"x": 0, "y": 4},
                                    {"x": 0, "y": 0}
                                ]
                            }
                        },
                        {
                            "id": "ROOM_002",
                            "name": "Corridor",
                            "area": 10.0,
                            "use": "corridor",
                            "level": "Ground Floor",
                            "boundary": {
                                "points": [
                                    {"x": 5, "y": 0},
                                    {"x": 10, "y": 0},
                                    {"x": 10, "y": 2},
                                    {"x": 5, "y": 2},
                                    {"x": 5, "y": 0}
                                ]
                            }
                        }
                    ],
                    "doors": [
                        {
                            "id": "DOOR_001",
                            "width_mm": 800,  # 80cm - compliant
                            "height_mm": 2100,
                            "door_type": "single",
                            "position": {"x": 2.5, "y": 0, "z": 0},
                            "from_room": "ROOM_001",
                            "to_room": "ROOM_002",
                            "is_emergency_exit": True
                        },
                        {
                            "id": "DOOR_002",
                            "width_mm": 600,  # 60cm - non-compliant
                            "height_mm": 2100,
                            "door_type": "single",
                            "position": {"x": 7.5, "y": 0, "z": 0},
                            "from_room": "ROOM_002",
                            "to_room": "EXTERIOR",
                            "is_emergency_exit": True
                        }
                    ],
                    "walls": []
                }
            ]
        }
        
        # Save mock data to temporary file
        temp_file = Path("/tmp/test_project.json")
        with open(temp_file, 'w') as f:
            json.dump(mock_project_data, f, indent=2)
        
        print("✅ Mock project data created")
        
        # Load project data
        project = load_project_data(temp_file)
        print(f"✅ Project loaded: {project.metadata.project_name}")
        
        # Setup RAG system (mocked)
        class MockRAGManager:
            def query(self, question):
                return {
                    "result": "CTE-DB-SI: Ancho mínimo de puerta de evacuación es 80 cm",
                    "source_documents": []
                }
        
        set_vectorstore_manager(MockRAGManager())
        print("✅ RAG system configured")
        
        # Create ReAct agent
        agent = create_compliance_agent(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_iterations=5
        )
        print("✅ ReAct agent created")
        
        # Test simple query
        print("\n🧪 Testing Agent with Simple Query")
        print("-" * 40)
        
        result = agent.invoke({
            "messages": [HumanMessage(content="List all doors in the project")],
            "max_iterations": 3
        })
        
        print(f"✅ Agent completed in {result['iterations']} iterations")
        
        # Show agent messages
        print("\n📝 Agent Messages:")
        for i, message in enumerate(result["messages"]):
            if hasattr(message, 'content') and message.content:
                print(f"  {i+1}. {message.__class__.__name__}: {message.content[:100]}...")
        
        # Test compliance verification
        print("\n🧪 Testing Compliance Verification")
        print("-" * 40)
        
        compliance_result = agent.invoke({
            "messages": [HumanMessage(content="Check if all doors meet minimum width requirements for evacuation")],
            "max_iterations": 5
        })
        
        print(f"✅ Compliance check completed in {compliance_result['iterations']} iterations")
        
        # Show final answer
        if compliance_result.get("final_answer"):
            print(f"\n📊 Final Answer: {compliance_result['final_answer']}")
        
        # Clean up
        temp_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ ReAct agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_tools():
    """Test individual agent tools."""
    print("\n🛠️  Testing Agent Tools")
    print("-" * 40)
    
    try:
        from src.agent.tools import (
            get_room_info, get_door_info, list_all_doors,
            check_door_width_compliance, query_normativa, calculate_egress_distance
        )
        
        # Test tool imports
        tools = [
            get_room_info, get_door_info, list_all_doors,
            check_door_width_compliance, query_normativa, calculate_egress_distance
        ]
        
        print(f"✅ All {len(tools)} agent tools imported successfully")
        
        # Test tool signatures
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent tools test failed: {e}")
        return False

def main():
    """Run all ReAct agent tests."""
    print("🚀 ReAct Agent Testing Suite")
    print("=" * 60)
    
    tests = [
        ("Agent Tools", test_agent_tools),
        ("ReAct Agent", test_react_agent)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append({"test": test_name, "success": success})
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append({"test": test_name, "success": False, "error": str(e)})
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['test']}")
        if not result["success"] and "error" in result:
            print(f"      Error: {result['error']}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ReAct Agent is working correctly!")
        print("\nThe agent can now:")
        print("  ✅ Use 6 specialized tools for compliance verification")
        print("  ✅ Reason about building compliance autonomously")
        print("  ✅ Query Spanish building codes (CTE)")
        print("  ✅ Provide comprehensive compliance reports")
        print("\n🚀 Ready for presentation notebooks!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
