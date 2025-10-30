#!/usr/bin/env python3
"""
Test script for the AEC compliance agent system.

This script tests the complete ReAct agent functionality without requiring
full pipeline setup, using mock data and simplified components.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agent.tools import load_project_data, set_vectorstore_manager
from agent.graph import create_compliance_agent


class MockRAGManager:
    """Mock RAG manager for testing without full vectorstore setup."""
    
    def query(self, question: str) -> Dict[str, Any]:
        """Mock query response."""
        return {
            "answer": f"Mock response for: {question}. According to CTE DB-SI, minimum door width is 800mm for standard doors and 900mm for emergency exits.",
            "sources": [{"source": "CTE_DB-SI.pdf", "page": 1}],
            "confidence": 0.8,
            "regulation_references": ["CTE DB-SI Section 3.1"]
        }


def test_agent_tools():
    """Test individual agent tools."""
    print("=" * 60)
    print("TESTING AGENT TOOLS")
    print("=" * 60)
    
    # Find a test project file
    test_files = list(Path("data/extracted").glob("*.json"))
    if not test_files:
        print("❌ No test project files found in data/extracted/")
        return False
    
    test_file = test_files[0]
    print(f"Using test file: {test_file}")
    
    try:
        # Load project data
        print("\n1. Loading project data...")
        project_data = load_project_data(test_file)
        print(f"   ✅ Loaded project: {project_data.metadata.project_name}")
        print(f"   - Rooms: {len(project_data.rooms)}")
        print(f"   - Doors: {len(project_data.doors)}")
        print(f"   - Walls: {len(project_data.walls)}")
        
        # Set up mock RAG manager
        print("\n2. Setting up mock RAG manager...")
        set_vectorstore_manager(MockRAGManager())
        print("   ✅ Mock RAG manager ready")
        
        # Test individual tools
        print("\n3. Testing individual tools...")
        
        # Test list_all_doors
        from agent.tools import list_all_doors
        doors = list_all_doors()
        print(f"   ✅ list_all_doors: Found {len(doors)} doors")
        
        # Test get_door_info for first door
        if doors and len(doors) > 0:
            first_door_id = doors[0]["id"]
            from agent.tools import get_door_info
            door_info = get_door_info(first_door_id)
            print(f"   ✅ get_door_info: Retrieved info for {first_door_id}")
        
        # Test check_door_width_compliance
        if doors and len(doors) > 0:
            first_door_id = doors[0]["id"]
            from agent.tools import check_door_width_compliance
            compliance = check_door_width_compliance(first_door_id)
            print(f"   ✅ check_door_width_compliance: Checked {first_door_id}")
            print(f"      Status: {compliance.get('compliance_status', 'Unknown')}")
        
        # Test query_normativa
        from agent.tools import query_normativa
        normativa_result = query_normativa("What is the minimum door width requirement?")
        print(f"   ✅ query_normativa: Got response")
        print(f"      Answer: {normativa_result.get('answer', 'No answer')[:100]}...")
        
        print("\n✅ All tool tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_reasoning():
    """Test the complete agent reasoning system."""
    print("\n" + "=" * 60)
    print("TESTING AGENT REASONING")
    print("=" * 60)
    
    try:
        # Create agent
        print("1. Creating compliance agent...")
        agent = create_compliance_agent(
            model_name="gemini-pro",
            temperature=0.1,
            max_iterations=5  # Reduced for testing
        )
        print("   ✅ Agent created")
        
        # Test simple verification query
        print("\n2. Running simple verification...")
        query = "Please list all doors in the project and check if they meet minimum width requirements."
        
        print(f"   Query: {query}")
        print("   Running agent...")
        
        results = agent.verify_compliance(query, agent_mode="verification")
        
        print(f"   ✅ Agent completed in {results['iterations']} iterations")
        print(f"   - Total checks: {results['compliance_summary']['total_checks']}")
        print(f"   - Overall status: {results['compliance_summary']['overall_status']}")
        
        # Show agent response
        if results["agent_response"]:
            print(f"\n   Agent Response:")
            print(f"   {results['agent_response'][:200]}...")
        
        print("\n✅ Agent reasoning test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Agent reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline integration."""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE")
    print("=" * 60)
    
    try:
        # Import pipeline class
        from scripts.run_full_pipeline import CompliancePipeline
        
        # Find test files
        test_files = list(Path("data/extracted").glob("*.json"))
        if not test_files:
            print("❌ No test project files found")
            return False
        
        test_file = test_files[0]
        
        # Create pipeline (without full RAG setup)
        print("1. Creating pipeline...")
        pipeline = CompliancePipeline(
            project_file=test_file,
            vectorstore_dir=Path("vectorstore/test_db"),
            normativa_dir=Path("data/normativa"),
            output_dir=Path("outputs/test_reports"),
            verbose=True
        )
        
        # Set up mock RAG manager
        set_vectorstore_manager(MockRAGManager())
        
        # Load project data manually
        pipeline.project_data = load_project_data(test_file)
        
        # Create agent
        pipeline.agent = create_compliance_agent(max_iterations=3)
        
        print("   ✅ Pipeline created")
        
        # Run verification
        print("\n2. Running verification...")
        results = pipeline.run_verification(
            query="Check door width compliance for all doors.",
            mode="door_widths"
        )
        
        print(f"   ✅ Verification completed")
        print(f"   - Iterations: {results['iterations']}")
        print(f"   - Status: {results['compliance_summary']['overall_status']}")
        
        # Generate report
        print("\n3. Generating report...")
        report_file = pipeline.generate_report(results, "door_widths")
        print(f"   ✅ Report generated: {report_file}")
        
        print("\n✅ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 AEC COMPLIANCE AGENT SYSTEM TESTS")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not Path("data/extracted").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Agent Tools", test_agent_tools),
        ("Agent Reasoning", test_agent_reasoning),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The agent system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
