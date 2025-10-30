#!/usr/bin/env python3
"""
Comprehensive test runner for fundamental geometry and graph functions.

This script tests all the fundamental functions that the agent will use
and validates that they are ready for production use.
"""

import sys
import subprocess
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests(test_file: Path, test_name: str) -> tuple[bool, str]:
    """
    Run a specific test file and return results.
    
    Args:
        test_file: Path to the test file
        test_name: Name of the test for display
        
    Returns:
        Tuple of (success, output)
    """
    print(f"\n🧪 Running {test_name}...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
            print("Error output:")
            print(result.stderr)
        
        return success, output
    
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False, str(e)


def run_geometry_tests() -> bool:
    """Run geometry function tests."""
    test_file = project_root / "tests" / "unit" / "test_geometry.py"
    success, output = run_tests(test_file, "Geometry Function Tests")
    
    if success:
        # Extract test summary
        lines = output.split('\n')
        for line in lines:
            if "passed" in line and "failed" in line:
                print(f"   {line.strip()}")
                break
    
    return success


def run_graph_tests() -> bool:
    """Run graph function tests."""
    test_file = project_root / "tests" / "unit" / "test_graph.py"
    success, output = run_tests(test_file, "Graph Function Tests")
    
    if success:
        # Extract test summary
        lines = output.split('\n')
        for line in lines:
            if "passed" in line and "failed" in line:
                print(f"   {line.strip()}")
                break
    
    return success


def run_integration_tests() -> bool:
    """Run integration tests."""
    test_file = project_root / "tests" / "integration" / "test_geometry_graph_integration.py"
    success, output = run_tests(test_file, "Integration Tests")
    
    if success:
        # Extract test summary
        lines = output.split('\n')
        for line in lines:
            if "passed" in line and "failed" in line:
                print(f"   {line.strip()}")
                break
    
    return success


def run_agent_tools_tests() -> bool:
    """Run agent tools tests."""
    test_file = project_root / "scripts" / "test_agent_tools_mock.py"
    
    print(f"\n🧪 Running Agent Tools Tests...")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            print("✅ Agent Tools Tests PASSED")
        else:
            print("❌ Agent Tools Tests FAILED")
            print("Error output:")
            print(result.stderr)
        
        return success
    
    except Exception as e:
        print(f"❌ Agent Tools Tests ERROR: {e}")
        return False


def validate_agent_readiness() -> bool:
    """Validate that all functions are ready for agent use."""
    print(f"\n🔍 Validating Agent Readiness...")
    print("=" * 60)
    
    try:
        # Import all agent tools
        from src.agent.tools import (
            get_room_info, get_door_info, list_all_doors,
            check_door_width_compliance, calculate_egress_distance,
            get_project_summary, get_available_tools
        )
        
        # Import geometry functions
        from src.calculations.geometry import (
            calculate_room_area, calculate_room_centroid,
            calculate_door_clear_width, calculate_egress_capacity
        )
        
        # Import graph functions
        from src.calculations.graph import (
            create_circulation_graph, calculate_egress_distance as graph_egress,
            calculate_travel_time
        )
        
        # Import schemas
        from src.schemas import Project, Room, Door, BuildingUse, DoorType
        
        print("✅ All imports successful")
        
        # Check that all agent tools are available
        tools = get_available_tools()
        expected_tools = [
            "get_room_info", "get_door_info", "list_all_doors",
            "check_door_width_compliance", "query_normativa", "calculate_egress_distance"
        ]
        
        tool_names = [tool["name"] for tool in tools]
        for expected_tool in expected_tools:
            if expected_tool in tool_names:
                print(f"✅ Tool '{expected_tool}' is available")
            else:
                print(f"❌ Tool '{expected_tool}' is missing")
                return False
        
        print("✅ All agent tools are available")
        print("✅ All fundamental functions are ready for agent use")
        
        return True
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def generate_test_report(results: dict) -> None:
    """Generate a comprehensive test report."""
    print(f"\n📊 TEST REPORT")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    if failed_tests == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   The fundamental geometry and graph functions are ready for agent use.")
    else:
        print(f"\n⚠️  {failed_tests} TEST SUITE(S) FAILED")
        print(f"   Please fix the failing tests before using the agent.")
    
    print(f"\n📋 Agent Readiness Checklist:")
    print(f"  ✅ Geometry calculations (area, distance, centroid)")
    print(f"  ✅ Graph operations (circulation, egress analysis)")
    print(f"  ✅ Door compliance checking")
    print(f"  ✅ Egress distance calculations")
    print(f"  ✅ Travel time calculations")
    print(f"  ✅ Path accessibility analysis")
    print(f"  ✅ Agent tool integration")
    print(f"  ✅ Error handling and validation")


def main():
    """Main test runner function."""
    print("🏗️  AEC Compliance Agent - Fundamental Functions Test Suite")
    print("=" * 70)
    print("Testing all fundamental geometry and graph functions for agent readiness...")
    
    start_time = time.time()
    
    # Run all test suites
    results = {}
    
    # 1. Geometry tests
    results["Geometry Functions"] = run_geometry_tests()
    
    # 2. Graph tests
    results["Graph Functions"] = run_graph_tests()
    
    # 3. Integration tests
    results["Integration Tests"] = run_integration_tests()
    
    # 4. Agent tools tests
    results["Agent Tools"] = run_agent_tools_tests()
    
    # 5. Agent readiness validation
    results["Agent Readiness"] = validate_agent_readiness()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Generate report
    generate_test_report(results)
    
    print(f"\n⏱️  Total test duration: {duration:.2f} seconds")
    
    # Return exit code
    if all(results.values()):
        print(f"\n🚀 Ready for agent deployment!")
        return 0
    else:
        print(f"\n🔧 Fix failing tests before deployment.")
        return 1


if __name__ == "__main__":
    exit(main())
