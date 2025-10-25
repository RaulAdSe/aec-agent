#!/usr/bin/env python3
"""
Comprehensive RAG System Testing Script

This script runs all tests for the RAG system components and provides
a comprehensive test report.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def run_command(command: List[str], description: str) -> Dict[str, Any]:
    """Run a command and return results."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(command)}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "description": description
        }
        
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "duration": 0,
            "description": description
        }

def test_imports():
    """Test that all RAG components can be imported."""
    print("\n📦 Testing RAG Component Imports")
    print("=" * 50)
    
    imports_to_test = [
        ("src.rag.vectorstore_manager", "VectorstoreManager"),
        ("src.rag.document_loader", "DocumentLoader"),
        ("src.rag.embeddings_config", "get_embeddings"),
        ("src.rag.qa_chain", "create_qa_chain"),
    ]
    
    results = []
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            results.append({"module": module_name, "class": class_name, "success": True})
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            results.append({"module": module_name, "class": class_name, "success": False, "error": str(e)})
    
    return results

def test_environment():
    """Test environment setup."""
    print("\n🌍 Testing Environment Setup")
    print("=" * 50)
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file exists")
    else:
        print("⚠️  .env file not found")
    
    # Check environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    env_vars = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "VECTORSTORE_DIR",
        "EMBEDDINGS_MODEL"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if var == "OPENAI_API_KEY":
                # Don't print the actual key
                print(f"✅ {var}: {'*' * 10}...{value[-4:]}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set")

def run_unit_tests():
    """Run unit tests for RAG components."""
    print("\n🧪 Running RAG Unit Tests")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "tests/unit/rag/test_vectorstore_manager.py",
        "tests/unit/rag/test_document_loader.py", 
        "tests/unit/rag/test_embeddings_config.py",
        "tests/unit/rag/test_qa_chain.py"
    ]
    
    results = []
    
    for test_file in test_files:
        if Path(test_file).exists():
            result = run_command(
                ["python3", "-m", "pytest", test_file, "-v"],
                f"Running {test_file}"
            )
            results.append(result)
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append({
                "success": False,
                "description": f"Test file not found: {test_file}",
                "stdout": "",
                "stderr": "File not found"
            })
    
    return results

def test_rag_functionality():
    """Test basic RAG functionality."""
    print("\n🔍 Testing RAG Functionality")
    print("=" * 50)
    
    try:
        from src.rag.embeddings_config import get_embeddings
        from src.rag.vectorstore_manager import VectorstoreManager
        
        # Test embeddings
        print("Testing embeddings...")
        embeddings = get_embeddings()
        test_query = "test query"
        embedding = embeddings.embed_query(test_query)
        print(f"✅ Embeddings working: {len(embedding)} dimensions")
        
        # Test vectorstore manager
        print("Testing vectorstore manager...")
        temp_dir = Path("/tmp/test_vectorstore")
        manager = VectorstoreManager(temp_dir)
        print("✅ VectorstoreManager created")
        
        # Test document loader
        from src.rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        print("✅ DocumentLoader created")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG functionality test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\n🔗 Testing Component Integration")
    print("=" * 50)
    
    try:
        # Test that all components work together
        from src.rag.vectorstore_manager import VectorstoreManager
        from src.rag.document_loader import DocumentLoader
        from src.rag.embeddings_config import get_embeddings
        from src.rag.qa_chain import create_qa_chain
        
        print("✅ All components imported successfully")
        
        # Test embeddings
        embeddings = get_embeddings()
        print("✅ Embeddings loaded")
        
        # Test document loader
        loader = DocumentLoader()
        print("✅ Document loader created")
        
        # Test vectorstore manager
        temp_dir = Path("/tmp/test_integration")
        manager = VectorstoreManager(temp_dir)
        print("✅ Vectorstore manager created")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def generate_report(results: List[Dict[str, Any]]):
    """Generate a comprehensive test report."""
    print("\n📊 Test Report Summary")
    print("=" * 50)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.get("success", False))
    failed_tests = total_tests - successful_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\n❌ Failed Tests:")
        for result in results:
            if not result.get("success", False):
                print(f"  - {result.get('description', 'Unknown')}")
                if result.get("stderr"):
                    print(f"    Error: {result['stderr'][:100]}...")
    
    # Save detailed report
    report_file = Path("test_report.json")
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    """Main test function."""
    print("🚀 AEC Compliance Agent - RAG System Testing")
    print("=" * 60)
    
    all_results = []
    
    # Test imports
    import_results = test_imports()
    all_results.extend([{"success": r["success"], "description": f"Import {r['module']}.{r['class']}"} for r in import_results])
    
    # Test environment
    test_environment()
    
    # Test basic functionality
    functionality_success = test_rag_functionality()
    all_results.append({"success": functionality_success, "description": "RAG Functionality Test"})
    
    # Test integration
    integration_success = test_integration()
    all_results.append({"success": integration_success, "description": "Component Integration Test"})
    
    # Run unit tests
    unit_test_results = run_unit_tests()
    all_results.extend(unit_test_results)
    
    # Generate report
    generate_report(all_results)
    
    # Final status
    total_successful = sum(1 for r in all_results if r.get("success", False))
    total_tests = len(all_results)
    
    if total_successful == total_tests:
        print("\n🎉 All tests passed! RAG system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_successful} tests failed. Please check the report.")
        return 1

if __name__ == "__main__":
    exit(main())
