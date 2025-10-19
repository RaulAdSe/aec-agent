#!/usr/bin/env python3
"""
Simple RAG system test script.

This script tests the basic functionality without complex LangChain dependencies.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test that environment variables are set correctly."""
    print("🔍 Testing environment setup...")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found!")
        return False
    
    if openai_key == "your-openai-api-key-here":
        print("❌ Please set your actual OpenAI API key!")
        return False
    
    print(f"✅ OpenAI API Key: {openai_key[:20]}...")
    
    # Check other important variables
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    vectorstore_dir = os.getenv("VECTORSTORE_DIR", "vectorstore/normativa_db")
    
    print(f"✅ OpenAI Model: {model}")
    print(f"✅ Vectorstore Directory: {vectorstore_dir}")
    
    return True

def test_pdf_availability():
    """Test that PDF files are available."""
    print("\n📄 Testing PDF availability...")
    
    normativa_dir = Path("data/normativa")
    if not normativa_dir.exists():
        print(f"❌ Directory {normativa_dir} does not exist!")
        return False
    
    pdf_files = list(normativa_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {normativa_dir}")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s):")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    return True

def test_dependencies():
    """Test that required dependencies are installed."""
    print("\n📦 Testing dependencies...")
    
    required_modules = [
        "openai",
        "chromadb", 
        "sentence_transformers",
        "pypdf",
        "dotenv"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - not installed")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Install with: pip3 install " + " ".join(missing_modules))
        return False
    
    return True

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🤖 Testing OpenAI connection...")
    
    try:
        import openai
        
        # Set API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Test with a simple completion
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "user", "content": "Say 'Hello, RAG system is working!' in Spanish."}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        print(f"✅ OpenAI connection successful!")
        print(f"📝 Response: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 AEC Compliance Agent - Simple RAG Test")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment),
        ("PDF Availability", test_pdf_availability), 
        ("Dependencies", test_dependencies),
        ("OpenAI Connection", test_openai_connection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG system is ready to use.")
        print("\n📋 Next steps:")
        print("1. Run the full RAG test: python3 scripts/test_rag_system.py")
        print("2. Start querying your building codes!")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
