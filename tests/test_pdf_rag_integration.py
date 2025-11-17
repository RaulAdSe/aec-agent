#!/usr/bin/env python3
"""
Test script for PDF RAG integration with Streamlit
Tests the complete workflow: PDF upload → Gemini processing → RAG search
"""

import os
import sys
from pathlib import Path
from io import BytesIO

# Add project to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed, skipping .env loading")

from services.pdf_rag_manager import PDFRAGManager


class MockStreamlitFile:
    """Mock Streamlit uploaded file for testing."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.name = file_path.name
        self.size = file_path.stat().st_size
        self.type = "application/pdf"
        
        with open(file_path, 'rb') as f:
            self._content = f.read()
    
    def read(self):
        return self._content


def test_pdf_rag_complete_workflow():
    """Test the complete PDF RAG workflow."""
    print("🧪 Testing Complete PDF RAG Integration")
    print("=" * 60)
    
    # Test 1: Initialize RAG Manager
    print("1. Initializing PDF RAG Manager...")
    manager = PDFRAGManager()
    print("   ✅ PDFRAGManager initialized")
    
    # Test 2: Check if we have local PDF files
    print("\n2. Checking for local PDF documents...")
    doc_folder = Path("data/doc")
    pdf_files = list(doc_folder.glob("*.pdf")) if doc_folder.exists() else []
    
    if not pdf_files:
        print("   ❌ No PDF files found in data/doc/")
        print("   📝 To test fully, add some PDF files to data/doc/ directory")
        print("   Available files in data/doc/:")
        if doc_folder.exists():
            for file in doc_folder.iterdir():
                print(f"     • {file.name}")
        return False
    
    print(f"   ✅ Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files[:3]:  # Show first 3
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"     📄 {pdf.name} ({size_mb:.1f} MB)")
    
    # Test 3: Check knowledge base status
    print("\n3. Checking knowledge base status...")
    status = manager.get_knowledge_base_summary()
    print(f"   Status: {status['status']}")
    print(f"   Documents: {status['document_count']}")
    print(f"   Message: {status['message']}")
    
    # Test 4: Simulate Streamlit file upload for first PDF
    test_pdf = pdf_files[0]
    print(f"\n4. Testing PDF upload simulation with {test_pdf.name}...")
    
    try:
        # Create mock Streamlit file
        mock_file = MockStreamlitFile(test_pdf)
        file_content = mock_file.read()
        
        print(f"   📁 Mock file created: {mock_file.name} ({mock_file.size} bytes)")
        
        # Test upload
        upload_result = manager.upload_pdf_from_streamlit(mock_file, file_content)
        
        if upload_result["status"] == "success":
            print(f"   ✅ Upload successful: {upload_result['message']}")
            data = upload_result["data"]
            print(f"     File: {data['file_name']}")
            print(f"     Size: {data['file_size']} bytes") 
            print(f"     Type: {data['document_type']}")
            
        elif upload_result["status"] == "already_exists":
            print(f"   ℹ️ Already uploaded: {upload_result['message']}")
            
        else:
            print(f"   ❌ Upload failed: {upload_result['message']}")
            return False
            
    except Exception as e:
        print(f"   ❌ Upload test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Check updated knowledge base status
    print("\n5. Checking updated knowledge base status...")
    status = manager.get_knowledge_base_summary()
    print(f"   Status: {status['status']}")
    print(f"   Documents: {status['document_count']}")
    
    if status['document_count'] > 0:
        print("   📄 Available documents:")
        for doc in status.get('documents', [])[:3]:  # Show first 3
            print(f"     • {doc.get('display_name', 'Unknown')}")
    
    # Test 6: Test RAG search functionality
    print("\n6. Testing RAG search functionality...")
    
    if status['document_count'] > 0:
        test_queries = [
            "fire safety requirements",
            "door width regulations", 
            "accessibility standards",
            "emergency exit requirements"
        ]
        
        for query in test_queries:
            print(f"\n   🔍 Query: '{query}'")
            
            try:
                search_result = manager.search_legal_documents(query, max_results=2)
                
                if search_result["status"] == "success":
                    answer = search_result.get("answer", "No answer")
                    print(f"     ✅ Answer found ({len(answer)} chars)")
                    print(f"     Preview: {answer[:100]}...")
                    
                    citations = search_result.get("formatted_citations", [])
                    if citations:
                        print(f"     📚 Citations: {len(citations)} sources")
                        for i, citation in enumerate(citations[:2], 1):  # Show first 2
                            source = citation.get('display_name', 'Unknown')
                            print(f"       [{i}] {source}")
                    else:
                        raw_citations = search_result.get("citations", [])
                        if raw_citations:
                            print(f"     📚 Raw citations: {len(raw_citations)} found")
                        else:
                            print("     📚 No citations found")
                    
                elif search_result["status"] == "no_documents":
                    print(f"     ⚠️ No documents: {search_result.get('answer', 'Empty knowledge base')}")
                    
                else:
                    print(f"     ❌ Search failed: {search_result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                print(f"     ❌ Search error: {e}")
            
    else:
        print("   ⚠️ Skipping search tests - no documents in knowledge base")
    
    # Test 7: Test utility functions
    print("\n7. Testing utility functions...")
    
    try:
        from services.pdf_rag_manager import process_uploaded_pdf, query_legal_knowledge_base, get_rag_status
        
        # Test get_rag_status
        status = get_rag_status()
        print(f"   ✅ get_rag_status(): {status['status']} with {status['document_count']} docs")
        
        # Test query function if we have docs
        if status['document_count'] > 0:
            result = query_legal_knowledge_base("building regulations")
            print(f"   ✅ query_legal_knowledge_base(): {result['status']}")
        
    except Exception as e:
        print(f"   ❌ Utility functions failed: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("🎉 PDF RAG Integration Test Complete!")
    
    final_status = manager.get_knowledge_base_summary()
    if final_status['document_count'] > 0:
        print(f"✅ System ready with {final_status['document_count']} documents")
        print("🚀 Ready for Streamlit app testing!")
    else:
        print("⚠️ No documents uploaded - add PDFs to data/doc/ and run sync")
    
    return final_status['document_count'] > 0


def test_streamlit_chat_simulation():
    """Simulate Streamlit chat functionality."""
    print("\n" + "=" * 60)
    print("💬 Simulating Streamlit Chat Interface")
    print("=" * 60)
    
    manager = PDFRAGManager()
    
    # Sample chat prompts to test
    chat_prompts = [
        "What are the fire safety requirements?",
        "Tell me about accessibility standards",
        "What are the door width regulations?",
        "Emergency exit requirements",
        "Building code compliance"
    ]
    
    print("Testing chat responses with legal document search...")
    
    for prompt in chat_prompts:
        print(f"\n👤 User: {prompt}")
        
        try:
            # This simulates the generate_response function logic for legal queries
            search_result = manager.search_legal_documents(prompt, max_results=3)
            
            if search_result["status"] == "success":
                response = search_result.get("answer", "No answer found.")
                
                # Add citations like the Streamlit app does
                citations = search_result.get("formatted_citations", [])
                if citations:
                    response += "\n\n**Sources:**\n"
                    for i, citation in enumerate(citations[:3], 1):
                        source = citation.get('display_name', 'Unknown')
                        response += f"{i}. {source}\n"
                
                print(f"🤖 Assistant: {response[:200]}{'...' if len(response) > 200 else ''}")
                
            elif search_result["status"] == "no_documents":
                print(f"🤖 Assistant: {search_result.get('answer', 'No documents available')}")
                
            else:
                print(f"🤖 Assistant: Sorry, I couldn't search the legal documents right now.")
                
        except Exception as e:
            print(f"🤖 Assistant: Error occurred: {e}")
    
    print("\n✅ Chat simulation complete!")


if __name__ == "__main__":
    print("🏗️ AEC Compliance Agent - PDF RAG Integration Test")
    print("=" * 70)
    
    # Load environment
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment")
        print("💡 Make sure to set your Gemini API key in .env file")
        sys.exit(1)
    else:
        print("✅ GEMINI_API_KEY found in environment")
    
    try:
        # Run main workflow test
        success = test_pdf_rag_complete_workflow()
        
        # Run chat simulation if we have documents
        if success:
            test_streamlit_chat_simulation()
        
        print("\n" + "=" * 70)
        if success:
            print("🎉 All tests completed successfully!")
            print("🚀 PDF RAG integration is ready for use!")
        else:
            print("⚠️ Tests completed with warnings - check output above")
            print("💡 Add PDF documents to data/doc/ for full testing")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)