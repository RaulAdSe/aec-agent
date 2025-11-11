#!/usr/bin/env python3
"""
Example script demonstrating document retrieval functionality.

This shows how to use the Gemini File Search retrieval tool for 
compliance document search in your AEC compliance agent.

Make sure to set your GEMINI_API_KEY environment variable before running.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from aec_agent.tools.document_retrieval_toolkit import (
    initialize_gemini_client,
    create_document_store,
    upload_documents,
    search_documents,
    get_store_info,
    list_available_stores
)

def main():
    """Demonstrate document retrieval functionality."""
    print("🏗️  AEC Compliance Agent - Document Retrieval Demo")
    print("=" * 50)
    
    # Check if API key is available
    if not os.environ.get("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    # 1. Initialize Gemini client
    print("1. Initializing Gemini client...")
    result = initialize_gemini_client()
    if result["status"] == "success":
        print("✅ Gemini client initialized successfully")
    else:
        print(f"❌ Error: {result['logs'][0]}")
        return
    
    # 2. Create document stores for different compliance domains
    print("\n2. Creating document stores...")
    
    stores = [
        ("fire_safety", "Fire safety codes and regulations"),
        ("accessibility", "Accessibility compliance standards"), 
        ("building_codes", "General building codes and regulations")
    ]
    
    for store_name, description in stores:
        result = create_document_store(store_name, description)
        if result["status"] == "success":
            print(f"✅ Created store: {store_name}")
        else:
            print(f"❌ Failed to create store {store_name}: {result['logs'][0]}")
    
    # 3. List available stores
    print("\n3. Available document stores:")
    result = list_available_stores()
    if result["status"] == "success":
        for store in result["data"]["stores"]:
            print(f"   📁 {store['name']}: {store['description']} ({store['document_count']} docs)")
    
    # 4. Demo document upload (if you have sample docs)
    print("\n4. Document upload demo:")
    print("   ℹ️  To upload documents, add PDF/TXT files and use:")
    print("   upload_documents('fire_safety', ['path/to/fire_code.pdf'])")
    
    # Example of how to upload (commented out as we don't have sample docs)
    # sample_docs = [
    #     "docs/fire_safety_code.pdf",
    #     "docs/accessibility_guidelines.pdf"
    # ]
    # if all(os.path.exists(doc) for doc in sample_docs):
    #     result = upload_documents("fire_safety", [sample_docs[0]], ["fire_code"])
    #     if result["status"] == "success":
    #         print(f"✅ Uploaded {result['data']['successful_uploads']} documents")
    
    # 5. Demo search functionality
    print("\n5. Search functionality demo:")
    print("   ℹ️  Example searches you can perform after uploading documents:")
    
    example_queries = [
        ("fire_safety", "minimum door width for emergency exits"),
        ("accessibility", "wheelchair ramp slope requirements"), 
        ("building_codes", "minimum ceiling height for residential spaces"),
        ("fire_safety", "sprinkler system requirements for office buildings"),
        ("accessibility", "accessible parking space dimensions")
    ]
    
    for store_name, query in example_queries:
        print(f"   🔍 search_documents('{store_name}', '{query}')")
    
    print("\n6. Complete workflow example:")
    print("""
# Initialize client
initialize_gemini_client()

# Create store  
create_document_store("fire_safety", "Fire safety regulations")

# Upload documents
upload_documents("fire_safety", ["fire_code_2023.pdf", "sprinkler_regs.pdf"])

# Search for information
result = search_documents("fire_safety", "emergency exit door width requirements")
if result["status"] == "success":
    print("Search Results:")
    print(result["data"]["content"])
    
    # Use citations if available
    for citation in result["data"]["citations"]:
        print(f"Source: {citation.get('source', 'Unknown')}")
""")
    
    print("\n🎯 Integration with your agent:")
    print("""
Your compliance agent can now:
✅ Search building codes and regulations
✅ Get relevant compliance information for specific elements
✅ Provide citations and sources for requirements
✅ Handle multiple compliance domains (fire, accessibility, etc.)

Example agent workflow:
1. Agent loads building data with load_building_data()
2. Agent finds doors with get_all_elements("doors") 
3. For each door, agent searches relevant compliance docs:
   search_documents("fire_safety", "emergency exit door requirements")
4. Agent validates compliance with validate_rule()
""")

if __name__ == "__main__":
    main()