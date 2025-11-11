#!/usr/bin/env python3
"""
Simple script to upload documents to Gemini File Search stores.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aec_agent.tools.document_retrieval_toolkit import (
    initialize_gemini_client,
    create_document_store, 
    upload_documents,
    get_store_info
)

def main():
    # Initialize client
    print("Initializing Gemini client...")
    result = initialize_gemini_client()
    if result["status"] != "success":
        print(f"Error: {result['logs'][0]}")
        return
    
    # Create a compliance documents store
    print("Creating document store...")
    result = create_document_store("compliance_docs", "Building compliance documents")
    print(f"Store created: {result['status']}")
    
    # Example document paths - replace with your actual document paths
    document_paths = [
        # Add your document paths here, for example:
        # "docs/fire_safety_code.pdf",
        # "docs/accessibility_guidelines.pdf", 
        # "docs/building_code_2023.pdf",
        # "data/regulations/emergency_exits.pdf"
    ]
    
    # If you have documents, uncomment and modify this:
    # if document_paths:
    #     print(f"Uploading {len(document_paths)} documents...")
    #     result = upload_documents("compliance_docs", document_paths)
    #     if result["status"] == "success":
    #         print(f"✅ Successfully uploaded {result['data']['successful_uploads']} documents")
    #         for upload_result in result["data"]["upload_results"]:
    #             print(f"   📄 {upload_result['file_path']}: {upload_result['status']}")
    #     else:
    #         print(f"❌ Upload failed: {result['logs'][0]}")
    
    print("\nTo upload your documents:")
    print("1. Put your PDF/TXT files in a folder (e.g., docs/)")
    print("2. Update the document_paths list above with your file paths")
    print("3. Uncomment the upload code")
    print("4. Run this script again")
    
    # Show store info
    print("\nCurrent store info:")
    result = get_store_info("compliance_docs")
    if result["status"] == "success":
        info = result["data"]
        print(f"Store: {info['name']}")
        print(f"Documents: {info['document_count']}")
        print(f"Description: {info['description']}")

if __name__ == "__main__":
    main()