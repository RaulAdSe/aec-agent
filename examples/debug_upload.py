#!/usr/bin/env python3
"""Debug script to test Gemini upload directly"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aec_agent.tools.document_retrieval_toolkit import *
import time

# Initialize
print("1. Initializing Gemini client...")
result = initialize_gemini_client()
print(f"   Status: {result['status']}")

# Create fresh store
print("\n2. Creating document store...")
result = create_document_store("test_store", "Test store")
print(f"   Status: {result['status']}")

# Upload document
print("\n3. Uploading document...")
project_root = os.path.join(os.path.dirname(__file__), '..')
doc_path = os.path.join(project_root, "data", "doc", "sample_building_code.txt")
result = upload_documents("test_store", [doc_path])
print(f"   Status: {result['status']}")
print(f"   Details: {result}")

# Wait for processing
print("\n4. Waiting 10 seconds for processing...")
time.sleep(10)

# Check store info
print("\n5. Checking store info...")
result = get_store_info("test_store")
print(f"   Status: {result['status']}")
if result["status"] == "success":
    print(f"   Document count: {result['data'].get('document_count', 'unknown')}")

# Try search
print("\n6. Testing search...")
result = search_documents("test_store", "minimum door width")
print(f"   Status: {result['status']}")
if result["status"] == "success":
    print(f"   Answer: {result['data']['content'][:200]}...")
else:
    print(f"   Error: {result.get('logs', 'Unknown error')}")

# Cleanup
print("\n7. Cleaning up test store...")
delete_result = delete_document_store("test_store")
print(f"   Cleanup: {delete_result['status']}")