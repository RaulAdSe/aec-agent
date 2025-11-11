#!/usr/bin/env python3
"""Quick sync for testing with one document first"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aec_agent.tools.document_retrieval_toolkit import *

# Initialize
print("Initializing...")
initialize_gemini_client()

# Create fresh store
print("Creating store...")
create_document_store("compliance_knowledge_base", "AEC compliance documents knowledge base")

# Upload one small document first
print("Uploading one document...")
project_root = os.path.join(os.path.dirname(__file__), '..')
doc_path = os.path.join(project_root, "data", "doc", "Parte_I_version_modificaciones.pdf")
result = upload_documents("compliance_knowledge_base", [doc_path])
print(f"Upload result: {result['status']}")
print(f"Details: {result['data']['upload_results'][0]}")

# Wait and test search
import time
print("Waiting 15 seconds for processing...")
time.sleep(15)

print("Testing search...")
search_result = search_documents("compliance_knowledge_base", "requisitos de accesibilidad")
print(f"Search status: {search_result['status']}")
if search_result["status"] == "success":
    print(f"Answer preview: {search_result['data']['content'][:200]}...")
else:
    print(f"Search error: {search_result.get('logs', 'Unknown error')}")