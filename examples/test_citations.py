#!/usr/bin/env python3
"""
Test script demonstrating enhanced citation capabilities.

This script shows how to use the improved citation features
for compliance document retrieval and source tracking.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aec_agent.tools import (
    search_compliance_docs,
    format_citation_text,
    format_citation_markdown,
    extract_document_sources,
    validate_citation_quality
)


def test_citation_features():
    """Test the enhanced citation features."""
    
    print("🔍 Testing Enhanced Citation Features")
    print("=" * 50)
    
    # Test queries with different types of compliance questions
    test_queries = [
        "What is the minimum door width for accessibility?",
        "Fire safety requirements for emergency exits",
        "Maximum slope for wheelchair ramps",
        "Building code requirements for ceiling height"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing query: '{query}'")
        print("-" * 60)
        
        # Perform the search
        result = search_compliance_docs(query)
        
        if result["status"] == "success":
            print("✅ Search successful!")
            print(f"📊 Citations found: {result.get('citation_count', 0)}")
            print(f"📚 Documents searched: {result.get('documents_searched', 0)}")
            
            # Show basic answer
            print(f"\n💡 Answer preview:")
            answer = result["answer"]
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"   {preview}")
            
            # Show formatted citations
            if result.get("formatted_citations"):
                print(f"\n📚 Formatted citations:")
                for citation in result["formatted_citations"]:
                    print(f"   {citation['formatted_reference']}")
                    if citation.get("cited_text"):
                        cited_preview = citation["cited_text"][:100] + "..." if len(citation["cited_text"]) > 100 else citation["cited_text"]
                        print(f"      Quote: \"{cited_preview}\"")
            
            # Show document sources
            sources = extract_document_sources(result)
            if sources:
                print(f"\n📄 Source documents:")
                for source in sources:
                    print(f"   - {source}")
            
            # Validate citation quality
            quality = validate_citation_quality(result)
            print(f"\n🎯 Citation quality score: {quality['quality_score']:.2f}")
            print(f"   Issues: {', '.join(quality['issues'])}")
            
        elif result["status"] == "no_documents":
            print("⚠️  No documents in knowledge base")
            print(f"   Suggestion: {result['answer']}")
        else:
            print(f"❌ Search failed: {result.get('answer', 'Unknown error')}")
        
        print()
        

def demo_citation_formatting():
    """Demonstrate different citation formatting options."""
    
    print("\n🎨 Citation Formatting Demo")
    print("=" * 50)
    
    # Search for a common compliance question
    result = search_compliance_docs("minimum door width for emergency exits")
    
    if result["status"] == "success":
        print("\n📝 Text Format:")
        print("-" * 30)
        text_format = format_citation_text(result)
        print(text_format)
        
        print("\n📋 Markdown Format:")
        print("-" * 30)
        markdown_format = format_citation_markdown(result)
        print(markdown_format)
        
    else:
        print("❌ Could not demonstrate formatting - search failed")
        print(f"   Reason: {result.get('answer', 'Unknown error')}")


def check_knowledge_base_status():
    """Check if knowledge base is ready for testing."""
    
    print("🏥 Knowledge Base Status")
    print("=" * 50)
    
    from aec_agent.tools import check_knowledge_base_status
    
    status = check_knowledge_base_status()
    
    print(f"Status: {status['status']}")
    print(f"Document count: {status.get('document_count', 0)}")
    print(f"Message: {status['message']}")
    
    if status["status"] != "ready":
        print(f"\n⚠️  Knowledge base not ready for citation testing.")
        print(f"   Please run: ./kb sync")
        return False
    
    return True


def main():
    """Main test function."""
    
    print("🧪 AEC Compliance Agent - Citation Testing")
    print("=" * 60)
    
    # Check if knowledge base is ready
    if not check_knowledge_base_status():
        print("\n❌ Exiting - knowledge base not ready")
        sys.exit(1)
    
    # Run citation tests
    test_citation_features()
    
    # Demo formatting
    demo_citation_formatting()
    
    print("\n✅ Citation testing complete!")
    print("\n🎯 Key Features Demonstrated:")
    print("   - Enhanced citation extraction")
    print("   - Source document tracking") 
    print("   - Multiple output formats")
    print("   - Citation quality validation")
    print("   - Readable source references")


if __name__ == "__main__":
    main()