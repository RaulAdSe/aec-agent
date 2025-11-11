"""
Citation utilities for compliance document retrieval.

This module provides utilities for formatting and displaying citations
from compliance documents in a readable, professional format.
"""

from typing import List, Dict, Any


def format_citation_text(result: Dict[str, Any]) -> str:
    """
    Format a search result with citations into a readable text format.
    
    Args:
        result: Search result from search_compliance_docs()
        
    Returns:
        Formatted string with answer and citations
    """
    if result["status"] != "success":
        return f"Error: {result.get('answer', 'Search failed')}"
    
    text = []
    
    # Add the main answer
    text.append("📋 COMPLIANCE INFORMATION")
    text.append("=" * 50)
    text.append(result["answer"])
    
    # Add citations if available
    citations = result.get("citations", [])
    formatted_citations = result.get("formatted_citations", [])
    
    if citations and len(citations) > 0:
        text.append("\n📚 SOURCES")
        text.append("-" * 30)
        
        for citation in formatted_citations:
            text.append(f"{citation['formatted_reference']}")
            
            if citation.get("cited_text"):
                # Show the specific text that was cited
                cited_text = citation["cited_text"]
                if len(cited_text) > 150:
                    cited_text = cited_text[:150] + "..."
                text.append(f"   Quote: \"{cited_text}\"")
            
            if citation.get("confidence"):
                text.append(f"   Confidence: {citation['confidence']}")
            
            text.append("")  # Empty line between citations
    else:
        text.append(f"\n📚 SOURCES: Based on {result.get('documents_searched', 0)} documents in knowledge base")
    
    # Add metadata
    text.append(f"\n📊 SEARCH METADATA")
    text.append(f"Query: {result['query']}")
    text.append(f"Documents searched: {result.get('documents_searched', 0)}")
    text.append(f"Citations found: {result.get('citation_count', 0)}")
    
    return "\n".join(text)


def format_citation_markdown(result: Dict[str, Any]) -> str:
    """
    Format a search result with citations into markdown format.
    
    Args:
        result: Search result from search_compliance_docs()
        
    Returns:
        Markdown formatted string with answer and citations
    """
    if result["status"] != "success":
        return f"**Error:** {result.get('answer', 'Search failed')}"
    
    lines = []
    
    # Main answer
    lines.append("## 📋 Compliance Information")
    lines.append("")
    lines.append(result["answer"])
    lines.append("")
    
    # Citations
    citations = result.get("citations", [])
    formatted_citations = result.get("formatted_citations", [])
    
    if citations and len(citations) > 0:
        lines.append("## 📚 Sources")
        lines.append("")
        
        for citation in formatted_citations:
            lines.append(f"**{citation['formatted_reference']}**")
            
            if citation.get("cited_text"):
                cited_text = citation["cited_text"]
                if len(cited_text) > 200:
                    cited_text = cited_text[:200] + "..."
                lines.append(f"> \"{cited_text}\"")
            
            if citation.get("confidence"):
                lines.append(f"*Confidence: {citation['confidence']}*")
            
            lines.append("")
    else:
        lines.append(f"**Sources:** Based on {result.get('documents_searched', 0)} documents in knowledge base")
        lines.append("")
    
    # Metadata
    lines.append("---")
    lines.append(f"**Query:** {result['query']}")
    lines.append(f"**Documents searched:** {result.get('documents_searched', 0)}")
    lines.append(f"**Citations found:** {result.get('citation_count', 0)}")
    
    return "\n".join(lines)


def extract_document_sources(result: Dict[str, Any]) -> List[str]:
    """
    Extract a list of unique source documents from citations.
    
    Args:
        result: Search result from search_compliance_docs()
        
    Returns:
        List of unique document names that were cited
    """
    sources = set()
    
    for citation in result.get("citations", []):
        source = citation.get("source", "")
        if source:
            # Extract just the filename for cleaner display
            if "/" in source:
                source = source.split("/")[-1]
            sources.add(source)
    
    return sorted(list(sources))


def get_citations_by_document(result: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """
    Group citations by source document.
    
    Args:
        result: Search result from search_compliance_docs()
        
    Returns:
        Dict mapping document names to lists of citations from that document
    """
    by_document = {}
    
    for citation in result.get("citations", []):
        source = citation.get("source", "Unknown")
        
        # Extract document name
        if "/" in source:
            doc_name = source.split("/")[-1]
        else:
            doc_name = source
        
        if doc_name not in by_document:
            by_document[doc_name] = []
        
        by_document[doc_name].append(citation)
    
    return by_document


def validate_citation_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the quality and completeness of citations in a result.
    
    Args:
        result: Search result from search_compliance_docs()
        
    Returns:
        Dict with citation quality analysis
    """
    citations = result.get("citations", [])
    
    if not citations:
        return {
            "has_citations": False,
            "citation_count": 0,
            "quality_score": 0.0,
            "issues": ["No citations provided"]
        }
    
    quality_checks = {
        "has_source": 0,
        "has_cited_text": 0,
        "has_confidence": 0,
        "source_is_detailed": 0
    }
    
    issues = []
    
    for citation in citations:
        if citation.get("source"):
            quality_checks["has_source"] += 1
            if len(citation["source"]) > 10:  # More than just a short identifier
                quality_checks["source_is_detailed"] += 1
        
        if citation.get("cited_text"):
            quality_checks["has_cited_text"] += 1
        
        if citation.get("confidence") is not None:
            quality_checks["has_confidence"] += 1
    
    total_citations = len(citations)
    
    # Calculate quality score (0-1)
    quality_score = sum(quality_checks.values()) / (total_citations * len(quality_checks))
    
    # Identify issues
    if quality_checks["has_source"] < total_citations:
        issues.append("Some citations missing source information")
    
    if quality_checks["has_cited_text"] < total_citations * 0.5:
        issues.append("Many citations missing specific quoted text")
    
    if quality_checks["source_is_detailed"] < total_citations * 0.7:
        issues.append("Source information could be more detailed")
    
    return {
        "has_citations": True,
        "citation_count": total_citations,
        "quality_score": quality_score,
        "quality_checks": quality_checks,
        "issues": issues if issues else ["Citation quality looks good"]
    }