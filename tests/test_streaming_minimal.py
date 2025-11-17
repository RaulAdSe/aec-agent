#!/usr/bin/env python3
"""
Test minimal LLM-powered streaming insights
"""

import os
import sys
from pathlib import Path

# Add project to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

def test_llm_insight_generation():
    """Test the minimal LLM insight generation."""
    print("🔍 Testing Minimal LLM Insight Generation")
    print("=" * 50)
    
    # Import after setting up path
    from app import get_llm_insight
    
    test_cases = [
        ("Analyzing Question", "Tell me about the doors"),
        ("Data Source Check", "Found: 1 IFC building model(s)"),
        ("Tool Selection", "IFC Building Data Analyzer"),
        ("Building Analysis", "Processing building data"),
        ("Response Preparation", "Finalizing answer")
    ]
    
    print("\n🤖 Testing LLM Insights:")
    print("-" * 50)
    
    for action, context in test_cases:
        print(f"\n📋 {action}")
        print(f"   Context: {context}")
        
        try:
            insight = get_llm_insight(action, context)
            print(f"   💡 LLM says: {insight}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Minimal LLM insights test complete!")

if __name__ == "__main__":
    test_llm_insight_generation()