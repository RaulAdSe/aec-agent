#!/usr/bin/env python3
"""
Test OpenAI-powered dynamic insights generation
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_openai_insights():
    """Test OpenAI API for generating dynamic insights."""
    print("🔍 Testing OpenAI API for Dynamic Insights")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OPENAI_API_KEY found")
        return False
    
    print(f"✅ API key found: {api_key[:15]}...")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test different types of insights
        test_prompts = [
            ("Analyzing Question", "Tell me about the doors", "In 1 sentence, explain what an AEC compliance agent is analyzing when a user asks: 'Tell me about the doors'. Focus on building safety and code compliance."),
            ("Data Source Check", "Found: 1 IFC building model(s)", "In 1 sentence, explain what it means for a building compliance system to find: Found: 1 IFC building model(s)"),
            ("Tool Selection", "IFC Building Data Analyzer", "In 1 sentence, explain why a building compliance agent would select these tools: IFC Building Data Analyzer"),
            ("Door Analyzer", "Checking 5 doors against ADA standards", "In 1 sentence, explain what door compliance analysis involves in building safety: Checking 5 doors against ADA standards"),
            ("Stair Analysis", "Examining stairs for code compliance", "In 1 sentence, explain what stair safety analysis means for building compliance: Examining stairs for code compliance")
        ]
        
        print("\n🤖 Testing Dynamic Insight Generation:")
        print("-" * 60)
        
        for action, context, prompt in test_prompts:
            print(f"\n📋 {action}")
            print(f"   Context: {context}")
            
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert in AEC (Architecture, Engineering, Construction) compliance. Provide concise, professional insights about building analysis steps. Keep responses under 120 characters and focus on safety, codes, and regulations."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    max_tokens=50,
                    temperature=0.3
                )
                
                insight = response.choices[0].message.content.strip()
                print(f"   💡 Insight: {insight}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                return False
        
        print("\n" + "=" * 60)
        print("✅ OpenAI API working perfectly for dynamic insights!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing OpenAI API: {e}")
        return False

if __name__ == "__main__":
    test_openai_insights()