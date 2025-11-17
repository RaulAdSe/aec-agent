#!/usr/bin/env python3
"""
Quick test to verify Gemini API integration works
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini_api():
    """Test if Gemini API is working properly."""
    print("🔍 Testing Gemini API Connection")
    print("=" * 50)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No GEMINI_API_KEY found in environment")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        genai.configure(api_key=api_key)
        
        # First, list available models
        print("📋 Listing available models...")
        models = genai.list_models()
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        print(f"✅ Found {len(available_models)} models that support text generation:")
        for model_name in available_models[:5]:  # Show first 5
            print(f"  - {model_name}")
        
        # Try with multiple models
        models_to_try = [
            "models/gemini-2.5-flash",
            "models/gemini-pro",
            available_models[0] if available_models else None
        ]
        
        for model_name in models_to_try:
            if model_name and model_name in available_models:
                print(f"\n🤖 Testing with model: {model_name}")
                
                try:
                    model = genai.GenerativeModel(model_name)
                    
                    test_prompt = "Explain what analyzing building doors means in construction compliance."
                    
                    response = model.generate_content(
                        test_prompt,
                        generation_config={
                            "max_output_tokens": 50,
                            "temperature": 0.1,
                        }
                    )
                    
                    # Check if response has content
                    if response.candidates and response.candidates[0].content:
                        print(f"🎯 Response: {response.text}")
                        print("✅ Gemini API working correctly!")
                        return True, model_name
                    else:
                        print(f"⚠️ Model {model_name} returned empty response or was blocked")
                        
                except Exception as e:
                    print(f"❌ Error with model {model_name}: {e}")
                    continue
        
        print("❌ No models worked successfully")
        return False
        
    except Exception as e:
        print(f"❌ Error testing Gemini API: {e}")
        return False

if __name__ == "__main__":
    result = test_gemini_api()
    if isinstance(result, tuple):
        success, model_name = result
        if success:
            print(f"\n🎉 Use this model in your app: {model_name}")
    else:
        print("\n❌ API test failed")