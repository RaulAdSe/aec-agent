#!/usr/bin/env python3
"""
Test script for RAG system implementation.

This script demonstrates how to use the RAG system to query building code documents.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import VectorstoreManager, create_qa_chain, query, format_response


def main():
    """Main function to test RAG system."""
    print("🔍 AEC Compliance Agent - RAG System Test")
    print("=" * 50)
    
    # Configuration
    PDF_DIR = Path("data/normativa")
    VECTORSTORE_DIR = Path("vectorstore/normativa_db")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    # Check if PDF directory exists
    if not PDF_DIR.exists():
        print(f"❌ Error: PDF directory {PDF_DIR} does not exist")
        return
    
    try:
        # Initialize vectorstore manager
        print("📚 Initializing vectorstore manager...")
        rag = VectorstoreManager(VECTORSTORE_DIR)
        
        # Create or load vectorstore
        if not VECTORSTORE_DIR.exists():
            print("🔄 Creating vectorstore from PDFs (this may take a few minutes)...")
            rag.create_from_pdfs(PDF_DIR)
        else:
            print("📂 Loading existing vectorstore...")
            rag.load_existing()
        
        # Get vectorstore info
        info = rag.get_vectorstore_info()
        print(f"✅ Vectorstore ready: {info['document_count']} documents indexed")
        
        # Create retriever
        print("🔗 Creating retriever...")
        retriever = rag.get_retriever(k=3)
        
        # Create QA chain
        print("🤖 Creating QA chain with OpenAI...")
        qa_chain = create_qa_chain(retriever)
        
        # Test questions
        test_questions = [
            "¿Cuál es el ancho mínimo de una puerta de evacuación?",
            "¿Qué dice el CTE DB-SI sobre las distancias máximas de evacuación?",
            "¿Cuáles son los requisitos de resistencia al fuego para muros?",
            "¿Qué tipos de sistemas de detección de incendios se mencionan?",
            "¿Cuáles son las condiciones para la sectorización de incendios?"
        ]
        
        print("\n" + "=" * 50)
        print("🧪 Testing RAG System")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Pregunta {i}: {question}")
            print("-" * 40)
            
            try:
                # Query the system
                result = query(qa_chain, question)
                
                # Format and display response
                formatted_response = format_response(result)
                print(formatted_response)
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
        
        print("\n" + "=" * 50)
        print("✅ RAG System Test Complete")
        print("=" * 50)
        
        # Interactive mode
        print("\n💬 Interactive Mode (type 'quit' to exit)")
        while True:
            try:
                user_question = input("\n❓ Tu pregunta: ").strip()
                
                if user_question.lower() in ['quit', 'exit', 'salir']:
                    break
                
                if not user_question:
                    continue
                
                print("🤔 Procesando...")
                result = query(qa_chain, user_question)
                formatted_response = format_response(result)
                print(f"\n💡 Respuesta:\n{formatted_response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
