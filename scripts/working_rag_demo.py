#!/usr/bin/env python3
"""
Working RAG Demo - Simplified version that works with current dependencies.

This script demonstrates the RAG system functionality using direct OpenAI API calls
and basic document processing without complex LangChain dependencies.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import openai
from pypdf import PdfReader

# Load environment variables
load_dotenv()

def load_pdf_content(pdf_path: Path) -> str:
    """Load and extract text from PDF."""
    print(f"📄 Loading PDF: {pdf_path.name}")
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Página {page_num + 1} ---\n"
            text += page_text
        
        print(f"✅ Loaded {len(reader.pages)} pages, {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into chunks for processing."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > start + chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    print(f"📝 Created {len(chunks)} text chunks")
    return chunks

def find_relevant_chunks(query: str, chunks: list, top_k: int = 3) -> list:
    """Find most relevant chunks using simple keyword matching."""
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        # Simple scoring based on word overlap
        score = len(query_words.intersection(chunk_words))
        scored_chunks.append((score, i, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(reverse=True)
    return [chunk for score, i, chunk in scored_chunks[:top_k]]

def query_openai(context: str, question: str, model: str = None) -> str:
    """Query OpenAI with context and question."""
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    system_prompt = """Eres un asistente experto en normativa de construcción española, especializado en el Código Técnico de la Edificación (CTE), especialmente en el Documento Básico de Seguridad en caso de Incendio (DB-SI).

Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado de la normativa.

REGLAS IMPORTANTES:
1. Responde SIEMPRE en español
2. Si la información está en el contexto, responde de forma clara y precisa
3. SIEMPRE cita la fuente específica (documento, sección, página si está disponible)
4. Si no puedes responder con el contexto dado, di claramente "No tengo información suficiente sobre esto en la normativa proporcionada"
5. NO inventes ni supongas información que no esté en el contexto
6. Si hay múltiples requisitos, enuméralos claramente
7. Incluye valores numéricos específicos cuando estén disponibles"""

    user_prompt = f"""CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA DETALLADA:"""

    try:
        # Use model-specific parameters
        if "gpt-5" in model:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        elif "gpt-4" in model:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_completion_tokens=1000
            )
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error al consultar OpenAI: {e}"

def main():
    """Main demo function."""
    print("🚀 AEC Compliance Agent - Working RAG Demo")
    print("=" * 50)
    
    # Check environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found!")
        return 1
    
    # Set OpenAI API key
    openai.api_key = openai_key
    
    # Load PDF
    pdf_path = Path("data/normativa/DBSI.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return 1
    
    # Load and chunk PDF content
    print("\n📚 Processing PDF document...")
    pdf_text = load_pdf_content(pdf_path)
    if not pdf_text:
        print("❌ Failed to load PDF content")
        return 1
    
    chunks = chunk_text(pdf_text, chunk_size=1500, overlap=300)
    
    # Test questions
    test_questions = [
        "¿Cuál es el ancho mínimo de una puerta de evacuación?",
        "¿Qué dice el CTE DB-SI sobre las distancias máximas de evacuación?",
        "¿Cuáles son los requisitos de resistencia al fuego para muros?",
        "¿Qué tipos de sistemas de detección de incendios se mencionan?",
        "¿Cuáles son las condiciones para la sectorización de incendios?"
    ]
    
    print(f"\n🧪 Testing RAG system with {len(test_questions)} questions...")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Pregunta {i}: {question}")
        print("-" * 40)
        
        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(question, chunks, top_k=3)
        context = "\n\n".join(relevant_chunks)
        
        # Query OpenAI
        print("🤔 Procesando...")
        answer = query_openai(context, question)
        
        print(f"💡 Respuesta:\n{answer}")
        print("\n" + "="*50)
    
    # Interactive mode
    print("\n💬 Modo Interactivo (escribe 'quit' para salir)")
    while True:
        try:
            user_question = input("\n❓ Tu pregunta: ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'salir']:
                break
            
            if not user_question:
                continue
            
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(user_question, chunks, top_k=3)
            context = "\n\n".join(relevant_chunks)
            
            # Query OpenAI
            print("🤔 Procesando...")
            answer = query_openai(context, user_question)
            
            print(f"\n💡 Respuesta:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())
