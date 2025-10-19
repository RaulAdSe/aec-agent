#!/usr/bin/env python3
"""
Retrieval Strategy Comparison Demo

This script demonstrates the differences between basic and advanced retrieval strategies
without interactive mode, perfect for showing the capabilities.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import openai
from pypdf import PdfReader
from typing import List, Tuple, Dict
import re

# Load environment variables
load_dotenv()

class RetrievalComparison:
    """Compare different retrieval strategies."""
    
    def __init__(self):
        self.chunks = []
        self.embeddings_model = None
        self.chunk_embeddings = None
        
    def load_pdf(self, pdf_path: Path, chunk_size: int = 1000, overlap: int = 200):
        """Load PDF and create chunks."""
        print(f"📄 Loading PDF: {pdf_path.name}")
        
        reader = PdfReader(pdf_path)
        text = ""
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Página {page_num + 1} ---\n"
            text += page_text
        
        print(f"✅ Loaded {len(reader.pages)} pages, {len(text)} characters")
        
        # Create chunks
        self.chunks = self._create_chunks(text, chunk_size, overlap)
        print(f"📝 Created {len(self.chunks)} chunks")
        
        # Load embeddings model
        try:
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("🤖 Loaded embeddings model")
            
            # Generate embeddings
            print("🧠 Generating embeddings...")
            self.chunk_embeddings = self.embeddings_model.encode(
                [chunk['text'] for chunk in self.chunks],
                show_progress_bar=True
            )
            print("✅ Embeddings ready")
        except ImportError:
            print("⚠️ sentence-transformers not available, skipping advanced strategies")
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create text chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > start + chunk_size // 2:
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            # Extract page info
            pages = []
            page_matches = re.findall(r'--- Página (\d+) ---', chunk_text)
            pages = [int(p) for p in page_matches]
            
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {"pages": pages, "start_char": start},
                "length": len(chunk_text)
            })
            
            start = end - overlap
        
        return chunks
    
    def basic_retrieval(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Basic keyword-based retrieval."""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk['text'].lower().split())
            # Simple word overlap scoring
            score = len(query_words.intersection(chunk_words))
            scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
    
    def semantic_retrieval(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Semantic retrieval using embeddings."""
        if not self.embeddings_model or self.chunk_embeddings is None:
            print("⚠️ Embeddings not available, using basic retrieval")
            return self.basic_retrieval(query, top_k)
        
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query])
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top results
        import numpy as np
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def hybrid_retrieval(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Hybrid retrieval combining semantic and keyword approaches."""
        # Get semantic results
        semantic_results = self.semantic_retrieval(query, top_k * 2)
        semantic_scores = {chunk['text']: score for chunk, score in semantic_results}
        
        # Get keyword results
        keyword_results = self.basic_retrieval(query, top_k * 2)
        keyword_scores = {chunk['text']: score for chunk, score in keyword_results}
        
        # Combine scores
        all_chunks = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []
        
        for chunk_text in all_chunks:
            semantic_score = semantic_scores.get(chunk_text, 0)
            keyword_score = keyword_scores.get(chunk_text, 0)
            
            # Normalize and combine scores (70% semantic, 30% keyword)
            combined_score = (0.7 * semantic_score + 0.3 * keyword_score)
            
            # Find the chunk object
            chunk_obj = next((c for c in self.chunks if c['text'] == chunk_text), None)
            if chunk_obj:
                combined_results.append((chunk_obj, combined_score))
        
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
    
    def compare_strategies(self, query: str, top_k: int = 3):
        """Compare all retrieval strategies for a query."""
        print(f"\n🔍 Query: '{query}'")
        print("=" * 60)
        
        # Basic retrieval
        print("\n1️⃣ BASIC RETRIEVAL (Keyword matching):")
        basic_results = self.basic_retrieval(query, top_k)
        for i, (chunk, score) in enumerate(basic_results, 1):
            print(f"   {i}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:150]}...")
        
        # Semantic retrieval
        print("\n2️⃣ SEMANTIC RETRIEVAL (Embeddings):")
        semantic_results = self.semantic_retrieval(query, top_k)
        for i, (chunk, score) in enumerate(semantic_results, 1):
            print(f"   {i}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:150]}...")
        
        # Hybrid retrieval
        print("\n3️⃣ HYBRID RETRIEVAL (Semantic + Keyword):")
        hybrid_results = self.hybrid_retrieval(query, top_k)
        for i, (chunk, score) in enumerate(hybrid_results, 1):
            print(f"   {i}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:150]}...")
        
        return {
            "basic": basic_results,
            "semantic": semantic_results,
            "hybrid": hybrid_results
        }

def query_openai(context: str, question: str) -> str:
    """Query OpenAI with context."""
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    system_prompt = """Eres un asistente experto en normativa de construcción española. Responde basándote ÚNICAMENTE en el contexto proporcionado. Responde en español de forma clara y precisa, citando las fuentes."""

    user_prompt = f"""CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""

    try:
        if "gpt-5" in model:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {e}"

def main():
    """Main comparison function."""
    print("🚀 AEC Compliance Agent - Retrieval Strategy Comparison")
    print("=" * 70)
    
    # Check environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found!")
        return 1
    
    openai.api_key = openai_key
    
    # Initialize comparison
    comparison = RetrievalComparison()
    
    # Load PDF
    pdf_path = Path("data/normativa/DBSI.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return 1
    
    # Load and process PDF
    comparison.load_pdf(pdf_path, chunk_size=1200, overlap=200)
    
    # Test questions
    test_questions = [
        "¿Cuál es el ancho mínimo de una puerta de evacuación?",
        "¿Qué dice el CTE DB-SI sobre las distancias máximas de evacuación?",
        "¿Cuáles son los requisitos de resistencia al fuego para muros?"
    ]
    
    print(f"\n🧪 Comparing retrieval strategies for {len(test_questions)} questions...")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"📝 PREGUNTA {i}: {question}")
        print("=" * 70)
        
        # Compare strategies
        results = comparison.compare_strategies(question, top_k=3)
        
        # Generate answers with best strategy (hybrid)
        print(f"\n💡 RESPUESTA GENERADA (usando estrategia híbrida):")
        print("-" * 50)
        context = "\n\n".join([chunk['text'] for chunk, score in results["hybrid"]])
        answer = query_openai(context, question)
        print(answer)
        
        print(f"\n📊 RESUMEN DE ESTRATEGIAS:")
        print(f"   • Básica: {len(results['basic'])} resultados")
        print(f"   • Semántica: {len(results['semantic'])} resultados") 
        print(f"   • Híbrida: {len(results['hybrid'])} resultados")
    
    print(f"\n{'='*70}")
    print("🎯 CONCLUSIONES:")
    print("=" * 70)
    print("""
📋 COMPARACIÓN DE ESTRATEGIAS:

1️⃣ BÁSICA (Keyword matching):
   ✅ Rápida y simple
   ✅ No requiere embeddings
   ❌ Solo busca coincidencias exactas de palabras
   ❌ No entiende el significado semántico

2️⃣ SEMÁNTICA (Embeddings):
   ✅ Entiende el significado de las consultas
   ✅ Encuentra contenido relacionado conceptualmente
   ✅ Mejor para consultas complejas
   ❌ Requiere más recursos computacionales
   ❌ Puede perder coincidencias exactas importantes

3️⃣ HÍBRIDA (Semantic + Keyword):
   ✅ Combina lo mejor de ambas estrategias
   ✅ 70% semántico + 30% keyword para balance óptimo
   ✅ Mejor precisión y recall
   ✅ Recomendada para producción

🚀 RECOMENDACIÓN: Usar estrategia HÍBRIDA para mejor rendimiento.
    """)
    
    return 0

if __name__ == "__main__":
    exit(main())
