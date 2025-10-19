#!/usr/bin/env python3
"""
Advanced RAG Demo with proper retrieval and reranking.

This script implements:
1. Semantic embeddings for better retrieval
2. Multiple retrieval strategies
3. Reranking for improved relevance
4. Hybrid search (semantic + keyword)
"""

import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import openai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Dict
from collections import Counter

# Load environment variables
load_dotenv()

class AdvancedRetriever:
    """Advanced retrieval system with embeddings and reranking."""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize the retriever with embeddings model."""
        print(f"🤖 Loading embeddings model: {model_name}")
        self.embeddings_model = SentenceTransformer(model_name)
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []
        
    def load_and_chunk_pdf(self, pdf_path: Path, chunk_size: int = 1000, overlap: int = 200) -> None:
        """Load PDF and create semantic chunks."""
        print(f"📄 Loading PDF: {pdf_path.name}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Página {page_num + 1} ---\n"
                text += page_text
            
            print(f"✅ Loaded {len(reader.pages)} pages, {len(text)} characters")
            
            # Create chunks with metadata
            self.chunks = self._create_semantic_chunks(text, chunk_size, overlap)
            print(f"📝 Created {len(self.chunks)} semantic chunks")
            
            # Generate embeddings for all chunks
            print("🧠 Generating embeddings...")
            self.chunk_embeddings = self.embeddings_model.encode(
                [chunk['text'] for chunk in self.chunks],
                show_progress_bar=True
            )
            print(f"✅ Generated {len(self.chunk_embeddings)} embeddings")
            
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise
    
    def _create_semantic_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create chunks with semantic boundaries."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_metadata = {"pages": [], "start_char": 0}
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Extract page info
            page_match = re.search(r'--- Página (\d+) ---', para)
            if page_match:
                page_num = int(page_match.group(1))
                current_metadata["pages"].append(page_num)
                para = re.sub(r'--- Página \d+ ---', '', para).strip()
            
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": current_metadata.copy(),
                    "length": len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                current_metadata["start_char"] += len(current_chunk) - len(overlap_text)
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": current_metadata.copy(),
                "length": len(current_chunk)
            })
        
        return chunks
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Semantic search using embeddings."""
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Keyword-based search with TF-IDF-like scoring."""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks:
            chunk_words = set(chunk['text'].lower().split())
            
            # Calculate various keyword scores
            word_overlap = len(query_words.intersection(chunk_words))
            word_density = word_overlap / len(query_words) if query_words else 0
            chunk_word_count = len(chunk_words)
            
            # Combined score
            score = word_overlap * 0.5 + word_density * 0.3 + (1 / (1 + chunk_word_count)) * 0.2
            scored_chunks.append((chunk, score))
        
        # Sort and return top results
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 10, semantic_weight: float = 0.7) -> List[Tuple[Dict, float]]:
        """Hybrid search combining semantic and keyword approaches."""
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k * 2)
        semantic_scores = {chunk['text']: score for chunk, score in semantic_results}
        
        # Get keyword results
        keyword_results = self.keyword_search(query, top_k * 2)
        keyword_scores = {chunk['text']: score for chunk, score in keyword_results}
        
        # Combine scores
        all_chunks = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []
        
        for chunk_text in all_chunks:
            semantic_score = semantic_scores.get(chunk_text, 0)
            keyword_score = keyword_scores.get(chunk_text, 0)
            
            # Normalize scores to 0-1 range
            combined_score = (semantic_weight * semantic_score + 
                            (1 - semantic_weight) * keyword_score)
            
            # Find the chunk object
            chunk_obj = next((c for c in self.chunks if c['text'] == chunk_text), None)
            if chunk_obj:
                combined_results.append((chunk_obj, combined_score))
        
        # Sort and return top results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
    
    def rerank_results(self, query: str, results: List[Tuple[Dict, float]], 
                      rerank_top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Rerank results using more sophisticated scoring."""
        if not results:
            return results
        
        reranked = []
        
        for chunk, original_score in results:
            # Calculate additional relevance signals
            relevance_score = self._calculate_relevance_score(query, chunk)
            
            # Combine original score with relevance score
            final_score = original_score * 0.6 + relevance_score * 0.4
            reranked.append((chunk, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:rerank_top_k]
    
    def _calculate_relevance_score(self, query: str, chunk: Dict) -> float:
        """Calculate additional relevance signals."""
        text = chunk['text'].lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Exact phrase matching
        if query_lower in text:
            score += 0.3
        
        # Query word density
        query_words = query_lower.split()
        chunk_words = text.split()
        word_matches = sum(1 for word in query_words if word in chunk_words)
        word_density = word_matches / len(query_words) if query_words else 0
        score += word_density * 0.2
        
        # Length penalty (prefer medium-length chunks)
        length_penalty = 1.0 - abs(len(chunk['text']) - 800) / 1000
        score += max(0, length_penalty) * 0.1
        
        # Page position bonus (earlier pages often more important)
        if chunk['metadata']['pages']:
            page_bonus = 1.0 - (min(chunk['metadata']['pages']) / 100)
            score += max(0, page_bonus) * 0.1
        
        return min(1.0, score)

def query_openai_with_context(context: str, question: str, model: str = None) -> str:
    """Query OpenAI with enhanced context."""
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
7. Incluye valores numéricos específicos cuando estén disponibles
8. Estructura tu respuesta de forma clara y organizada"""

    user_prompt = f"""CONTEXTO RELEVANTE:
{context}

PREGUNTA: {question}

RESPUESTA DETALLADA Y ESTRUCTURADA:"""

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
                max_completion_tokens=1500
            )
        else:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error al consultar OpenAI: {e}"

def main():
    """Main demo function."""
    print("🚀 AEC Compliance Agent - Advanced RAG Demo")
    print("=" * 60)
    
    # Check environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found!")
        return 1
    
    # Set OpenAI API key
    openai.api_key = openai_key
    
    # Initialize retriever
    retriever = AdvancedRetriever()
    
    # Load PDF
    pdf_path = Path("data/normativa/DBSI.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return 1
    
    # Load and process PDF
    retriever.load_and_chunk_pdf(pdf_path, chunk_size=1200, overlap=200)
    
    # Test different retrieval strategies
    test_questions = [
        "¿Cuál es el ancho mínimo de una puerta de evacuación?",
        "¿Qué dice el CTE DB-SI sobre las distancias máximas de evacuación?",
        "¿Cuáles son los requisitos de resistencia al fuego para muros?",
        "¿Qué tipos de sistemas de detección de incendios se mencionan?",
        "¿Cuáles son las condiciones para la sectorización de incendios?"
    ]
    
    print(f"\n🧪 Testing Advanced RAG with {len(test_questions)} questions...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Pregunta {i}: {question}")
        print("-" * 50)
        
        # Test different retrieval methods
        print("\n🔍 Métodos de recuperación:")
        
        # 1. Semantic search
        print("1️⃣ Búsqueda semántica:")
        semantic_results = retriever.semantic_search(question, top_k=5)
        for j, (chunk, score) in enumerate(semantic_results[:2], 1):
            print(f"   {j}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:100]}...")
        
        # 2. Keyword search
        print("\n2️⃣ Búsqueda por palabras clave:")
        keyword_results = retriever.keyword_search(question, top_k=5)
        for j, (chunk, score) in enumerate(keyword_results[:2], 1):
            print(f"   {j}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:100]}...")
        
        # 3. Hybrid search
        print("\n3️⃣ Búsqueda híbrida:")
        hybrid_results = retriever.hybrid_search(question, top_k=5)
        for j, (chunk, score) in enumerate(hybrid_results[:2], 1):
            print(f"   {j}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:100]}...")
        
        # 4. Reranked results
        print("\n4️⃣ Resultados rerankeados:")
        reranked_results = retriever.rerank_results(question, hybrid_results, rerank_top_k=3)
        for j, (chunk, score) in enumerate(reranked_results, 1):
            print(f"   {j}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:100]}...")
        
        # Generate answer with best results
        print(f"\n💡 Generando respuesta con los mejores resultados...")
        context = "\n\n".join([chunk['text'] for chunk, score in reranked_results])
        answer = query_openai_with_context(context, question)
        
        print(f"\n🎯 RESPUESTA FINAL:")
        print(f"{answer}")
        print("\n" + "="*60)
    
    # Interactive mode
    print("\n💬 Modo Interactivo Avanzado (escribe 'quit' para salir)")
    print("Comandos especiales:")
    print("  - 'semantic <pregunta>' - Solo búsqueda semántica")
    print("  - 'keyword <pregunta>' - Solo búsqueda por palabras clave")
    print("  - 'hybrid <pregunta>' - Búsqueda híbrida")
    print("  - 'rerank <pregunta>' - Búsqueda híbrida + reranking")
    
    while True:
        try:
            user_input = input("\n❓ Tu consulta: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'salir']:
                break
            
            if not user_input:
                continue
            
            # Parse command
            if user_input.startswith('semantic '):
                question = user_input[9:]
                results = retriever.semantic_search(question, top_k=3)
                method = "semántica"
            elif user_input.startswith('keyword '):
                question = user_input[8:]
                results = retriever.keyword_search(question, top_k=3)
                method = "por palabras clave"
            elif user_input.startswith('hybrid '):
                question = user_input[7:]
                results = retriever.hybrid_search(question, top_k=3)
                method = "híbrida"
            elif user_input.startswith('rerank '):
                question = user_input[7:]
                results = retriever.rerank_results(
                    question, 
                    retriever.hybrid_search(question, top_k=5), 
                    rerank_top_k=3
                )
                method = "híbrida + reranking"
            else:
                question = user_input
                results = retriever.rerank_results(
                    question, 
                    retriever.hybrid_search(question, top_k=5), 
                    rerank_top_k=3
                )
                method = "híbrida + reranking (por defecto)"
            
            print(f"\n🔍 Usando búsqueda {method}")
            print("📊 Resultados encontrados:")
            for j, (chunk, score) in enumerate(results, 1):
                print(f"   {j}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            
            # Generate answer
            print("\n🤔 Generando respuesta...")
            context = "\n\n".join([chunk['text'] for chunk, score in results])
            answer = query_openai_with_context(context, question)
            
            print(f"\n💡 RESPUESTA:")
            print(f"{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main())
