#!/usr/bin/env python3
"""
Unified RAG System with configurable retrieval strategies.

This script allows switching between:
1. Basic retrieval (keyword-based)
2. Advanced retrieval (semantic + reranking)
3. Hybrid modes
"""

import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import openai
from pypdf import PdfReader
from typing import List, Tuple, Dict, Optional
from enum import Enum
import argparse

# Load environment variables
load_dotenv()

class RetrievalStrategy(Enum):
    """Available retrieval strategies."""
    BASIC = "basic"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    RERANK = "rerank"

class UnifiedRAGSystem:
    """Unified RAG system with configurable retrieval strategies."""
    
    def __init__(self, strategy: RetrievalStrategy = RetrievalStrategy.RERANK):
        """Initialize the RAG system with specified strategy."""
        self.strategy = strategy
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []
        
        # Initialize embeddings model only if needed
        if strategy in [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID, RetrievalStrategy.RERANK]:
            print(f"🤖 Loading embeddings model for {strategy.value} strategy...")
            try:
                from sentence_transformers import SentenceTransformer
                self.embeddings_model = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                print("✅ Embeddings model loaded")
            except ImportError:
                print("❌ sentence-transformers not available. Falling back to basic strategy.")
                self.strategy = RetrievalStrategy.BASIC
                self.embeddings_model = None
        else:
            self.embeddings_model = None
            print(f"📝 Using {strategy.value} strategy (no embeddings needed)")
    
    def load_and_chunk_pdf(self, pdf_path: Path, chunk_size: int = 1000, overlap: int = 200) -> None:
        """Load PDF and create chunks based on strategy."""
        print(f"📄 Loading PDF: {pdf_path.name}")
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Página {page_num + 1} ---\n"
                page_text
            
            print(f"✅ Loaded {len(reader.pages)} pages, {len(text)} characters")
            
            # Create chunks based on strategy
            if self.strategy == RetrievalStrategy.BASIC:
                self.chunks = self._create_basic_chunks(text, chunk_size, overlap)
            else:
                self.chunks = self._create_semantic_chunks(text, chunk_size, overlap)
            
            print(f"📝 Created {len(self.chunks)} chunks using {self.strategy.value} strategy")
            
            # Generate embeddings if needed
            if self.embeddings_model and self.chunks:
                print("🧠 Generating embeddings...")
                self.chunk_embeddings = self.embeddings_model.encode(
                    [chunk['text'] for chunk in self.chunks],
                    show_progress_bar=True
                )
                print(f"✅ Generated {len(self.chunk_embeddings)} embeddings")
            
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise
    
    def _create_basic_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create simple text chunks for basic retrieval."""
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
    
    def _create_semantic_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Create semantic chunks with better boundaries."""
        import re
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
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve relevant chunks using the configured strategy."""
        if not self.chunks:
            return []
        
        if self.strategy == RetrievalStrategy.BASIC:
            return self._basic_retrieval(query, top_k)
        elif self.strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_retrieval(query, top_k)
        elif self.strategy == RetrievalStrategy.KEYWORD:
            return self._keyword_retrieval(query, top_k)
        elif self.strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_retrieval(query, top_k)
        elif self.strategy == RetrievalStrategy.RERANK:
            return self._rerank_retrieval(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _basic_retrieval(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
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
    
    def _semantic_retrieval(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Semantic retrieval using embeddings."""
        if not self.embeddings_model or self.chunk_embeddings is None:
            print("⚠️ Embeddings not available, falling back to basic retrieval")
            return self._basic_retrieval(query, top_k)
        
        # Generate query embedding
        query_embedding = self.embeddings_model.encode([query])
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def _keyword_retrieval(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Advanced keyword-based retrieval."""
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
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
    
    def _hybrid_retrieval(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Hybrid retrieval combining semantic and keyword approaches."""
        # Get semantic results
        semantic_results = self._semantic_retrieval(query, top_k * 2)
        semantic_scores = {chunk['text']: score for chunk, score in semantic_results}
        
        # Get keyword results
        keyword_results = self._keyword_retrieval(query, top_k * 2)
        keyword_scores = {chunk['text']: score for chunk, score in keyword_results}
        
        # Combine scores
        all_chunks = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_results = []
        
        for chunk_text in all_chunks:
            semantic_score = semantic_scores.get(chunk_text, 0)
            keyword_score = keyword_scores.get(chunk_text, 0)
            
            # Normalize and combine scores
            combined_score = (0.7 * semantic_score + 0.3 * keyword_score)
            
            # Find the chunk object
            chunk_obj = next((c for c in self.chunks if c['text'] == chunk_text), None)
            if chunk_obj:
                combined_results.append((chunk_obj, combined_score))
        
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]
    
    def _rerank_retrieval(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Hybrid retrieval with reranking."""
        # Get hybrid results
        hybrid_results = self._hybrid_retrieval(query, top_k * 2)
        
        # Rerank with additional signals
        reranked = []
        for chunk, original_score in hybrid_results:
            relevance_score = self._calculate_relevance_score(query, chunk)
            final_score = original_score * 0.6 + relevance_score * 0.4
            reranked.append((chunk, final_score))
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    def _calculate_relevance_score(self, query: str, chunk: Dict) -> float:
        """Calculate additional relevance signals for reranking."""
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
        
        # Page position bonus
        if chunk['metadata']['pages']:
            page_bonus = 1.0 - (min(chunk['metadata']['pages']) / 100)
            score += max(0, page_bonus) * 0.1
        
        return min(1.0, score)
    
    def switch_strategy(self, new_strategy: RetrievalStrategy) -> None:
        """Switch to a different retrieval strategy."""
        print(f"🔄 Switching from {self.strategy.value} to {new_strategy.value}")
        
        # Load embeddings if needed for new strategy
        if new_strategy in [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID, RetrievalStrategy.RERANK]:
            if not self.embeddings_model:
                print("🤖 Loading embeddings model...")
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embeddings_model = SentenceTransformer(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    
                    # Generate embeddings if we have chunks
                    if self.chunks:
                        print("🧠 Generating embeddings...")
                        self.chunk_embeddings = self.embeddings_model.encode(
                            [chunk['text'] for chunk in self.chunks],
                            show_progress_bar=True
                        )
                        print("✅ Embeddings ready")
                except ImportError:
                    print("❌ sentence-transformers not available. Cannot switch to advanced strategy.")
                    return
        
        self.strategy = new_strategy
        print(f"✅ Switched to {self.strategy.value} strategy")

def query_openai_with_context(context: str, question: str, model: str = None) -> str:
    """Query OpenAI with context."""
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
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Unified RAG System")
    parser.add_argument("--strategy", 
                       choices=[s.value for s in RetrievalStrategy],
                       default=RetrievalStrategy.RERANK.value,
                       help="Retrieval strategy to use")
    parser.add_argument("--pdf", 
                       default="data/normativa/DBSI.pdf",
                       help="Path to PDF file")
    
    args = parser.parse_args()
    
    print("🚀 AEC Compliance Agent - Unified RAG System")
    print("=" * 60)
    
    # Check environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ OPENAI_API_KEY not found!")
        return 1
    
    # Set OpenAI API key
    openai.api_key = openai_key
    
    # Initialize RAG system
    strategy = RetrievalStrategy(args.strategy)
    rag = UnifiedRAGSystem(strategy)
    
    # Load PDF
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return 1
    
    # Load and process PDF
    rag.load_and_chunk_pdf(pdf_path, chunk_size=1200, overlap=200)
    
    print(f"\n🎯 Using {rag.strategy.value} retrieval strategy")
    print("=" * 60)
    
    # Test questions
    test_questions = [
        "¿Cuál es el ancho mínimo de una puerta de evacuación?",
        "¿Qué dice el CTE DB-SI sobre las distancias máximas de evacuación?",
        "¿Cuáles son los requisitos de resistencia al fuego para muros?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Pregunta {i}: {question}")
        print("-" * 50)
        
        # Retrieve relevant chunks
        results = rag.retrieve(question, top_k=3)
        
        print(f"🔍 Encontrados {len(results)} resultados con {rag.strategy.value}:")
        for j, (chunk, score) in enumerate(results, 1):
            print(f"   {j}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            print(f"      Texto: {chunk['text'][:100]}...")
        
        # Generate answer
        print("\n🤔 Generando respuesta...")
        context = "\n\n".join([chunk['text'] for chunk, score in results])
        answer = query_openai_with_context(context, question)
        
        print(f"\n💡 RESPUESTA:")
        print(f"{answer}")
        print("\n" + "="*60)
    
    # Interactive mode
    print("\n💬 Modo Interactivo")
    print("Comandos especiales:")
    print("  - 'switch <strategy>' - Cambiar estrategia (basic, semantic, keyword, hybrid, rerank)")
    print("  - 'quit' - Salir")
    
    while True:
        try:
            user_input = input(f"\n❓ [{rag.strategy.value}] Tu consulta: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'salir']:
                break
            
            if not user_input:
                continue
            
            # Handle strategy switching
            if user_input.startswith('switch '):
                new_strategy_name = user_input[7:].strip()
                try:
                    new_strategy = RetrievalStrategy(new_strategy_name)
                    rag.switch_strategy(new_strategy)
                except ValueError:
                    print(f"❌ Estrategia inválida: {new_strategy_name}")
                    print(f"Estrategias disponibles: {[s.value for s in RetrievalStrategy]}")
                continue
            
            # Regular query
            results = rag.retrieve(user_input, top_k=3)
            
            print(f"\n🔍 Resultados ({rag.strategy.value}):")
            for j, (chunk, score) in enumerate(results, 1):
                print(f"   {j}. Score: {score:.3f} | Páginas: {chunk['metadata']['pages']}")
            
            # Generate answer
            print("\n🤔 Generando respuesta...")
            context = "\n\n".join([chunk['text'] for chunk, score in results])
            answer = query_openai_with_context(context, user_input)
            
            print(f"\n💡 RESPUESTA:")
            print(f"{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 0

if __name__ == "__main__":
    import re
    exit(main())
