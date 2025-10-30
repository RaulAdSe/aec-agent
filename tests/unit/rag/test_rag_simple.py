#!/usr/bin/env python3
"""
Simple RAG system test to verify it works correctly.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_rag_system():
    print("🔍 Testing RAG System")
    print("=" * 50)
    
    try:
        from rag.vectorstore_manager import VectorstoreManager
        from rag.qa_chain import create_qa_chain
        from rag.document_loader import DocumentLoader
        from rag.embeddings_config import get_embeddings
        
        print("✅ RAG modules imported successfully")
        
        # Test embeddings
        embeddings = get_embeddings()
        test_vector = embeddings.embed_query("test query")
        print(f"✅ Embeddings working: {len(test_vector)} dimensions")
        
        # Test document loader
        doc_loader = DocumentLoader()
        print("✅ Document loader created")
        
        # Test vectorstore manager
        vectorstore_dir = Path("vectorstore/test_db")
        if not vectorstore_dir.exists():
            vectorstore_dir.mkdir(parents=True)
        
        rag_manager = VectorstoreManager(vectorstore_dir)
        print("✅ Vectorstore manager created")
        
        # Test with a simple document
        test_docs = [
            "Ancho mínimo de puerta de evacuación: 80 cm según CTE DB-SI",
            "Distancia máxima de evacuación: 30 metros en edificios de uso residencial"
        ]
        
        # Add documents to vectorstore
        rag_manager.add_documents(test_docs)
        print("✅ Documents added to vectorstore")
        
        # Test retrieval
        retriever = rag_manager.get_retriever(k=2)
        docs = retriever.get_relevant_documents("ancho puerta evacuación")
        print(f"✅ Retrieval working: found {len(docs)} relevant documents")
        
        # Test QA chain
        qa_chain = create_qa_chain(retriever, temperature=0.1)
        result = qa_chain.invoke({"query": "¿Cuál es el ancho mínimo de puerta de evacuación?"})
        print(f"✅ QA chain working: {result['result'][:100]}...")
        
        # Clean up
        import shutil
        if vectorstore_dir.exists():
            shutil.rmtree(vectorstore_dir)
        print("✅ Cleanup completed")
        
        print("\n🎉 RAG System Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ RAG System Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1)
