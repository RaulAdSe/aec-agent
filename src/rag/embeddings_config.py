"""
Embeddings configuration for RAG system.

This module provides the embeddings setup for the vectorstore.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Create embeddings instance.
    Using multilingual model for Spanish support.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
