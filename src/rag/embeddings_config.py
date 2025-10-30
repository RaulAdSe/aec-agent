"""
Embeddings configuration for RAG system.

Uses multilingual sentence transformers optimized for Spanish text.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """
    Create embeddings instance for multilingual text.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Configured HuggingFaceEmbeddings instance
        
    Notes:
        Default model is optimized for Spanish and other languages.
        Uses CPU by default for compatibility.
    """
    if model_name is None:
        # Use multilingual model optimized for Spanish
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            'device': 'cpu',  # Use CPU for compatibility
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True,  # Normalize for better similarity search
            'batch_size': 32
        }
    )


def get_fast_embeddings() -> HuggingFaceEmbeddings:
    """
    Get faster, smaller embeddings model for development.
    
    Returns:
        Faster embeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_high_quality_embeddings() -> HuggingFaceEmbeddings:
    """
    Get higher quality embeddings model for production.
    
    Returns:
        Higher quality embeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
