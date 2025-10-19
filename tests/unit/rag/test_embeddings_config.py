"""
Unit tests for embeddings configuration.
"""

import pytest
from unittest.mock import patch, Mock

from src.rag.embeddings_config import (
    get_embeddings, 
    get_fast_embeddings, 
    get_high_quality_embeddings
)


class TestEmbeddingsConfig:
    """Test cases for embeddings configuration."""
    
    @patch('src.rag.embeddings_config.HuggingFaceEmbeddings')
    def test_get_embeddings_default(self, mock_embeddings_class):
        """Test default embeddings configuration."""
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        result = get_embeddings()
        
        # Verify HuggingFaceEmbeddings was called with correct parameters
        mock_embeddings_class.assert_called_once_with(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        
        assert result == mock_embeddings
    
    @patch('src.rag.embeddings_config.HuggingFaceEmbeddings')
    def test_get_embeddings_custom_model(self, mock_embeddings_class):
        """Test embeddings with custom model name."""
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        custom_model = "custom-model-name"
        result = get_embeddings(custom_model)
        
        # Verify custom model name was used
        mock_embeddings_class.assert_called_once_with(
            model_name=custom_model,
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        
        assert result == mock_embeddings
    
    @patch('src.rag.embeddings_config.HuggingFaceEmbeddings')
    def test_get_fast_embeddings(self, mock_embeddings_class):
        """Test fast embeddings configuration."""
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        result = get_fast_embeddings()
        
        # Verify fast model was used
        mock_embeddings_class.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        assert result == mock_embeddings
    
    @patch('src.rag.embeddings_config.HuggingFaceEmbeddings')
    def test_get_high_quality_embeddings(self, mock_embeddings_class):
        """Test high quality embeddings configuration."""
        mock_embeddings = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        
        result = get_high_quality_embeddings()
        
        # Verify high quality model was used
        mock_embeddings_class.assert_called_once_with(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        assert result == mock_embeddings
