"""
Unit tests for embeddings configuration.
"""

import pytest
from unittest.mock import Mock, patch

from src.rag.embeddings_config import get_embeddings, get_embeddings_model_name


class TestEmbeddingsConfig:
    """Test cases for embeddings configuration."""
    
    def test_get_embeddings_model_name_default(self):
        """Test getting default embeddings model name."""
        with patch('os.getenv', return_value=None):
            model_name = get_embeddings_model_name()
            
            assert model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def test_get_embeddings_model_name_custom(self):
        """Test getting custom embeddings model name from environment."""
        custom_model = "custom/embeddings-model"
        
        with patch('os.getenv', return_value=custom_model):
            model_name = get_embeddings_model_name()
            
            assert model_name == custom_model
    
    @patch('src.rag.embeddings_config.SentenceTransformerEmbeddings')
    def test_get_embeddings_success(self, mock_sentence_transformer):
        """Test successful embeddings creation."""
        # Mock SentenceTransformerEmbeddings
        mock_embeddings = Mock()
        mock_sentence_transformer.return_value = mock_embeddings
        
        # Mock environment
        with patch('src.rag.embeddings_config.get_embeddings_model_name', return_value="test-model"):
            embeddings = get_embeddings()
            
            assert embeddings == mock_embeddings
            mock_sentence_transformer.assert_called_once_with(
                model_name="test-model",
                model_kwargs={"device": "cpu"}
            )
    
    @patch('src.rag.embeddings_config.SentenceTransformerEmbeddings')
    def test_get_embeddings_with_custom_device(self, mock_sentence_transformer):
        """Test embeddings creation with custom device."""
        mock_embeddings = Mock()
        mock_sentence_transformer.return_value = mock_embeddings
        
        with patch('os.getenv', side_effect=lambda key, default=None: "cuda" if key == "EMBEDDINGS_DEVICE" else None):
            embeddings = get_embeddings()
            
            mock_sentence_transformer.assert_called_once_with(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cuda"}
            )
    
    @patch('src.rag.embeddings_config.SentenceTransformerEmbeddings')
    def test_get_embeddings_exception_handling(self, mock_sentence_transformer):
        """Test exception handling in embeddings creation."""
        # Mock SentenceTransformerEmbeddings to raise an exception
        mock_sentence_transformer.side_effect = Exception("Model loading error")
        
        with pytest.raises(Exception, match="Model loading error"):
            get_embeddings()
    
    def test_get_embeddings_caching(self):
        """Test that embeddings are cached (if caching is implemented)."""
        with patch('src.rag.embeddings_config.SentenceTransformerEmbeddings') as mock_sentence_transformer:
            mock_embeddings = Mock()
            mock_sentence_transformer.return_value = mock_embeddings
            
            # Call get_embeddings multiple times
            embeddings1 = get_embeddings()
            embeddings2 = get_embeddings()
            
            # Both should return the same instance
            assert embeddings1 == embeddings2
    
    def test_embeddings_model_validation(self):
        """Test validation of embeddings model name."""
        # Test with valid model name
        valid_model = "sentence-transformers/all-MiniLM-L6-v2"
        with patch('os.getenv', return_value=valid_model):
            model_name = get_embeddings_model_name()
            assert model_name == valid_model
        
        # Test with empty string (should use default)
        with patch('os.getenv', return_value=""):
            model_name = get_embeddings_model_name()
            assert model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    @patch('src.rag.embeddings_config.SentenceTransformerEmbeddings')
    def test_embeddings_initialization_parameters(self, mock_sentence_transformer):
        """Test that embeddings are initialized with correct parameters."""
        mock_embeddings = Mock()
        mock_sentence_transformer.return_value = mock_embeddings
        
        # Test with default parameters
        get_embeddings()
        
        # Verify the call was made with expected parameters
        mock_sentence_transformer.assert_called_once()
        call_args = mock_sentence_transformer.call_args
        
        assert "model_name" in call_args.kwargs
        assert "model_kwargs" in call_args.kwargs
        assert call_args.kwargs["model_kwargs"]["device"] == "cpu"
    
    def test_environment_variable_handling(self):
        """Test proper handling of environment variables."""
        # Test EMBEDDINGS_MODEL
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                "EMBEDDINGS_MODEL": "custom-model",
                "EMBEDDINGS_DEVICE": "gpu"
            }.get(key, default)
            
            model_name = get_embeddings_model_name()
            assert model_name == "custom-model"
    
    @patch('src.rag.embeddings_config.SentenceTransformerEmbeddings')
    def test_embeddings_functionality(self, mock_sentence_transformer):
        """Test that embeddings object has expected functionality."""
        # Create a mock embeddings object with expected methods
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_sentence_transformer.return_value = mock_embeddings
        
        embeddings = get_embeddings()
        
        # Test embed_query method
        result = embeddings.embed_query("test query")
        assert result == [0.1, 0.2, 0.3]
        mock_embeddings.embed_query.assert_called_once_with("test query")
        
        # Test embed_documents method
        docs = ["doc1", "doc2"]
        result = embeddings.embed_documents(docs)
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings.embed_documents.assert_called_once_with(docs)


if __name__ == "__main__":
    pytest.main([__file__])
