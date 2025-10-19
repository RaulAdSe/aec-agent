"""
Unit tests for VectorstoreManager class.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.embeddings_config import get_embeddings


class TestVectorstoreManager:
    """Test cases for VectorstoreManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vectorstore_path = Path(self.temp_dir) / "test_vectorstore"
        self.manager = VectorstoreManager(self.vectorstore_path)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test VectorstoreManager initialization."""
        assert self.manager.vectorstore_path == self.vectorstore_path
        assert self.manager.vectorstore is None
        assert self.manager.retriever is None
    
    @patch('src.rag.vectorstore_manager.Chroma')
    def test_create_from_pdfs_success(self, mock_chroma):
        """Test successful vectorstore creation from PDFs."""
        # Mock Chroma and its methods
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        # Mock document loader
        with patch('src.rag.vectorstore_manager.DocumentLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_pdfs.return_value = [Mock(), Mock()]  # Mock documents
            
            # Test
            result = self.manager.create_from_pdfs(
                pdf_dir=Path("test_pdfs"),
                chunk_size=500,
                chunk_overlap=100
            )
            
            # Assertions
            assert result is True
            mock_loader.load_pdfs.assert_called_once()
            mock_chroma.from_documents.assert_called_once()
            assert self.manager.vectorstore == mock_vectorstore
    
    @patch('src.rag.vectorstore_manager.Chroma')
    def test_create_from_pdfs_no_documents(self, mock_chroma):
        """Test vectorstore creation when no documents are found."""
        with patch('src.rag.vectorstore_manager.DocumentLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            mock_loader.load_pdfs.return_value = []  # No documents
            
            result = self.manager.create_from_pdfs(Path("test_pdfs"))
            
            assert result is False
            mock_chroma.from_documents.assert_not_called()
    
    @patch('src.rag.vectorstore_manager.Chroma')
    def test_load_existing_success(self, mock_chroma):
        """Test loading existing vectorstore."""
        # Mock Chroma persistence
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        # Create a fake vectorstore directory
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)
        (self.vectorstore_path / "chroma.sqlite3").touch()
        
        result = self.manager.load_existing()
        
        assert result is True
        assert self.manager.vectorstore == mock_vectorstore
        mock_chroma.assert_called_once_with(
            persist_directory=str(self.vectorstore_path),
            embedding_function=get_embeddings()
        )
    
    def test_load_existing_not_found(self):
        """Test loading non-existent vectorstore."""
        result = self.manager.load_existing()
        
        assert result is False
        assert self.manager.vectorstore is None
    
    def test_get_retriever_no_vectorstore(self):
        """Test getting retriever when no vectorstore is loaded."""
        retriever = self.manager.get_retriever()
        
        assert retriever is None
    
    def test_get_retriever_success(self):
        """Test getting retriever from loaded vectorstore."""
        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        self.manager.vectorstore = mock_vectorstore
        
        retriever = self.manager.get_retriever(k=5)
        
        assert retriever == mock_retriever
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
    
    def test_query_simple_no_vectorstore(self):
        """Test simple query when no vectorstore is loaded."""
        docs = self.manager.query_simple("test query")
        
        assert docs == []
    
    def test_query_simple_success(self):
        """Test successful simple query."""
        # Mock vectorstore and retriever
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_docs = [Mock(), Mock()]
        mock_retriever.get_relevant_documents.return_value = mock_docs
        mock_vectorstore.as_retriever.return_value = mock_retriever
        self.manager.vectorstore = mock_vectorstore
        
        docs = self.manager.query_simple("test query", k=3)
        
        assert docs == mock_docs
        mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        mock_retriever.get_relevant_documents.assert_called_once_with("test query")
    
    def test_save_vectorstore_no_vectorstore(self):
        """Test saving when no vectorstore is loaded."""
        result = self.manager.save_vectorstore()
        
        assert result is False
    
    @patch('src.rag.vectorstore_manager.Chroma')
    def test_save_vectorstore_success(self, mock_chroma):
        """Test successful vectorstore saving."""
        # Mock vectorstore
        mock_vectorstore = Mock()
        self.manager.vectorstore = mock_vectorstore
        
        result = self.manager.save_vectorstore()
        
        assert result is True
        mock_vectorstore.persist.assert_called_once()
    
    def test_is_loaded_false(self):
        """Test is_loaded when no vectorstore is loaded."""
        assert self.manager.is_loaded() is False
    
    def test_is_loaded_true(self):
        """Test is_loaded when vectorstore is loaded."""
        self.manager.vectorstore = Mock()
        assert self.manager.is_loaded() is True


if __name__ == "__main__":
    pytest.main([__file__])
