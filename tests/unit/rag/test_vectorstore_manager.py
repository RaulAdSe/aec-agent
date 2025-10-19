"""
Unit tests for vectorstore manager.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from src.rag.vectorstore_manager import VectorstoreManager


class TestVectorstoreManager:
    """Test cases for vectorstore manager functionality."""
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_init(self, mock_get_embeddings, tmp_path):
        """Test VectorstoreManager initialization."""
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        
        # Verify initialization
        assert manager.persist_directory == persist_dir
        assert manager.embeddings == mock_embeddings
        assert manager.vectorstore is None
        assert persist_dir.exists()  # Directory should be created
        
        mock_get_embeddings.assert_called_once_with(None)
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_init_with_custom_embeddings(self, mock_get_embeddings, tmp_path):
        """Test initialization with custom embeddings model."""
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        persist_dir = tmp_path / "vectorstore"
        custom_model = "custom-model"
        manager = VectorstoreManager(persist_dir, custom_model)
        
        mock_get_embeddings.assert_called_once_with(custom_model)
    
    @patch('src.rag.vectorstore_manager.load_pdfs')
    @patch('src.rag.vectorstore_manager.RecursiveCharacterTextSplitter')
    @patch('src.rag.vectorstore_manager.Chroma')
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_create_from_pdfs(
        self, 
        mock_get_embeddings, 
        mock_chroma_class, 
        mock_splitter_class, 
        mock_load_pdfs,
        tmp_path
    ):
        """Test creating vectorstore from PDFs."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_documents = [
            Document(page_content="Doc 1 content", metadata={"source": "doc1.pdf"}),
            Document(page_content="Doc 2 content", metadata={"source": "doc2.pdf"})
        ]
        mock_load_pdfs.return_value = mock_documents
        
        mock_splitter = Mock()
        mock_splitter_class.return_value = mock_splitter
        mock_chunks = [
            Document(page_content="Chunk 1", metadata={"source": "doc1.pdf"}),
            Document(page_content="Chunk 2", metadata={"source": "doc2.pdf"})
        ]
        mock_splitter.split_documents.return_value = mock_chunks
        
        mock_vectorstore = Mock()
        mock_chroma_class.from_documents.return_value = mock_vectorstore
        
        # Test
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        
        manager.create_from_pdfs(pdf_dir)
        
        # Verify calls
        mock_load_pdfs.assert_called_once_with(pdf_dir)
        mock_splitter_class.assert_called_once()
        mock_splitter.split_documents.assert_called_once_with(mock_documents)
        mock_chroma_class.from_documents.assert_called_once_with(
            documents=mock_chunks,
            embedding=mock_embeddings,
            persist_directory=str(persist_dir)
        )
        
        assert manager.vectorstore == mock_vectorstore
    
    @patch('src.rag.vectorstore_manager.Chroma')
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_load_existing_success(self, mock_get_embeddings, mock_chroma_class, tmp_path):
        """Test loading existing vectorstore."""
        # Setup
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        persist_dir = tmp_path / "vectorstore"
        persist_dir.mkdir()
        
        mock_vectorstore = Mock()
        mock_chroma_class.return_value = mock_vectorstore
        
        # Test
        manager = VectorstoreManager(persist_dir)
        manager.load_existing()
        
        # Verify
        mock_chroma_class.assert_called_once_with(
            persist_directory=str(persist_dir),
            embedding_function=mock_embeddings
        )
        assert manager.vectorstore == mock_vectorstore
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_load_existing_directory_not_found(self, mock_get_embeddings, tmp_path):
        """Test loading non-existent vectorstore."""
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        persist_dir = tmp_path / "nonexistent"
        manager = VectorstoreManager(persist_dir)
        
        with pytest.raises(ValueError, match="does not exist"):
            manager.load_existing()
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_get_retriever_not_initialized(self, mock_get_embeddings, tmp_path):
        """Test getting retriever when vectorstore not initialized."""
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        
        with pytest.raises(ValueError, match="not initialized"):
            manager.get_retriever()
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_get_retriever_similarity(self, mock_get_embeddings, tmp_path):
        """Test getting similarity retriever."""
        # Setup
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        manager.vectorstore = mock_vectorstore
        
        # Test
        result = manager.get_retriever(k=5)
        
        # Verify
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        assert result == mock_retriever
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_get_retriever_mmr(self, mock_get_embeddings, tmp_path):
        """Test getting MMR retriever."""
        # Setup
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        manager.vectorstore = mock_vectorstore
        
        # Test
        result = manager.get_retriever(k=3, search_type="mmr")
        
        # Verify
        expected_kwargs = {
            "k": 3,
            "fetch_k": 9,  # k * 3
            "lambda_mult": 0.5
        }
        mock_vectorstore.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs=expected_kwargs
        )
        assert result == mock_retriever
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_similarity_search_not_initialized(self, mock_get_embeddings, tmp_path):
        """Test similarity search when vectorstore not initialized."""
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        
        with pytest.raises(ValueError, match="not initialized"):
            manager.similarity_search("test query")
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_similarity_search_success(self, mock_get_embeddings, tmp_path):
        """Test successful similarity search."""
        # Setup
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectorstore = Mock()
        mock_docs = [Document(page_content="Result 1"), Document(page_content="Result 2")]
        mock_vectorstore.similarity_search.return_value = mock_docs
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        manager.vectorstore = mock_vectorstore
        
        # Test
        result = manager.similarity_search("test query", k=2)
        
        # Verify
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=2)
        assert result == mock_docs
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_get_vectorstore_info_not_initialized(self, mock_get_embeddings, tmp_path):
        """Test getting info when vectorstore not initialized."""
        mock_embeddings = Mock()
        mock_get_embeddings.return_value = mock_embeddings
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        
        result = manager.get_vectorstore_info()
        
        assert result == {"status": "not_initialized"}
    
    @patch('src.rag.vectorstore_manager.get_embeddings')
    def test_get_vectorstore_info_success(self, mock_get_embeddings, tmp_path):
        """Test getting vectorstore info successfully."""
        # Setup
        mock_embeddings = Mock()
        mock_embeddings.model_name = "test-model"
        mock_get_embeddings.return_value = mock_embeddings
        
        mock_vectorstore = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_vectorstore._collection = mock_collection
        
        persist_dir = tmp_path / "vectorstore"
        manager = VectorstoreManager(persist_dir)
        manager.vectorstore = mock_vectorstore
        
        # Test
        result = manager.get_vectorstore_info()
        
        # Verify
        expected = {
            "status": "initialized",
            "persist_directory": str(persist_dir),
            "document_count": 100,
            "embedding_model": "test-model"
        }
        assert result == expected
