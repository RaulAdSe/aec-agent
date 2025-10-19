"""
Unit tests for DocumentLoader class.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.rag.document_loader import DocumentLoader


class TestDocumentLoader:
    """Test cases for DocumentLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DocumentLoader()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_default_params(self):
        """Test DocumentLoader initialization with default parameters."""
        loader = DocumentLoader()
        
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200
        assert loader.embeddings_model is None
    
    def test_init_custom_params(self):
        """Test DocumentLoader initialization with custom parameters."""
        loader = DocumentLoader(
            chunk_size=500,
            chunk_overlap=100,
            embeddings_model="test-model"
        )
        
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 100
        assert loader.embeddings_model == "test-model"
    
    def test_load_pdfs_no_directory(self):
        """Test loading PDFs from non-existent directory."""
        non_existent_dir = Path("non_existent")
        
        with pytest.raises(FileNotFoundError):
            self.loader.load_pdfs(non_existent_dir)
    
    def test_load_pdfs_empty_directory(self):
        """Test loading PDFs from empty directory."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        docs = self.loader.load_pdfs(empty_dir)
        
        assert docs == []
    
    def test_load_pdfs_no_pdf_files(self):
        """Test loading from directory with no PDF files."""
        no_pdf_dir = Path(self.temp_dir) / "no_pdfs"
        no_pdf_dir.mkdir()
        
        # Create non-PDF files
        (no_pdf_dir / "test.txt").touch()
        (no_pdf_dir / "test.docx").touch()
        
        docs = self.loader.load_pdfs(no_pdf_dir)
        
        assert docs == []
    
    @patch('src.rag.document_loader.PyPDFLoader')
    def test_load_pdfs_single_pdf(self, mock_pypdf_loader):
        """Test loading a single PDF file."""
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        # Create a fake PDF file
        pdf_file = pdf_dir / "test.pdf"
        pdf_file.touch()
        
        # Mock PyPDFLoader
        mock_loader_instance = Mock()
        mock_docs = [Mock(), Mock()]
        mock_loader_instance.load.return_value = mock_docs
        mock_pypdf_loader.return_value = mock_loader_instance
        
        docs = self.loader.load_pdfs(pdf_dir)
        
        assert len(docs) == 2
        mock_pypdf_loader.assert_called_once_with(str(pdf_file))
        mock_loader_instance.load.assert_called_once()
    
    @patch('src.rag.document_loader.PyPDFLoader')
    def test_load_pdfs_multiple_pdfs(self, mock_pypdf_loader):
        """Test loading multiple PDF files."""
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        # Create multiple PDF files
        (pdf_dir / "test1.pdf").touch()
        (pdf_dir / "test2.pdf").touch()
        (pdf_dir / "test3.pdf").touch()
        
        # Mock PyPDFLoader
        mock_loader_instance = Mock()
        mock_docs = [Mock(), Mock()]
        mock_loader_instance.load.return_value = mock_docs
        mock_pypdf_loader.return_value = mock_loader_instance
        
        docs = self.loader.load_pdfs(pdf_dir)
        
        # Should have 6 documents (2 per PDF * 3 PDFs)
        assert len(docs) == 6
        assert mock_pypdf_loader.call_count == 3
    
    @patch('src.rag.document_loader.PyPDFLoader')
    def test_load_pdfs_with_metadata(self, mock_pypdf_loader):
        """Test loading PDFs with metadata."""
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        pdf_file = pdf_dir / "test.pdf"
        pdf_file.touch()
        
        # Mock PyPDFLoader
        mock_loader_instance = Mock()
        mock_doc = Mock()
        mock_doc.metadata = {"source": str(pdf_file)}
        mock_loader_instance.load.return_value = [mock_doc]
        mock_pypdf_loader.return_value = mock_loader_instance
        
        docs = self.loader.load_pdfs(pdf_dir)
        
        assert len(docs) == 1
        assert docs[0].metadata["source"] == str(pdf_file)
    
    @patch('src.rag.document_loader.PyPDFLoader')
    def test_load_pdfs_error_handling(self, mock_pypdf_loader):
        """Test error handling when loading PDFs."""
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        pdf_file = pdf_dir / "corrupted.pdf"
        pdf_file.touch()
        
        # Mock PyPDFLoader to raise an exception
        mock_pypdf_loader.side_effect = Exception("PDF loading error")
        
        # Should not raise exception, but return empty list
        docs = self.loader.load_pdfs(pdf_dir)
        
        assert docs == []
    
    def test_chunk_documents_no_documents(self):
        """Test chunking when no documents are provided."""
        chunks = self.loader.chunk_documents([])
        
        assert chunks == []
    
    @patch('src.rag.document_loader.RecursiveCharacterTextSplitter')
    def test_chunk_documents_success(self, mock_splitter):
        """Test successful document chunking."""
        # Mock documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "This is a test document."
        mock_doc2 = Mock()
        mock_doc2.page_content = "Another test document."
        documents = [mock_doc1, mock_doc2]
        
        # Mock text splitter
        mock_splitter_instance = Mock()
        mock_chunks = [Mock(), Mock(), Mock()]
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance
        
        chunks = self.loader.chunk_documents(documents)
        
        assert chunks == mock_chunks
        mock_splitter.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=200
        )
        mock_splitter_instance.split_documents.assert_called_once_with(documents)
    
    def test_get_pdf_files_no_directory(self):
        """Test getting PDF files from non-existent directory."""
        non_existent_dir = Path("non_existent")
        
        pdf_files = self.loader._get_pdf_files(non_existent_dir)
        
        assert pdf_files == []
    
    def test_get_pdf_files_success(self):
        """Test getting PDF files from directory."""
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        # Create various files
        (pdf_dir / "test1.pdf").touch()
        (pdf_dir / "test2.PDF").touch()  # Different case
        (pdf_dir / "test3.txt").touch()  # Not a PDF
        (pdf_dir / "subdir").mkdir()
        (pdf_dir / "subdir" / "test4.pdf").touch()  # In subdirectory
        
        pdf_files = self.loader._get_pdf_files(pdf_dir)
        
        # Should find 3 PDF files (including subdirectory)
        assert len(pdf_files) == 3
        assert all(f.suffix.lower() == '.pdf' for f in pdf_files)
    
    def test_validate_pdf_file_valid(self):
        """Test validation of valid PDF file."""
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        pdf_file = pdf_dir / "test.pdf"
        pdf_file.touch()
        
        is_valid = self.loader._validate_pdf_file(pdf_file)
        
        assert is_valid is True
    
    def test_validate_pdf_file_invalid_extension(self):
        """Test validation of file with invalid extension."""
        pdf_dir = Path(self.temp_dir) / "pdfs"
        pdf_dir.mkdir()
        
        txt_file = pdf_dir / "test.txt"
        txt_file.touch()
        
        is_valid = self.loader._validate_pdf_file(txt_file)
        
        assert is_valid is False
    
    def test_validate_pdf_file_nonexistent(self):
        """Test validation of non-existent file."""
        non_existent_file = Path("non_existent.pdf")
        
        is_valid = self.loader._validate_pdf_file(non_existent_file)
        
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__])
