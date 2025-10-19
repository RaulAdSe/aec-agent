"""
Unit tests for document loader.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from langchain.schema import Document

from src.rag.document_loader import load_pdfs, load_single_pdf


class TestDocumentLoader:
    """Test cases for document loader functionality."""
    
    def test_load_single_pdf_file_not_found(self):
        """Test error handling for non-existent PDF file."""
        with pytest.raises(FileNotFoundError):
            load_single_pdf(Path("nonexistent.pdf"))
    
    def test_load_single_pdf_invalid_extension(self):
        """Test error handling for non-PDF file."""
        with pytest.raises(ValueError, match="is not a PDF"):
            load_single_pdf(Path("test.txt"))
    
    @patch('src.rag.document_loader.PyPDFLoader')
    def test_load_single_pdf_success(self, mock_loader_class):
        """Test successful loading of single PDF."""
        # Mock the loader and its load method
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Mock documents
        mock_docs = [
            Document(page_content="Page 1 content", metadata={"page": 0}),
            Document(page_content="Page 2 content", metadata={"page": 1})
        ]
        mock_loader.load.return_value = mock_docs
        
        # Test loading
        result = load_single_pdf(Path("test.pdf"))
        
        # Verify
        assert len(result) == 2
        assert result[0].page_content == "Page 1 content"
        assert result[0].metadata["source"] == "test.pdf"
        assert result[0].metadata["source_path"] == str(Path("test.pdf"))
        
        mock_loader_class.assert_called_once_with(str(Path("test.pdf")))
        mock_loader.load.assert_called_once()
    
    def test_load_pdfs_directory_not_found(self):
        """Test error handling for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            load_pdfs(Path("nonexistent_dir"))
    
    def test_load_pdfs_no_pdfs_found(self, tmp_path):
        """Test error handling when no PDFs found in directory."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="No PDF files found"):
            load_pdfs(empty_dir)
    
    @patch('src.rag.document_loader.PyPDFLoader')
    def test_load_pdfs_success(self, mock_loader_class, tmp_path):
        """Test successful loading of multiple PDFs."""
        # Create test PDF files
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.touch()
        pdf2.touch()
        
        # Mock the loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Mock documents for each PDF
        mock_docs1 = [Document(page_content="Doc1 content", metadata={"page": 0})]
        mock_docs2 = [Document(page_content="Doc2 content", metadata={"page": 0})]
        
        # Configure mock to return different docs for different files
        def mock_load(file_path):
            if "doc1.pdf" in file_path:
                return mock_docs1
            elif "doc2.pdf" in file_path:
                return mock_docs2
            return []
        
        mock_loader.load.side_effect = mock_load
        
        # Test loading
        result = load_pdfs(tmp_path)
        
        # Verify
        assert len(result) == 2
        assert result[0].page_content == "Doc1 content"
        assert result[0].metadata["source"] == "doc1.pdf"
        assert result[1].page_content == "Doc2 content"
        assert result[1].metadata["source"] == "doc2.pdf"
        
        # Verify loader was called for both files
        assert mock_loader_class.call_count == 2
    
    @patch('src.rag.document_loader.PyPDFLoader')
    def test_load_pdfs_with_errors(self, mock_loader_class, tmp_path):
        """Test loading PDFs with some files causing errors."""
        # Create test PDF files
        pdf1 = tmp_path / "good.pdf"
        pdf2 = tmp_path / "bad.pdf"
        pdf1.touch()
        pdf2.touch()
        
        # Mock the loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        
        # Configure mock to raise error for bad.pdf
        def mock_load(file_path):
            if "bad.pdf" in file_path:
                raise Exception("Corrupted PDF")
            return [Document(page_content="Good content", metadata={"page": 0})]
        
        mock_loader.load.side_effect = mock_load
        
        # Test loading - should not raise exception
        result = load_pdfs(tmp_path)
        
        # Should only have one document (from good.pdf)
        assert len(result) == 1
        assert result[0].page_content == "Good content"
        assert result[0].metadata["source"] == "good.pdf"
