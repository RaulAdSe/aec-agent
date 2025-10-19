"""
Unit tests for QA chain.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag.qa_chain import (
    create_qa_chain, 
    query, 
    format_response, 
    batch_query, 
    get_retrieval_info
)


class TestQAChain:
    """Test cases for QA chain functionality."""
    
    def test_create_qa_chain_no_api_key(self):
        """Test error when OpenAI API key is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable is required"):
                create_qa_chain(Mock())
    
    @patch('src.rag.qa_chain.ChatOpenAI')
    @patch('src.rag.qa_chain.PromptTemplate')
    @patch('src.rag.qa_chain.RetrievalQA')
    def test_create_qa_chain_success(self, mock_qa_class, mock_prompt_class, mock_llm_class):
        """Test successful QA chain creation."""
        # Setup mocks
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_prompt = Mock()
        mock_prompt_class.return_value = mock_prompt
        mock_qa = Mock()
        mock_qa_class.from_chain_type.return_value = mock_qa
        
        # Test
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = create_qa_chain(mock_retriever)
        
        # Verify
        mock_llm_class.assert_called_once_with(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key='test-key'
        )
        
        mock_prompt_class.assert_called_once()
        mock_qa_class.from_chain_type.assert_called_once_with(
            llm=mock_llm,
            chain_type="stuff",
            retriever=mock_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": mock_prompt}
        )
        
        assert result == mock_qa
    
    @patch('src.rag.qa_chain.ChatOpenAI')
    @patch('src.rag.qa_chain.PromptTemplate')
    @patch('src.rag.qa_chain.RetrievalQA')
    def test_create_qa_chain_custom_params(self, mock_qa_class, mock_prompt_class, mock_llm_class):
        """Test QA chain creation with custom parameters."""
        # Setup mocks
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_prompt = Mock()
        mock_prompt_class.return_value = mock_prompt
        mock_qa = Mock()
        mock_qa_class.from_chain_type.return_value = mock_qa
        
        # Test with custom parameters
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = create_qa_chain(
                mock_retriever,
                model_name="gpt-4",
                temperature=0.5,
                max_tokens=2000
            )
        
        # Verify custom parameters were used
        mock_llm_class.assert_called_once_with(
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=2000,
            openai_api_key='test-key'
        )
        
        assert result == mock_qa
    
    def test_query(self):
        """Test querying the QA chain."""
        # Setup mock QA chain
        mock_qa_chain = Mock()
        expected_result = {
            "result": "Test answer",
            "source_documents": [Document(page_content="Source 1")]
        }
        mock_qa_chain.return_value = expected_result
        
        # Test
        result = query(mock_qa_chain, "Test question")
        
        # Verify
        mock_qa_chain.assert_called_once_with({"query": "Test question"})
        assert result == expected_result
    
    def test_format_response(self):
        """Test formatting QA response."""
        # Setup test data
        result = {
            "result": "Test answer",
            "source_documents": [
                Document(
                    page_content="Source content 1",
                    metadata={"source": "doc1.pdf", "page": 1}
                ),
                Document(
                    page_content="Source content 2", 
                    metadata={"source": "doc2.pdf", "page": 5}
                )
            ]
        }
        
        # Test
        formatted = format_response(result)
        
        # Verify
        assert "Test answer" in formatted
        assert "📚 **Fuentes consultadas:**" in formatted
        assert "1. doc1.pdf, Página 1" in formatted
        assert "2. doc2.pdf, Página 5" in formatted
    
    def test_format_response_no_sources(self):
        """Test formatting response with no sources."""
        result = {
            "result": "Test answer",
            "source_documents": []
        }
        
        formatted = format_response(result)
        
        assert formatted == "Test answer"
    
    def test_batch_query(self):
        """Test batch querying."""
        # Setup mock QA chain
        mock_qa_chain = Mock()
        mock_qa_chain.side_effect = [
            {"result": "Answer 1", "source_documents": []},
            {"result": "Answer 2", "source_documents": []},
            {"result": "Answer 3", "source_documents": []}
        ]
        
        questions = ["Question 1", "Question 2", "Question 3"]
        
        # Test
        results = batch_query(mock_qa_chain, questions)
        
        # Verify
        assert len(results) == 3
        assert results[0]["question"] == "Question 1"
        assert results[0]["answer"] == "Answer 1"
        assert results[1]["question"] == "Question 2"
        assert results[1]["answer"] == "Answer 2"
        assert results[2]["question"] == "Question 3"
        assert results[2]["answer"] == "Answer 3"
        
        # Verify QA chain was called for each question
        assert mock_qa_chain.call_count == 3
    
    def test_get_retrieval_info(self):
        """Test getting retrieval information."""
        # Setup mock retriever
        mock_retriever = Mock()
        mock_docs = [
            Document(
                page_content="This is a long document content that should be truncated for preview...",
                metadata={"source": "doc1.pdf", "page": 1}
            ),
            Document(
                page_content="Short content",
                metadata={"source": "doc2.pdf", "page": 2}
            )
        ]
        mock_retriever.get_relevant_documents.return_value = mock_docs
        
        # Setup mock QA chain
        mock_qa_chain = Mock()
        mock_qa_chain.retriever = mock_retriever
        
        # Test
        result = get_retrieval_info(mock_qa_chain, "Test question")
        
        # Verify
        assert result["question"] == "Test question"
        assert result["retrieved_docs_count"] == 2
        assert len(result["retrieved_docs"]) == 2
        
        # Check first doc (should be truncated)
        first_doc = result["retrieved_docs"][0]
        assert "This is a long document content that should be truncated for preview..." in first_doc["content_preview"]
        assert first_doc["metadata"]["source"] == "doc1.pdf"
        
        # Check second doc (should not be truncated)
        second_doc = result["retrieved_docs"][1]
        assert second_doc["content_preview"] == "Short content"
        assert second_doc["metadata"]["source"] == "doc2.pdf"
        
        mock_retriever.get_relevant_documents.assert_called_once_with("Test question")
