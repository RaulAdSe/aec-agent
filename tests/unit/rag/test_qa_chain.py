"""
Unit tests for QA chain functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.rag.qa_chain import create_qa_chain, create_llm


class TestQAChain:
    """Test cases for QA chain functionality."""
    
    def test_create_llm_default(self):
        """Test creating LLM with default parameters."""
        with patch('src.rag.qa_chain.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            llm = create_llm()
            
            assert llm == mock_llm
            mock_openai.assert_called_once_with(
                model="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=1000
            )
    
    def test_create_llm_custom_params(self):
        """Test creating LLM with custom parameters."""
        with patch('src.rag.qa_chain.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            llm = create_llm(
                model="gpt-4",
                temperature=0.5,
                max_tokens=2000
            )
            
            assert llm == mock_llm
            mock_openai.assert_called_once_with(
                model="gpt-4",
                temperature=0.5,
                max_tokens=2000
            )
    
    def test_create_llm_environment_variables(self):
        """Test creating LLM with environment variables."""
        with patch('src.rag.qa_chain.ChatOpenAI') as mock_openai, \
             patch('os.getenv') as mock_getenv:
            
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            # Mock environment variables
            mock_getenv.side_effect = lambda key, default=None: {
                "OPENAI_MODEL": "gpt-4",
                "OPENAI_TEMPERATURE": "0.3",
                "OPENAI_MAX_TOKENS": "1500"
            }.get(key, default)
            
            llm = create_llm()
            
            mock_openai.assert_called_once_with(
                model="gpt-4",
                temperature=0.3,
                max_tokens=1500
            )
    
    @patch('src.rag.qa_chain.create_llm')
    @patch('src.rag.qa_chain.RetrievalQA')
    def test_create_qa_chain_success(self, mock_retrieval_qa, mock_create_llm):
        """Test successful QA chain creation."""
        # Mock retriever and LLM
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        # Mock QA chain
        mock_qa_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        
        # Test
        qa_chain = create_qa_chain(mock_retriever)
        
        # Assertions
        assert qa_chain == mock_qa_chain
        mock_create_llm.assert_called_once_with()
        mock_retrieval_qa.from_chain_type.assert_called_once_with(
            llm=mock_llm,
            chain_type="stuff",
            retriever=mock_retriever,
            return_source_documents=True
        )
    
    @patch('src.rag.qa_chain.create_llm')
    @patch('src.rag.qa_chain.RetrievalQA')
    def test_create_qa_chain_custom_llm(self, mock_retrieval_qa, mock_create_llm):
        """Test QA chain creation with custom LLM."""
        # Mock retriever and LLM
        mock_retriever = Mock()
        mock_llm = Mock()
        
        # Mock QA chain
        mock_qa_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        
        # Test with custom LLM
        qa_chain = create_qa_chain(mock_retriever, llm=mock_llm)
        
        # Assertions
        assert qa_chain == mock_qa_chain
        mock_create_llm.assert_not_called()  # Should not create new LLM
        mock_retrieval_qa.from_chain_type.assert_called_once_with(
            llm=mock_llm,
            chain_type="stuff",
            retriever=mock_retriever,
            return_source_documents=True
        )
    
    @patch('src.rag.qa_chain.create_llm')
    @patch('src.rag.qa_chain.RetrievalQA')
    def test_create_qa_chain_custom_params(self, mock_retrieval_qa, mock_create_llm):
        """Test QA chain creation with custom parameters."""
        # Mock retriever and LLM
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        # Mock QA chain
        mock_qa_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        
        # Test with custom parameters
        qa_chain = create_qa_chain(
            mock_retriever,
            chain_type="map_reduce",
            temperature=0.5,
            max_tokens=2000
        )
        
        # Assertions
        assert qa_chain == mock_qa_chain
        mock_create_llm.assert_called_once_with(
            temperature=0.5,
            max_tokens=2000
        )
        mock_retrieval_qa.from_chain_type.assert_called_once_with(
            llm=mock_llm,
            chain_type="map_reduce",
            retriever=mock_retriever,
            return_source_documents=True
        )
    
    @patch('src.rag.qa_chain.create_llm')
    @patch('src.rag.qa_chain.RetrievalQA')
    def test_create_qa_chain_with_prompt(self, mock_retrieval_qa, mock_create_llm):
        """Test QA chain creation with custom prompt."""
        # Mock retriever and LLM
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        # Mock QA chain
        mock_qa_chain = Mock()
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        
        # Custom prompt
        custom_prompt = "Answer the question based on the context: {context}\nQuestion: {question}"
        
        # Test
        qa_chain = create_qa_chain(mock_retriever, prompt_template=custom_prompt)
        
        # Assertions
        assert qa_chain == mock_qa_chain
        mock_retrieval_qa.from_chain_type.assert_called_once()
        call_args = mock_retrieval_qa.from_chain_type.call_args
        
        # Check that prompt was passed correctly
        assert "chain_type_kwargs" in call_args.kwargs
        assert "prompt" in call_args.kwargs["chain_type_kwargs"]
    
    def test_create_qa_chain_no_retriever(self):
        """Test QA chain creation without retriever."""
        with pytest.raises(ValueError, match="Retriever is required"):
            create_qa_chain(None)
    
    @patch('src.rag.qa_chain.create_llm')
    @patch('src.rag.qa_chain.RetrievalQA')
    def test_qa_chain_execution(self, mock_retrieval_qa, mock_create_llm):
        """Test QA chain execution."""
        # Mock retriever and LLM
        mock_retriever = Mock()
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        # Mock QA chain
        mock_qa_chain = Mock()
        mock_result = {
            "result": "Test answer",
            "source_documents": [Mock(), Mock()]
        }
        mock_qa_chain.return_value = mock_result
        mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain
        
        # Create QA chain
        qa_chain = create_qa_chain(mock_retriever)
        
        # Test execution
        query = "What is the test question?"
        result = qa_chain({"query": query})
        
        # Assertions
        assert result == mock_result
        mock_qa_chain.assert_called_once_with({"query": query})
    
    @patch('src.rag.qa_chain.create_llm')
    def test_create_llm_api_key_validation(self, mock_create_llm):
        """Test LLM creation with API key validation."""
        with patch('os.getenv', return_value=None):
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                create_llm()
    
    @patch('src.rag.qa_chain.ChatOpenAI')
    def test_create_llm_with_api_key(self, mock_openai):
        """Test LLM creation with valid API key."""
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        with patch('os.getenv', return_value="test-api-key"):
            llm = create_llm()
            
            assert llm == mock_llm
            mock_openai.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
