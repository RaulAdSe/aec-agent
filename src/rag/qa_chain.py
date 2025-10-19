"""
QA Chain implementation for RAG system.

Uses OpenAI LLM with retrieval-augmented generation for building code queries.
"""

import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document


# System prompt optimized for Spanish building codes
SYSTEM_PROMPT = """
Eres un asistente experto en normativa de construcción española, especializado en el Código Técnico de la Edificación (CTE), especialmente en el Documento Básico de Seguridad en caso de Incendio (DB-SI).

Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado de la normativa.

REGLAS IMPORTANTES:
1. Responde SIEMPRE en español
2. Si la información está en el contexto, responde de forma clara y precisa
3. SIEMPRE cita la fuente específica (documento, sección, página si está disponible)
4. Si no puedes responder con el contexto dado, di claramente "No tengo información suficiente sobre esto en la normativa proporcionada"
5. NO inventes ni supongas información que no esté en el contexto
6. Si hay múltiples requisitos, enuméralos claramente
7. Incluye valores numéricos específicos cuando estén disponibles

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA DETALLADA:
"""


def create_qa_chain(
    retriever, 
    model_name: str = "gpt-3.5-turbo", 
    temperature: float = 0.1,
    max_tokens: int = 1000
) -> RetrievalQA:
    """
    Create QA chain with OpenAI LLM and retrieval.
    
    Args:
        retriever: Document retriever from vectorstore
        model_name: OpenAI model name
        temperature: LLM temperature (0.1 for precise answers)
        max_tokens: Maximum tokens in response
        
    Returns:
        Configured RetrievalQA chain
        
    Raises:
        ValueError: If OpenAI API key is not set
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it with your OpenAI API key."
        )
    
    # Create OpenAI LLM with model-specific parameters
    if "gpt-5" in model_name:
        # GPT-5 models have different parameter requirements
        llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
            # No temperature or max_tokens for gpt-5 models
        )
    elif "gpt-4" in model_name:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Create prompt template
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Put all retrieved docs in context
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def query(qa_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    Query the QA chain with a question.
    
    Args:
        qa_chain: Configured QA chain
        question: Question to ask
        
    Returns:
        Dictionary with 'result' and 'source_documents'
    """
    result = qa_chain({"query": question})
    return result


def format_response(result: Dict[str, Any]) -> str:
    """
    Format the QA response for better readability.
    
    Args:
        result: Result from QA chain
        
    Returns:
        Formatted response string
    """
    response = result["result"]
    sources = result["source_documents"]
    
    # Add source information
    if sources:
        response += "\n\n📚 **Fuentes consultadas:**\n"
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "Desconocido")
            page = doc.metadata.get("page", "N/A")
            response += f"{i}. {source}, Página {page}\n"
    
    return response


def batch_query(qa_chain: RetrievalQA, questions: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple questions in batch.
    
    Args:
        qa_chain: Configured QA chain
        questions: List of questions
        
    Returns:
        List of results for each question
    """
    results = []
    for question in questions:
        result = query(qa_chain, question)
        results.append({
            "question": question,
            "answer": result["result"],
            "sources": result["source_documents"]
        })
    return results


def get_retrieval_info(qa_chain: RetrievalQA, question: str) -> Dict[str, Any]:
    """
    Get detailed information about retrieval process.
    
    Args:
        qa_chain: Configured QA chain
        question: Question to analyze
        
    Returns:
        Dictionary with retrieval details
    """
    # Get retrieved documents
    docs = qa_chain.retriever.get_relevant_documents(question)
    
    return {
        "question": question,
        "retrieved_docs_count": len(docs),
        "retrieved_docs": [
            {
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
    }
