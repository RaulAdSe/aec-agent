"""
QA chain for RAG system.

This module creates the question-answering chain for building code queries.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = """
Eres un asistente experto en normativa de construcción española (CTE).
Responde preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Si la información está en el contexto, responde de forma clara y precisa
2. Siempre cita la fuente (documento, sección, página)
3. Si no puedes responder con el contexto dado, di "No tengo información suficiente"
4. No inventes ni supongas información

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""

def create_qa_chain(retriever, model_name: str = "gemini-pro", temperature: float = 0.1):
    """
    Create QA chain with retrieval.
    
    Args:
        retriever: Vectorstore retriever
        model_name: Google model name
        temperature: LLM temperature
    
    Returns:
        Configured QA chain
    """
    # LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature
    )
    
    # Prompt
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
    
    # Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain
