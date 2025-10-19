# 🔍 RAG (Retrieval Augmented Generation) - Explicación Técnica

## 📋 Tabla de Contenidos

1. [¿Qué es RAG?](#qué-es-rag)
2. [El Problema que Resuelve](#el-problema-que-resuelve)
3. [Arquitectura de RAG](#arquitectura-de-rag)
4. [Componentes Clave](#componentes-clave)
5. [Pipeline Completo](#pipeline-completo)
6. [Implementación con LangChain](#implementación-con-langchain)
7. [Embeddings](#embeddings)
8. [Vectorstores](#vectorstores)
9. [Retrieval Strategies](#retrieval-strategies)
10. [Best Practices](#best-practices)
11. [Debugging & Optimization](#debugging--optimization)

---

## ¿Qué es RAG?

**RAG = Retrieval Augmented Generation**

Framework que permite a los LLMs acceder a información externa actualizada mediante:
- 📚 **Retrieval**: Buscar documentos relevantes
- 🧠 **Augmentation**: Añadir contexto al prompt
- ✍️ **Generation**: Generar respuesta informada

### Analogía Simple

Imagina que eres un estudiante haciendo un examen:
- **Sin RAG**: Solo puedes usar lo que memorizaste
- **Con RAG**: Puedes consultar tus apuntes durante el examen

---

## El Problema que Resuelve

### LLMs Tradicionales: Limitaciones

#### Problema 1: Conocimiento Estático

```python
# LLM sin RAG
pregunta = "¿Qué dice el CTE DB-SI actualizado en 2023 sobre puertas de evacuación?"
respuesta = llm(pregunta)
# ❌ Responde con conocimiento antiguo o incorrecto
```

#### Problema 2: Alucinaciones

```python
# Sin RAG, el LLM puede inventar información
pregunta = "¿Cuál es el ancho mínimo de puerta según el CTE?"
respuesta = llm(pregunta)
# ❌ "El ancho mínimo es 75cm" (inventado)
```

#### Problema 3: Sin Citas

```python
# No puedes verificar la fuente
respuesta = "El ancho mínimo es 80cm"
# ❓ ¿De dónde sale este dato?
```

### RAG: La Solución

```python
# Con RAG
pregunta = "¿Qué dice el CTE DB-SI sobre puertas de evacuación?"

# 1. Buscar en la base de conocimiento
chunks_relevantes = vectorstore.similarity_search(pregunta)
# Encuentra: "CTE DB-SI, Sección 4.2: El ancho mínimo de puertas de evacuación..."

# 2. Añadir contexto al prompt
contexto = "\n".join(chunks_relevantes)
prompt = f"Contexto: {contexto}\n\nPregunta: {pregunta}"

# 3. Generar respuesta informada
respuesta = llm(prompt)
# ✅ "Según el CTE DB-SI Sección 4.2, el ancho mínimo es 80cm [Fuente: CTE_DB-SI.pdf, pág. 23]"
```

---

## Arquitectura de RAG

```
┌─────────────────────────────────────────────────────────────┐
│                     FASE 1: INDEXACIÓN                       │
│                    (Se hace una vez)                         │
└─────────────────────────────────────────────────────────────┘

    📄 Documentos (PDFs, TXT, etc.)
           │
           ▼
    ┌─────────────┐
    │   Loader    │  ← Carga documentos
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Splitter   │  ← Divide en chunks
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Embeddings  │  ← Convierte a vectores
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Vectorstore │  ← Almacena vectores
    └─────────────┘


┌─────────────────────────────────────────────────────────────┐
│                    FASE 2: CONSULTA                          │
│                  (Cada vez que preguntas)                    │
└─────────────────────────────────────────────────────────────┘

    💬 Pregunta del usuario
           │
           ▼
    ┌─────────────┐
    │ Embeddings  │  ← Convierte pregunta a vector
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Retriever  │  ← Busca chunks similares
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Prompt    │  ← Construye prompt con contexto
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │     LLM     │  ← Genera respuesta
    └──────┬──────┘
           │
           ▼
    ✅ Respuesta + Citas
```

---

## Componentes Clave

### 1. Document Loader

Carga documentos de diferentes fuentes:

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# PDF
loader = PyPDFLoader("CTE_DB-SI.pdf")
docs = loader.load()

# Texto plano
loader = TextLoader("normativa.txt")
docs = loader.load()
```

**Cada documento tiene**:
- `page_content`: El texto
- `metadata`: Info adicional (nombre archivo, página, etc.)

### 2. Text Splitter

Divide documentos en chunks manejables:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Tamaño de cada chunk (caracteres)
    chunk_overlap=200,      # Solapamiento entre chunks
    separators=["\n\n", "\n", " ", ""]  # Prioridad de separación
)

chunks = splitter.split_documents(docs)
```

**¿Por qué dividir?**
- Los LLMs tienen límite de tokens
- Chunks pequeños = retrieval más preciso
- Overlap = preservar contexto entre chunks

#### Ejemplo de Splitting

```
Documento original:
─────────────────────────────────────────────────
"El CTE DB-SI establece que las puertas de evacuación
deben tener un ancho mínimo de 80cm. En caso de edificios
de pública concurrencia, este ancho se incrementa a 1.20m.
Las puertas deben abrirse en el sentido de la evacuación..."
─────────────────────────────────────────────────

Después del splitting (chunk_size=100, overlap=20):

Chunk 1:
"El CTE DB-SI establece que las puertas de evacuación
deben tener un ancho mínimo de 80cm."

Chunk 2 (con overlap):
"ancho mínimo de 80cm. En caso de edificios de pública
concurrencia, este ancho se incrementa a 1.20m."

Chunk 3:
"ancho se incrementa a 1.20m. Las puertas deben abrirse
en el sentido de la evacuación..."
```

### 3. Embeddings

Convierte texto a vectores numéricos:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Convertir texto a vector
vector = embeddings.embed_query("¿Ancho mínimo de puerta?")
# Output: [0.12, -0.45, 0.89, ..., 0.34]  (384 dimensiones)
```

**Propiedad clave**: Textos similares → Vectores cercanos

```python
v1 = embeddings.embed_query("ancho de puerta")
v2 = embeddings.embed_query("anchura de entrada")
v3 = embeddings.embed_query("precio del tomate")

# Distancia coseno:
# similarity(v1, v2) = 0.89  ← Muy similar
# similarity(v1, v3) = 0.12  ← Poco similar
```

### 4. Vectorstore

Almacena y busca vectores eficientemente:

```python
from langchain_community.vectorstores import Chroma

# Crear vectorstore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectorstore"
)

# Buscar documentos similares
results = vectorstore.similarity_search(
    "¿Ancho mínimo de puerta?",
    k=3  # Top 3 resultados
)
```

**¿Cómo funciona?**
1. Convierte la pregunta a vector
2. Calcula distancia a todos los vectores almacenados
3. Devuelve los K más cercanos

### 5. Retriever

Interfaz para recuperar documentos:

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",  # O "mmr" para diversidad
    search_kwargs={"k": 3}
)

docs = retriever.get_relevant_documents("¿Ancho mínimo de puerta?")
```

### 6. LLM

Genera la respuesta final:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.1  # Bajo = más determinista
)
```

### 7. Prompt Template

Estructura el prompt con contexto:

```python
from langchain.prompts import ChatPromptTemplate

template = """
Eres un asistente experto en normativa de construcción.
Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.
Si no puedes responder con el contexto dado, di "No tengo información suficiente".

Contexto:
{context}

Pregunta: {question}

Respuesta:
"""

prompt = ChatPromptTemplate.from_template(template)
```

---

## Pipeline Completo

### Implementación Paso a Paso

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 1. Cargar documentos
loader = PyPDFLoader("CTE_DB-SI.pdf")
documents = loader.load()

# 2. Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Crear embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 4. Crear vectorstore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectorstore"
)

# 5. Crear retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. Crear LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

# 7. Crear cadena QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Hacer pregunta
query = "¿Cuál es el ancho mínimo de una puerta de evacuación?"
result = qa_chain({"query": query})

print(result["result"])  # Respuesta
print(result["source_documents"])  # Fuentes
```

---

## Implementación con LangChain

### Estructura del Proyecto

```
src/rag/
├── __init__.py
├── document_loader.py      # Carga PDFs
├── embeddings_config.py    # Configuración de embeddings
├── vectorstore_manager.py  # Gestión de vectorstore
└── qa_chain.py            # Cadena de QA
```

### document_loader.py

```python
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.schema import Document

def load_pdfs(pdf_dir: Path) -> List[Document]:
    """
    Carga todos los PDFs de un directorio.
    
    Args:
        pdf_dir: Directorio con PDFs
        
    Returns:
        Lista de documentos cargados
    """
    documents = []
    
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"Cargando: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
    
    print(f"Total documentos cargados: {len(documents)}")
    return documents
```

### embeddings_config.py

```python
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Crea instancia de embeddings.
    Usando modelo multilingual para soportar español.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},  # O 'cuda' si tienes GPU
        encode_kwargs={'normalize_embeddings': True}  # Normalizar vectores
    )
```

### vectorstore_manager.py

```python
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embeddings_config import get_embeddings
from .document_loader import load_pdfs

class VectorstoreManager:
    """Gestiona la creación y carga del vectorstore."""
    
    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vectorstore: Optional[Chroma] = None
    
    def create_from_pdfs(self, pdf_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Crea vectorstore desde PDFs.
        
        Args:
            pdf_dir: Directorio con PDFs
            chunk_size: Tamaño de cada chunk
            chunk_overlap: Solapamiento entre chunks
        """
        # 1. Cargar documentos
        documents = load_pdfs(pdf_dir)
        
        # 2. Dividir en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Total chunks: {len(chunks)}")
        
        # 3. Crear vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        print(f"Vectorstore creado en: {self.persist_directory}")
    
    def load_existing(self):
        """Carga vectorstore existente."""
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings
        )
        print(f"Vectorstore cargado desde: {self.persist_directory}")
    
    def get_retriever(self, k: int = 3):
        """
        Obtiene retriever para búsquedas.
        
        Args:
            k: Número de documentos a recuperar
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore no inicializado. Usa create_from_pdfs() o load_existing()")
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
```

### qa_chain.py

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = """
Eres un asistente experto en normativa de construcción española (CTE, CTE DB-SI, CTE DB-SUA).
Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Si la información está en el contexto, responde de forma clara y precisa
2. Siempre cita la fuente (nombre del documento, sección, página si está disponible)
3. Si no puedes responder con el contexto dado, di "No tengo información suficiente sobre esto en la normativa proporcionada"
4. No inventes ni supongas información

Contexto:
{context}

Pregunta: {question}

Respuesta detallada:
"""

def create_qa_chain(retriever, model_name: str = "gemini-pro", temperature: float = 0.1):
    """
    Crea cadena de QA con retrieval.
    
    Args:
        retriever: Retriever del vectorstore
        model_name: Nombre del modelo de Google
        temperature: Temperatura del LLM
    
    Returns:
        Cadena de QA configurada
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
    
    # Cadena
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Estrategia: meter todo el contexto
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def query(qa_chain, question: str):
    """
    Hace una pregunta a la cadena de QA.
    
    Args:
        qa_chain: Cadena de QA
        question: Pregunta del usuario
    
    Returns:
        Dict con 'result' y 'source_documents'
    """
    result = qa_chain({"query": question})
    return result
```

---

## Embeddings

### ¿Qué son?

Representaciones numéricas de texto que capturan significado semántico.

```
"ancho de puerta" → [0.12, -0.45, 0.89, ..., 0.34]
                     ↑
                  Vector de 384 dimensiones
```

### Tipos de Embeddings

#### 1. Word Embeddings (Antiguo)

Cada palabra tiene un vector fijo:

```python
# Word2Vec
"puerta" → [0.1, 0.5, ...]
"door" → [0.2, 0.6, ...]
```

**Problema**: "banco" (asiento) y "banco" (institución) tienen el mismo vector.

#### 2. Sentence Embeddings (Moderno)

Toda la frase tiene un vector:

```python
"ancho de puerta de evacuación" → [0.12, -0.45, ..., 0.34]
"width of evacuation door" → [0.15, -0.42, ..., 0.31]  # Similar
```

**Ventaja**: Captura contexto completo.

### Modelos Recomendados

#### Para Español/Multilingüe

```python
# Opción 1: Pequeño y rápido (recomendado)
"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# - Tamaño: 420 MB
# - Dimensiones: 384
# - Idiomas: 50+

# Opción 2: Más preciso pero más pesado
"sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# - Tamaño: 1.1 GB
# - Dimensiones: 768
# - Idiomas: 50+
```

#### Para Solo Inglés

```python
# Mejor rendimiento
"sentence-transformers/all-MiniLM-L6-v2"
# - Tamaño: 80 MB
# - Dimensiones: 384
```

### Uso en Código

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Embeddings para una query
query_vector = embeddings.embed_query("¿Ancho mínimo de puerta?")
print(f"Dimensiones: {len(query_vector)}")  # 384

# Embeddings para múltiples documentos
docs = ["doc1", "doc2", "doc3"]
doc_vectors = embeddings.embed_documents(docs)
print(f"Número de vectores: {len(doc_vectors)}")  # 3
```

---

## Vectorstores

### ¿Qué es un Vectorstore?

Base de datos especializada en:
1. Almacenar vectores de alta dimensionalidad
2. Buscar vectores similares eficientemente

```
┌───────────────────────────────────────┐
│         VECTORSTORE                   │
├───────────────────────────────────────┤
│ ID  │ Vector          │ Metadata      │
├─────┼─────────────────┼───────────────┤
│ 1   │ [0.1, 0.5, ...] │ {page: 1,...} │
│ 2   │ [0.2, 0.6, ...] │ {page: 2,...} │
│ 3   │ [0.3, 0.4, ...] │ {page: 3,...} │
└─────┴─────────────────┴───────────────┘
```

### Opciones Populares

#### ChromaDB (Recomendado para POC)

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectorstore"  # Persiste en disco
)
```

**Pros**:
- ✅ Simple de usar
- ✅ Funciona en local (no requiere servidor)
- ✅ Persiste automáticamente
- ✅ Bueno para desarrollo

**Cons**:
- ❌ No escalable para producción (millones de vectores)

#### Pinecone (Producción)

```python
from langchain_community.vectorstores import Pinecone

vectorstore = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="normativa"
)
```

**Pros**:
- ✅ Muy escalable
- ✅ Servicio gestionado
- ✅ Rápido en búsquedas

**Cons**:
- ❌ Requiere cuenta y pago
- ❌ Más complejo de configurar

#### FAISS (Local, Escalable)

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
vectorstore.save_local("./vectorstore")
```

**Pros**:
- ✅ Muy rápido
- ✅ Funciona en local
- ✅ Escalable a millones de vectores

**Cons**:
- ❌ Más complejo que ChromaDB
- ❌ Requiere más memoria RAM

### Búsqueda de Similitud

#### Distancia Coseno (Por Defecto)

Mide el ángulo entre vectores:

```python
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))

# Rango: [-1, 1]
# 1 = idénticos
# 0 = ortogonales
# -1 = opuestos
```

#### Distancia Euclidiana

Mide la distancia "real" entre vectores:

```python
from numpy import sqrt, sum

def euclidean_distance(v1, v2):
    return sqrt(sum((v1 - v2) ** 2))

# Rango: [0, ∞)
# 0 = idénticos
# Mayor valor = más diferentes
```

---

## Retrieval Strategies

### 🚀 Sistema Unificado de Estrategias

El sistema RAG implementado incluye **5 estrategias de recuperación** que puedes cambiar en tiempo real:

#### 1. **Basic Retrieval** (Búsqueda por Palabras Clave)

```python
# Estrategia básica - solo coincidencias de palabras
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

docs = retriever.get_relevant_documents("¿Ancho mínimo de puerta?")
# Devuelve los 3 chunks con más coincidencias de palabras
```

**Características**:
- ✅ Rápida y simple
- ✅ No requiere embeddings
- ❌ Solo busca coincidencias exactas de palabras
- ❌ No entiende el significado semántico

#### 2. **Semantic Retrieval** (Búsqueda Semántica)

```python
# Usando embeddings para entender significado
from sentence_transformers import SentenceTransformer

embeddings_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Convierte pregunta a vector y busca similares
query_embedding = embeddings_model.encode(["¿Ancho mínimo de puerta?"])
similarities = cosine_similarity(query_embedding, chunk_embeddings)
```

**Características**:
- ✅ Entiende el significado de las consultas
- ✅ Encuentra contenido relacionado conceptualmente
- ✅ Mejor para consultas complejas
- ❌ Requiere más recursos computacionales
- ❌ Puede perder coincidencias exactas importantes

#### 3. **Keyword Retrieval** (Búsqueda Avanzada por Palabras)

```python
# TF-IDF-like scoring con múltiples señales
def keyword_search(query, chunks):
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk['text'].lower().split())
        
        # Múltiples puntuaciones
        word_overlap = len(query_words.intersection(chunk_words))
        word_density = word_overlap / len(query_words)
        chunk_word_count = len(chunk_words)
        
        # Puntuación combinada
        score = word_overlap * 0.5 + word_density * 0.3 + (1 / (1 + chunk_word_count)) * 0.2
        scored_chunks.append((chunk, score))
    
    return sorted(scored_chunks, key=lambda x: x[1], reverse=True)
```

**Características**:
- ✅ Mejor que búsqueda básica
- ✅ Considera densidad de palabras y longitud
- ✅ No requiere embeddings
- ❌ Aún limitado a coincidencias exactas

#### 4. **Hybrid Retrieval** (Búsqueda Híbrida)

```python
# Combina semántica (70%) + keyword (30%)
def hybrid_search(query, chunks, semantic_weight=0.7):
    # Obtener resultados semánticos
    semantic_results = semantic_search(query, chunks)
    semantic_scores = {chunk['text']: score for chunk, score in semantic_results}
    
    # Obtener resultados por palabras clave
    keyword_results = keyword_search(query, chunks)
    keyword_scores = {chunk['text']: score for chunk, score in keyword_results}
    
    # Combinar puntuaciones
    combined_results = []
    for chunk_text in set(semantic_scores.keys()) | set(keyword_scores.keys()):
        semantic_score = semantic_scores.get(chunk_text, 0)
        keyword_score = keyword_scores.get(chunk_text, 0)
        
        # Puntuación combinada
        combined_score = (semantic_weight * semantic_score + 
                         (1 - semantic_weight) * keyword_score)
        combined_results.append((chunk_obj, combined_score))
    
    return sorted(combined_results, key=lambda x: x[1], reverse=True)
```

**Características**:
- ✅ Combina lo mejor de ambas estrategias
- ✅ 70% semántico + 30% keyword para balance óptimo
- ✅ Mejor precisión y recall
- ✅ Recomendada para producción

#### 5. **Rerank Retrieval** (Búsqueda con Reranking)

```python
# Híbrida + reranking con señales adicionales
def rerank_results(query, results):
    reranked = []
    
    for chunk, original_score in results:
        # Calcular señales de relevancia adicionales
        relevance_score = calculate_relevance_score(query, chunk)
        
        # Combinar puntuación original con relevancia
        final_score = original_score * 0.6 + relevance_score * 0.4
        reranked.append((chunk, final_score))
    
    return sorted(reranked, key=lambda x: x[1], reverse=True)

def calculate_relevance_score(query, chunk):
    text = chunk['text'].lower()
    query_lower = query.lower()
    
    score = 0.0
    
    # Coincidencia exacta de frase
    if query_lower in text:
        score += 0.3
    
    # Densidad de palabras de la consulta
    query_words = query_lower.split()
    chunk_words = text.split()
    word_matches = sum(1 for word in query_words if word in chunk_words)
    word_density = word_matches / len(query_words) if query_words else 0
    score += word_density * 0.2
    
    # Penalización por longitud (preferir chunks de longitud media)
    length_penalty = 1.0 - abs(len(chunk['text']) - 800) / 1000
    score += max(0, length_penalty) * 0.1
    
    # Bonus por posición de página (páginas tempranas más importantes)
    if chunk['metadata']['pages']:
        page_bonus = 1.0 - (min(chunk['metadata']['pages']) / 100)
        score += max(0, page_bonus) * 0.1
    
    return min(1.0, score)
```

**Características**:
- ✅ La estrategia más avanzada
- ✅ Considera múltiples señales de relevancia
- ✅ Optimiza longitud de chunks y posición de página
- ✅ Mejor rendimiento general

### 🔄 Cambio de Estrategias en Tiempo Real

```python
from scripts.unified_rag_system import UnifiedRAGSystem, RetrievalStrategy

# Inicializar con estrategia específica
rag = UnifiedRAGSystem(RetrievalStrategy.BASIC)

# Cambiar estrategia sin recargar documentos
rag.switch_strategy(RetrievalStrategy.SEMANTIC)
rag.switch_strategy(RetrievalStrategy.HYBRID)
rag.switch_strategy(RetrievalStrategy.RERANK)
```

### 📊 Comparación de Estrategias

| Estrategia | Velocidad | Precisión | Recursos | Uso Recomendado |
|------------|-----------|-----------|----------|-----------------|
| **Basic** | ⚡⚡⚡ | ⭐⭐ | 🟢 Bajo | Desarrollo rápido |
| **Semantic** | ⚡⚡ | ⭐⭐⭐⭐ | 🟡 Medio | Consultas complejas |
| **Keyword** | ⚡⚡⚡ | ⭐⭐⭐ | 🟢 Bajo | Búsquedas exactas |
| **Hybrid** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 🟡 Medio | **Producción** |
| **Rerank** | ⚡ | ⭐⭐⭐⭐⭐ | 🔴 Alto | **Máxima calidad** |

### 🛠️ Estrategias Tradicionales (LangChain)

#### MMR (Maximal Marginal Relevance)

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 20,  # Candidatos a considerar
        "lambda_mult": 0.5  # 0 = diversidad, 1 = relevancia
    }
)
```

**¿Cuándo usar MMR?**
- ✅ Cuando quieres perspectivas diferentes
- ✅ Para evitar chunks muy repetitivos
- ❌ No uses si necesitas máxima precisión

#### Threshold-based

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # Mínimo 80% de similitud
        "k": 5
    }
)
```

---

## 🚀 Sistema Unificado RAG

### Scripts Disponibles

El proyecto incluye múltiples scripts para diferentes casos de uso:

#### 1. **Sistema Unificado** (`unified_rag_system.py`)

```bash
# Usar con estrategia específica
python scripts/unified_rag_system.py --strategy basic
python scripts/unified_rag_system.py --strategy semantic
python scripts/unified_rag_system.py --strategy hybrid
python scripts/unified_rag_system.py --strategy rerank

# Modo interactivo con cambio de estrategias
python scripts/unified_rag_system.py --strategy basic
# Luego en el prompt:
❓ [basic] Tu consulta: switch semantic
🔄 Switching from basic to semantic
✅ Switched to semantic strategy
```

#### 2. **Comparación de Estrategias** (`retrieval_comparison.py`)

```bash
# Compara todas las estrategias lado a lado
python scripts/retrieval_comparison.py
```

**Salida esperada**:
```
🔍 Query: '¿Cuál es el ancho mínimo de una puerta de evacuación?'

1️⃣ BASIC RETRIEVAL (Keyword matching):
   1. Score: 5.000 | Páginas: [23]
      Texto: "El ancho mínimo de puertas de evacuación..."

2️⃣ SEMANTIC RETRIEVAL (Embeddings):
   1. Score: 0.755 | Páginas: [23]
      Texto: "CTE DB-SI establece que las puertas..."

3️⃣ HYBRID RETRIEVAL (Semantic + Keyword):
   1. Score: 2.029 | Páginas: [23]
      Texto: "El ancho mínimo de puertas de evacuación..."
```

#### 3. **Demo Avanzado** (`advanced_rag_demo.py`)

```bash
# Demo completo con todas las funcionalidades
python scripts/advanced_rag_demo.py
```

#### 4. **Test Básico** (`simple_rag_test.py`)

```bash
# Verificación rápida del sistema
python scripts/simple_rag_test.py
```

### Uso Programático

#### Inicialización Básica

```python
from scripts.unified_rag_system import UnifiedRAGSystem, RetrievalStrategy
from pathlib import Path

# Crear sistema RAG
rag = UnifiedRAGSystem(RetrievalStrategy.HYBRID)

# Cargar documentos
rag.load_and_chunk_pdf(Path("data/normativa/DBSI.pdf"))

# Hacer consulta
results = rag.retrieve("¿Cuál es el ancho mínimo de una puerta de evacuación?", top_k=3)

# Mostrar resultados
for i, (chunk, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.3f}")
    print(f"   Páginas: {chunk['metadata']['pages']}")
    print(f"   Texto: {chunk['text'][:100]}...")
```

#### Cambio Dinámico de Estrategias

```python
# Inicializar con estrategia básica
rag = UnifiedRAGSystem(RetrievalStrategy.BASIC)
rag.load_and_chunk_pdf(Path("data/normativa/DBSI.pdf"))

# Probar diferentes estrategias para la misma consulta
query = "¿Qué dice sobre distancias de evacuación?"

print("=== BASIC ===")
basic_results = rag.retrieve(query, top_k=2)
for chunk, score in basic_results:
    print(f"Score: {score:.3f} | {chunk['text'][:50]}...")

print("\n=== SEMANTIC ===")
rag.switch_strategy(RetrievalStrategy.SEMANTIC)
semantic_results = rag.retrieve(query, top_k=2)
for chunk, score in semantic_results:
    print(f"Score: {score:.3f} | {chunk['text'][:50]}...")

print("\n=== HYBRID ===")
rag.switch_strategy(RetrievalStrategy.HYBRID)
hybrid_results = rag.retrieve(query, top_k=2)
for chunk, score in hybrid_results:
    print(f"Score: {score:.3f} | {chunk['text'][:50]}...")
```

#### Integración con OpenAI

```python
import openai
from scripts.unified_rag_system import UnifiedRAGSystem, RetrievalStrategy

# Configurar OpenAI
openai.api_key = "tu-api-key"

# Sistema RAG
rag = UnifiedRAGSystem(RetrievalStrategy.RERANK)
rag.load_and_chunk_pdf(Path("data/normativa/DBSI.pdf"))

def query_with_openai(question: str) -> str:
    # Recuperar contexto relevante
    results = rag.retrieve(question, top_k=3)
    context = "\n\n".join([chunk['text'] for chunk, score in results])
    
    # Generar respuesta con OpenAI
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en normativa de construcción española."},
            {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {question}"}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content

# Usar
answer = query_with_openai("¿Cuál es el ancho mínimo de una puerta de evacuación?")
print(answer)
```

### Configuración Avanzada

#### Personalizar Parámetros

```python
# Crear sistema con parámetros personalizados
rag = UnifiedRAGSystem(RetrievalStrategy.HYBRID)

# Cargar con chunking personalizado
rag.load_and_chunk_pdf(
    Path("data/normativa/DBSI.pdf"),
    chunk_size=1500,  # Chunks más grandes
    overlap=300       # Más solapamiento
)

# Recuperar con más resultados
results = rag.retrieve("pregunta", top_k=5)
```

#### Manejo de Errores

```python
try:
    rag = UnifiedRAGSystem(RetrievalStrategy.SEMANTIC)
    rag.load_and_chunk_pdf(Path("data/normativa/DBSI.pdf"))
    
    results = rag.retrieve("pregunta", top_k=3)
    
except FileNotFoundError:
    print("❌ Archivo PDF no encontrado")
except ImportError:
    print("❌ Dependencias faltantes. Instala: pip install sentence-transformers")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Rendimiento y Optimización

#### Benchmarking de Estrategias

```python
import time
from scripts.unified_rag_system import UnifiedRAGSystem, RetrievalStrategy

rag = UnifiedRAGSystem(RetrievalStrategy.BASIC)
rag.load_and_chunk_pdf(Path("data/normativa/DBSI.pdf"))

query = "¿Cuál es el ancho mínimo de una puerta de evacuación?"

strategies = [
    RetrievalStrategy.BASIC,
    RetrievalStrategy.SEMANTIC,
    RetrievalStrategy.KEYWORD,
    RetrievalStrategy.HYBRID,
    RetrievalStrategy.RERANK
]

for strategy in strategies:
    rag.switch_strategy(strategy)
    
    start_time = time.time()
    results = rag.retrieve(query, top_k=3)
    end_time = time.time()
    
    print(f"{strategy.value:10} | {end_time - start_time:.3f}s | Score: {results[0][1]:.3f}")
```

**Salida esperada**:
```
basic      | 0.001s | Score: 5.000
semantic   | 0.045s | Score: 0.755
keyword    | 0.002s | Score: 3.733
hybrid     | 0.047s | Score: 1.369
rerank     | 0.048s | Score: 0.925
```

---

## Best Practices

### 1. Chunk Size Óptimo

```python
# Para documentación técnica
chunk_size = 1000  # 1000 caracteres ≈ 250 tokens
chunk_overlap = 200  # 20% overlap

# Para preguntas cortas
chunk_size = 500
chunk_overlap = 100

# Para contextos largos
chunk_size = 1500
chunk_overlap = 300
```

**Regla general**: 
- Chunks pequeños → Búsqueda precisa, pero puede perder contexto
- Chunks grandes → Más contexto, pero menos preciso

### 2. Metadata

Añade metadata útil a cada chunk:

```python
from langchain.schema import Document

doc = Document(
    page_content="El ancho mínimo es 80cm",
    metadata={
        "source": "CTE_DB-SI.pdf",
        "page": 23,
        "section": "4.2 Puertas de evacuación",
        "chapter": "Evacuación"
    }
)
```

Luego puedes filtrar por metadata:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"section": "4.2 Puertas de evacuación"}
    }
)
```

### 3. Prompt Engineering

**Mal prompt**:
```python
"Responde la pregunta: {question}"
```

**Buen prompt**:
```python
"""
Eres un experto en normativa de construcción.
Responde basándote ÚNICAMENTE en el contexto.
Cita siempre la fuente.

Contexto: {context}
Pregunta: {question}
Respuesta:
"""
```

### 4. Temperatura del LLM

```python
# Para respuestas técnicas/precisas
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

# Para respuestas creativas
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
```

### 5. Evaluación

Mide la calidad de tu RAG:

```python
from langchain.evaluation.qa import QAEvalChain

# Pares de pregunta-respuesta esperada
examples = [
    {
        "query": "¿Ancho mínimo de puerta?",
        "answer": "80cm según CTE DB-SI"
    },
    # ...
]

# Evaluar
predictions = [qa_chain(ex["query"]) for ex in examples]
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
```

---

## Debugging & Optimization

### Problema 1: Respuestas Incorrectas

**Diagnóstico**:
```python
result = qa_chain({"query": "¿Ancho mínimo de puerta?"})

# 1. Revisar chunks recuperados
for doc in result["source_documents"]:
    print(doc.page_content)
    print(doc.metadata)
    print("---")
```

**Posibles causas**:
- ❌ Chunks no contienen la información
- ❌ K demasiado bajo (aumenta a 5-7)
- ❌ Embeddings no son buenos para español

**Solución**:
```python
# Aumentar K
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Cambiar modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # Mejor
)
```

### Problema 2: Contexto Insuficiente

**Solución**: Aumentar chunk_size y overlap

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Era 1000
    chunk_overlap=300  # Era 200
)
```

### Problema 3: Lento

**Diagnóstico**:
```python
import time

start = time.time()
result = qa_chain({"query": "pregunta"})
print(f"Tiempo: {time.time() - start:.2f}s")
```

**Optimizaciones**:

1. **Usar GPU para embeddings**:
```python
embeddings = HuggingFaceEmbeddings(
    model_kwargs={'device': 'cuda'}  # Era 'cpu'
)
```

2. **Reducir K**:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Era 5
```

3. **Usar modelo de embeddings más pequeño**:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Más rápido
)
```

### Problema 4: Vectorstore Muy Grande

**Solución**: Limpieza y optimización

```python
# Solo indexar PDFs relevantes
pdfs_to_index = [
    "CTE_DB-SI.pdf",
    "CTE_DB-SUA.pdf"
]

documents = []
for pdf in pdfs_to_index:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())
```

---

## Ejemplo Completo

```python
# main.py
from pathlib import Path
from src.rag.vectorstore_manager import VectorstoreManager
from src.rag.qa_chain import create_qa_chain, query

def main():
    # Configuración
    PDF_DIR = Path("data/normativa")
    VECTORSTORE_DIR = Path("vectorstore/normativa_db")
    
    # Crear o cargar vectorstore
    rag = VectorstoreManager(VECTORSTORE_DIR)
    
    if not VECTORSTORE_DIR.exists():
        print("Creando vectorstore por primera vez...")
        rag.create_from_pdfs(PDF_DIR)
    else:
        print("Cargando vectorstore existente...")
        rag.load_existing()
    
    # Crear retriever
    retriever = rag.get_retriever(k=3)
    
    # Crear cadena de QA
    qa_chain = create_qa_chain(retriever)
    
    # Hacer preguntas
    questions = [
        "¿Cuál es el ancho mínimo de una puerta de evacuación?",
        "¿Qué dice el CTE sobre distancias máximas de evacuación?",
        "¿Requisitos de resistencia al fuego para muros?"
    ]
    
    for question in questions:
        print(f"\nPregunta: {question}")
        result = query(qa_chain, question)
        print(f"Respuesta: {result['result']}")
        print("\nFuentes:")
        for doc in result['source_documents']:
            print(f"  - {doc.metadata.get('source', 'Unknown')}, Página {doc.metadata.get('page', 'N/A')}")

if __name__ == "__main__":
    main()
```

---

## Recursos Adicionales

### Documentación
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
- [Chroma](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### Tutoriales
- [RAG from Scratch](https://www.youtube.com/watch?v=LhnCsygAvzY)
- [Advanced RAG Techniques](https://www.youtube.com/watch?v=sVcwVQRHIc8)

### Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

---

## 📋 Resumen del Sistema RAG Implementado

### ✅ Características Implementadas

- **5 Estrategias de Recuperación**: Basic, Semantic, Keyword, Hybrid, Rerank
- **Cambio en Tiempo Real**: Switch entre estrategias sin recargar documentos
- **Sistema Unificado**: Una interfaz para todas las estrategias
- **Scripts de Demostración**: Múltiples herramientas de testing y comparación
- **Integración OpenAI**: Compatible con gpt-3.5-turbo, gpt-4, gpt-5-nano
- **Embeddings Multilingües**: Optimizado para español
- **Reranking Avanzado**: Múltiples señales de relevancia
- **Documentación Completa**: Guías de uso y ejemplos

### 🚀 Scripts Disponibles

| Script | Propósito | Uso Recomendado |
|--------|-----------|-----------------|
| `unified_rag_system.py` | Sistema principal con cambio de estrategias | **Producción** |
| `retrieval_comparison.py` | Comparación lado a lado de estrategias | **Análisis** |
| `advanced_rag_demo.py` | Demo completo con todas las funcionalidades | **Demostración** |
| `simple_rag_test.py` | Test básico de funcionalidad | **Verificación** |
| `working_rag_demo.py` | Demo simplificado sin LangChain | **Fallback** |

### 🎯 Recomendaciones de Uso

#### Para Desarrollo
```bash
# Test rápido
python scripts/simple_rag_test.py

# Comparar estrategias
python scripts/retrieval_comparison.py
```

#### Para Producción
```bash
# Sistema unificado con estrategia híbrida
python scripts/unified_rag_system.py --strategy hybrid

# O con reranking para máxima calidad
python scripts/unified_rag_system.py --strategy rerank
```

#### Para Análisis
```bash
# Demo completo
python scripts/advanced_rag_demo.py
```

### 📊 Rendimiento por Estrategia

| Estrategia | Velocidad | Precisión | Recursos | Caso de Uso |
|------------|-----------|-----------|----------|-------------|
| **Basic** | ⚡⚡⚡ | ⭐⭐ | 🟢 Bajo | Desarrollo rápido |
| **Semantic** | ⚡⚡ | ⭐⭐⭐⭐ | 🟡 Medio | Consultas complejas |
| **Keyword** | ⚡⚡⚡ | ⭐⭐⭐ | 🟢 Bajo | Búsquedas exactas |
| **Hybrid** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 🟡 Medio | **Producción** |
| **Rerank** | ⚡ | ⭐⭐⭐⭐⭐ | 🔴 Alto | **Máxima calidad** |

### 🔧 Configuración Mínima

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar API key
export OPENAI_API_KEY="tu-api-key"

# 3. Probar sistema
python scripts/simple_rag_test.py

# 4. Usar sistema unificado
python scripts/unified_rag_system.py --strategy hybrid
```

### 📁 Estructura de Archivos

```
scripts/
├── unified_rag_system.py      # Sistema principal
├── retrieval_comparison.py    # Comparación de estrategias
├── advanced_rag_demo.py       # Demo avanzado
├── simple_rag_test.py         # Test básico
└── working_rag_demo.py        # Demo simplificado

src/rag/
├── __init__.py
├── document_loader.py         # Carga de PDFs
├── embeddings_config.py       # Configuración embeddings
├── vectorstore_manager.py     # Gestión vectorstore
└── qa_chain.py               # Cadena de QA

docs/
└── rag_explained_md.md       # Esta documentación
```

---

**Versión**: 2.0  
**Última actualización**: Diciembre 2024  
**Estado**: ✅ Sistema completo y funcional
