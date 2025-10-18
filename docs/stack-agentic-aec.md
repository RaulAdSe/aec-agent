# Stack Tecnológico: Agentic AI for AEC

## 🎯 Objetivo del Sistema

Single-agent con ReACT que:
1. Lee datos estructurados de planos (RVT/DWG → JSON)
2. Realiza cálculos geométricos y de rutas
3. Consulta normativa vía RAG
4. Verifica cumplimiento normativo
5. Genera informe con trazabilidad

---

## 🛠️ Stack Completo

### 1. **Extracción de Datos** (Pre-procesamiento)

**Para RVT:**
- `pyRevit` (ejecutar dentro de Revit)
- Script Python → JSON estructurado

**Para DWG/DXF:**
- `ezdxf` (Python puro, sin dependencias)
- Opcionalmente: ODA File Converter (DWG→DXF)

**Output:** JSON unificado con schema común

---

### 2. **Procesamiento Geométrico & Cálculos**

```bash
pip install shapely==2.0.2        # Geometría 2D
pip install networkx==3.2.1       # Grafos y rutas
pip install numpy==1.24.3         # Cálculos numéricos
pip install scipy==1.11.4         # Operaciones científicas
```

**Funciones:**
- Cálculo de áreas, perímetros, centroides
- Distancias shortest-path
- Intersecciones geométricas
- Buffer zones para pasillos

---

### 3. **Sistema RAG (Retrieval Augmented Generation)**

```bash
pip install langchain==0.1.0
pip install langchain-openai==0.0.2
pip install langchain-google-genai==0.0.6
pip install chromadb==0.4.22
pip install sentence-transformers==2.3.1
pip install pypdf==3.17.4
pip install pymupdf==1.23.8       # Mejor para PDFs técnicos
```

**Base de Conocimiento:**
- CTE DB-SI (Seguridad en caso de incendio)
- CTE DB-SUA (Seguridad de utilización y accesibilidad)
- RD 513/2017 (RIPCI)
- Normativa local si aplica

**Vectorstore:** ChromaDB (local, sin servidor)

---

### 4. **Agente con LangGraph**

```bash
pip install langgraph==0.0.20     # Framework de agentes
pip install langchain-core==0.1.10
```

**Arquitectura:**
- Single agent con ReACT pattern
- Tools deterministicas (Python functions)
- State management con TypedDict
- Human-in-the-loop opcional

---

### 5. **LLM (elige uno)**

**Opción A: Gemini (RECOMENDADO)**
```bash
pip install google-generativeai==0.3.2
```
- **Modelo:** `gemini-2.0-flash-exp` o `gemini-1.5-pro`
- **Pros:** Gratis hasta cierto límite, rápido, excelente con contexto largo
- **Cons:** API puede tener rate limits

**Opción B: OpenAI**
```bash
pip install openai==1.10.0
```
- **Modelo:** `gpt-4o` o `gpt-4-turbo`
- **Pros:** Muy fiable, function calling robusto
- **Cons:** Pago desde el primer token

**Opción C: Claude (API)**
```bash
pip install anthropic==0.8.1
```
- **Modelo:** `claude-3-5-sonnet-20241022`
- **Pros:** Excelente razonamiento, menos alucinaciones
- **Cons:** Pago, puede ser más lento

**Para tu charla: Gemini Flash (gratis y rápido)**

---

### 6. **Validación & Schemas**

```bash
pip install pydantic==2.5.3
```

Para definir schemas de datos y validar inputs/outputs.

---

### 7. **Visualización & Reporting**

```bash
pip install matplotlib==3.8.2
pip install jupyter==1.0.0
pip install ipywidgets==8.1.1
```

Para el notebook interactivo.

---

### 8. **Utilidades**

```bash
pip install python-dotenv==1.0.0  # Para API keys
pip install rich==13.7.0          # Pretty printing en terminal
pip install tqdm==4.66.1          # Progress bars
```

---

## 📁 Estructura del Proyecto

```
aec-compliance-agent/
│
├── requirements.txt
├── .env                          # API keys
├── .gitignore
│
├── data/
│   ├── raw/                      # RVT/DWG originales
│   ├── extracted/                # JSON extraídos
│   │   ├── project_001.json
│   │   └── project_002.json
│   └── normativa/                # PDFs de normativa
│       ├── CTE_DB-SI.pdf
│       ├── CTE_DB-SUA.pdf
│       └── RD_513_2017.pdf
│
├── notebooks/
│   └── agentic_ai_aec_tutorial.ipynb   # TUTORIAL PRINCIPAL
│
├── src/
│   ├── __init__.py
│   ├── extraction/               # Scripts de extracción
│   │   ├── rvt_export.py        # pyRevit script
│   │   └── dxf_export.py        # ezdxf script
│   ├── schemas.py                # Pydantic models
│   ├── geometry.py               # Shapely functions
│   ├── graph.py                  # NetworkX functions
│   ├── checks.py                 # Verification functions
│   ├── rag.py                    # RAG setup & queries
│   └── agent.py                  # LangGraph agent
│
├── vectorstore/                  # ChromaDB persistence
│   └── chroma_db/
│
└── outputs/                      # Informes generados
    └── reports/
```

---

## 🔧 Instalación Rápida

### requirements.txt completo

```txt
# Core
python-dotenv==1.0.0

# Extracción
ezdxf==1.1.3

# Geometría & Cálculos
shapely==2.0.2
networkx==3.2.1
numpy==1.24.3
scipy==1.11.4

# LLM & LangChain
langchain==0.1.0
langchain-core==0.1.10
langchain-openai==0.0.2
langchain-google-genai==0.0.6
langgraph==0.0.20

# RAG
chromadb==0.4.22
sentence-transformers==2.3.1
pypdf==3.17.4
pymupdf==1.23.8

# LLM APIs
google-generativeai==0.3.2
# openai==1.10.0          # Descomentar si usas OpenAI
# anthropic==0.8.1        # Descomentar si usas Claude

# Validación
pydantic==2.5.3

# Visualización
matplotlib==3.8.2
jupyter==1.0.0
ipywidgets==8.1.1

# Utilidades
rich==13.7.0
tqdm==4.66.1
```

### Setup

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Mac/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Crear .env con tu API key
echo "GOOGLE_API_KEY=tu-api-key-aqui" > .env
```

---

## 🏗️ Arquitectura del Agente (LangGraph)

### State Schema

```python
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel

class AgentState(TypedDict):
    """Estado del agente que se propaga entre nodos"""
    messages: List[Dict[str, Any]]     # Historial de conversación
    model_data: Dict[str, Any]         # JSON del plano cargado
    rules: Dict[str, Any]              # Reglas extraídas vía RAG
    checks_results: List[Dict[str, Any]]  # Resultados de verificaciones
    graph_handle: Any                  # Grafo de NetworkX
    current_task: str                  # Tarea actual
    iterations: int                    # Contador de iteraciones
    final_report: str                  # Informe final
```

### Nodos del Grafo

```
START
  ↓
[load_model]  ← Carga JSON del plano
  ↓
[query_rag]  ← Obtiene reglas normativas
  ↓
[agent_loop] ← ReACT: piensa → actúa → observa
  ↓  ↑
  ↓  └─ (loop hasta resolver)
  ↓
[generate_report] ← Consolida resultados
  ↓
END
```

### Tools Disponibles

1. **get_room_info(room_id)** → área, perímetro, centroide
2. **get_door_info(door_id)** → ancho, alto, fire rating
3. **calculate_egress_distance(room_id, exit_ids)** → distancia mínima
4. **check_door_width(door_id, min_mm)** → cumple/no cumple
5. **check_corridor_width(corridor_id, min_m)** → cumple/no cumple
6. **check_fire_rating(element_id, min_rating)** → cumple/no cumple
7. **list_all_rooms()** → lista de IDs y nombres
8. **list_all_doors()** → lista de puertas
9. **query_normativa(question)** → respuesta desde RAG

---

## 🎓 Para la Charla (29 Octubre)

### Estructura Notebook (45 min)

**Parte 1: Extracción (5 min)**
- Mostrar JSON ya extraído
- Explicar schema unificado
- Quick tour de pyRevit/ezdxf

**Parte 2: Cálculos Básicos (5 min)**
- Shapely: calcular área de room
- NetworkX: shortest path
- Demo visual con matplotlib

**Parte 3: RAG (10 min)**
- Qué es RAG y por qué es crucial
- Cargar PDFs de normativa
- Query: "¿cuál es el ancho mínimo de puerta de evacuación?"
- Mostrar chunks relevantes + respuesta

**Parte 4: Agents & ReACT (15 min)**
- Qué son los agentes
- ReACT framework (Reason → Act → Observe)
- Tools como funciones Python
- LangGraph: state + nodes + edges
- Demo: agente verifica 1 habitación

**Parte 5: Sistema Completo (10 min)**
- Run completo: proyecto → informe
- Mostrar iteraciones del agente
- Informe final con trazabilidad
- Discusión: limitaciones y próximos pasos

---

## 🚀 Diferenciadores vs. Ichi/CivCheck

| Feature | Ichi/CivCheck | Tu Sistema |
|---------|---------------|------------|
| **Extracción** | Upload PDF | RVT/DWG nativos |
| **Cálculos** | ❓ Black box | ✅ Transparente (Python) |
| **Normativa** | Hardcoded | 🔥 RAG (actualizable) |
| **Agente** | Fixed workflow | 🧠 ReACT (adaptativo) |
| **Trazabilidad** | Citations | Citations + código |
| **Local** | ❌ Cloud only | ✅ Puede ser 100% local |
| **Educativo** | ❌ No | ✅ Open source concept |

---

## 💡 Tips para la Presentación

1. **Empieza con el "wow"**: Muestra el informe final primero, luego desglosa
2. **Visualiza**: Usa matplotlib para mostrar planos con rutas de evacuación
3. **Interactivo**: Deja que cambien un parámetro y vean el resultado
4. **Honesto**: Menciona limitaciones (precisión OCR, cobertura normativa)
5. **Futuro**: Menciona multi-agente, visión computacional, BIM colaborativo

---

## 📚 Referencias para la Charla

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **ReACT Paper**: "ReAct: Synergizing Reasoning and Acting in Language Models"
- **RAG Tutorial**: LangChain RAG from scratch
- **AEC Tech**: Mencion a Armeta, SOCOTEC BlueGen como industria context

---

## 🎯 Next Steps Después de la Charla

1. **MVP con 3 checks** (ancho puertas, distancia evacuación, fire rating)
2. **Más normativa** en el RAG (DB-SE, DB-HE, normativas locales)
3. **Multi-agente** (un agente por disciplina)
4. **Computer Vision** para extraer datos de planos escaneados
5. **Web app** con Streamlit para demo interactiva
6. **Validación** con expertos reales (¿tu contacto Raúl?)

---

## 🔥 Killer Features para Impresionar

1. **Trazabilidad completa**: Cada veredicto con código normativo específico
2. **Visualización de rutas**: Plot 2D con matplotlib mostrando egress paths
3. **Iteraciones visibles**: Mostrar el "pensamiento" del agente en tiempo real
4. **Actualización rápida**: Subir nuevo PDF de normativa → automáticamente en RAG
5. **Explicaciones**: El agente puede explicar "por qué" algo no cumple

---

## 🎬 Demo Script Sugerido

```
"Imaginad que acabáis de recibir un proyecto de un cliente.
Tenéis el RVT/DWG y necesitáis verificar contra CTE DB-SI.

[CLICK: Carga JSON]
'Aquí tenemos los datos extraídos: 15 rooms, 23 puertas, 8 salidas.'

[CLICK: Query RAG]
'Preguntamos a nuestra base de conocimiento: ¿qué dice el CTE sobre evacuación?'

[MUESTRA: Chunks + respuesta]
'Nos devuelve las secciones relevantes con citas exactas.'

[CLICK: Run agent]
'Ahora el agente empieza a trabajar. Veréis cómo razona...'

[MUESTRA: Iteraciones en tiempo real]
- 'Necesito saber el ancho de la puerta D-12'
- 'Calculo distancia desde room 1.02 a salidas'
- 'Verifico contra regla: max 25m'
- 'ALERTA: Distancia 28m > 25m'

[CLICK: Ver informe final]
'Y genera un informe profesional con todos los hallazgos y referencias.'

¿Preguntas?"
```

---

## ✅ Checklist Pre-Charla

- [ ] Entorno Python configurado
- [ ] API key de Gemini funcionando
- [ ] Al menos 1 proyecto de ejemplo (JSON) cargado
- [ ] PDFs de normativa descargados y en RAG
- [ ] Notebook testeado end-to-end
- [ ] Visualizaciones preparadas
- [ ] Tiempo estimado por sección
- [ ] Plan B si API falla (resultados pre-generados)
- [ ] Slides de apoyo (mínimas)
- [ ] Repositorio GitHub listo para compartir

---

## 🌟 Mensaje de Cierre

> "La IA no va a reemplazar a los arquitectos técnicos.
> Pero los arquitectos técnicos que usen IA van a
> reemplazar a los que no lo hagan.
> 
> Este sistema es solo el principio. El futuro de AEC
> es colaboración humano-IA, con trazabilidad,
> transparencia y eficiencia.
> 
> El código está disponible. Experimentad. Mejoradlo.
> Y cuando lo hagáis, compartidlo con la comunidad.
> 
> Gracias."

---

**¿Listo para construir el notebook?** 🚀