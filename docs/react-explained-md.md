# 🤖 ReAct Framework - Explicación Técnica

## 📋 Tabla de Contenidos

1. [¿Qué es ReAct?](#qué-es-react)
2. [El Problema que Resuelve](#el-problema-que-resuelve)
3. [Cómo Funciona ReAct](#cómo-funciona-react)
4. [Componentes del Framework](#componentes-del-framework)
5. [Implementación con LangGraph](#implementación-con-langgraph)
6. [Tools (Herramientas)](#tools-herramientas)
7. [Ejemplos Paso a Paso](#ejemplos-paso-a-paso)
8. [Best Practices](#best-practices)
9. [Debugging](#debugging)
10. [Comparación con Otros Frameworks](#comparación-con-otros-frameworks)

---

## ¿Qué es ReAct?

**ReAct = Reasoning + Acting**

Framework de prompting que permite a los LLMs:
- 🧠 **Reason** (Razonar): Pensar sobre qué hacer
- 🛠️ **Act** (Actuar): Ejecutar acciones (usar herramientas)
- 👁️ **Observe** (Observar): Ver resultados de las acciones
- 🔄 **Iterate** (Iterar): Repetir hasta completar la tarea

### Paper Original

**"ReAct: Synergizing Reasoning and Acting in Language Models"**
- Autores: Yao et al., ICLR 2023
- Link: https://arxiv.org/abs/2210.03629

---

## El Problema que Resuelve

### LLMs Tradicionales: Limitaciones

#### Problema 1: No Pueden Actuar en el Mundo Real

```python
# LLM tradicional
pregunta = "¿Cuál es el ancho de la puerta D201 del proyecto?"

respuesta = llm.invoke(pregunta)
# Output: "No tengo acceso a esa información específica" ❌
```

#### Problema 2: No Pueden Actualizar Conocimiento

```python
pregunta = "¿Qué dice la última versión del CTE sobre rampas?"

respuesta = llm.invoke(pregunta)
# Output: [Información desactualizada del training] ❌
```

#### Problema 3: No Pueden Realizar Cálculos Complejos

```python
pregunta = "Calcula la distancia de evacuación desde R201 a la salida más cercana"

respuesta = llm.invoke(pregunta)
# Output: "Aproximadamente 15-20 metros" (inventa) ❌
```

### ReAct: La Solución

El agente puede **usar herramientas** (tools) para:
- ✅ Leer datos reales de planos
- ✅ Consultar normativa actualizada (RAG)
- ✅ Ejecutar cálculos precisos (Python functions)
- ✅ Tomar decisiones basadas en resultados

```
Agent ReAct:
1. THOUGHT: "Necesito saber el ancho de D201"
2. ACTION: get_door_info("D201")
3. OBSERVATION: {"width_mm": 900}
4. THOUGHT: "900mm. La normativa requiere mínimo..."
5. ACTION: query_normativa("ancho mínimo puerta")
6. OBSERVATION: "Mínimo 800mm según CTE..."
7. THOUGHT: "900 > 800, entonces cumple"
8. ANSWER: "La puerta D201 cumple (900mm > 800mm requerido)"
```

---

## Cómo Funciona ReAct

### Loop Principal

```
┌─────────────────────────────────────────────────┐
│              START: User Query                   │
│       "Verifica puertas del proyecto"           │
└────────────────┬────────────────────────────────┘
                 │
       ┌─────────▼──────────┐
       │   THOUGHT PHASE    │
       │  (LLM Reasoning)   │
       │                    │
       │ "¿Qué necesito     │
       │  hacer primero?"   │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │   ACTION PHASE     │
       │  (Tool Selection)  │
       │                    │
       │ "Usar tool:        │
       │  list_all_doors()" │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │  OBSERVATION PHASE │
       │  (Tool Execution)  │
       │                    │
       │ Tool returns:      │
       │ [D201, D202, D203] │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │   THOUGHT PHASE    │
       │                    │
       │ "Ahora necesito    │
       │  verificar cada    │
       │  una..."           │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │   ACTION PHASE     │
       │                    │
       │ "Usar tool:        │
       │  get_door_info     │
       │  (D201)"           │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │  OBSERVATION PHASE │
       │                    │
       │ {"width_mm": 900,  │
       │  "fire_rating":""}│
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │   THOUGHT PHASE    │
       │                    │
       │ "Falta fire rating!│
       │  ¿Es obligatorio?" │
       └─────────┬──────────┘
                 │
       ┌─────────▼──────────┐
       │   ACTION PHASE     │
       │                    │
       │ query_normativa(   │
       │ "fire rating door")│
       └─────────┬──────────┘
                 │
                ...
                 │
       ┌─────────▼──────────┐
       │   FINAL ANSWER     │
       │                    │
       │ "Verificación      │
       │  completa:         │
       │  - D201: ✓         │
       │  - D202: ✗ (...)  │
       │  - D203: ✓         │
       └────────────────────┘
```

### Prompt Template (Simplificado)

```
You are an AI agent with access to tools.

Your task: {user_query}

You can use these tools:
- get_door_info(door_id): Get door details
- query_normativa(question): Query building codes
- calculate_distance(from, to): Calculate distances

FORMAT:
Thought: [Your reasoning about what to do next]
Action: [tool_name(arg1, arg2, ...)]
Observation: [Result will appear here]
... (repeat Thought/Action/Observation as needed)
Thought: I have enough information to answer
Final Answer: [Your complete answer]

BEGIN:

Thought: {agent_scratchpad}
```

---

## Componentes del Framework

### 1. **Agent** (El Cerebro)

El LLM que toma decisiones.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

agent_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,  # Determinístico para tareas específicas
)
```

**Configuración clave**:
- `temperature=0`: Para decisiones consistentes
- `top_p=1.0`: No muestrear, tomar lo más probable
- `max_tokens=2048`: Suficiente para razonar

### 2. **Tools** (Las Manos)

Funciones Python que el agente puede llamar.

```python
from langchain_core.tools import tool

@tool
def get_door_info(door_id: str) -> dict:
    """
    Get detailed information about a door.
    
    Args:
        door_id: The ID of the door (e.g., 'D201')
    
    Returns:
        Dictionary with door properties
    """
    # Implementation...
    return {"id": door_id, "width_mm": 900, ...}
```

**Características importantes**:
- ✅ Docstring clara (el LLM la lee para decidir cuándo usar la tool)
- ✅ Type hints (ayudan al LLM a pasar argumentos correctos)
- ✅ Retorno estructurado (dict/JSON preferible)

### 3. **Memory/State** (La Memoria)

Almacena el historial de la conversación.

```python
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Estado del agente que persiste entre iteraciones"""
    messages: List[BaseMessage]      # Historial completo
    iterations: int                  # Contador de loops
    current_task: str                # Tarea actual
    data: dict                       # Datos del proyecto
    checks_completed: List[dict]     # Verificaciones ya hechas
```

### 4. **Executor** (El Coordinador)

Orquesta el loop Thought → Action → Observation.

```python
from langgraph.graph import StateGraph, END

def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Nodo: agente piensa y actúa
    workflow.add_node("agent", agent_node)
    
    # Nodo: ejecuta tools
    workflow.add_node("tools", tools_node)
    
    # Edges condicionales
    workflow.add_conditional_edges(
        "agent",
        should_continue,  # Función que decide: ¿continuar o terminar?
        {
            "continue": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")  # Después de tool → volver a agente
    
    workflow.set_entry_point("agent")
    
    return workflow.compile()
```

---

## Implementación con LangGraph

### Setup Completo

```python
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# 1. Definir Estado
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    iterations: int

# 2. Definir Tools
@tool
def get_room_area(room_id: str) -> float:
    """Get the area of a room in square meters."""
    # Implementación real
    return 85.5

@tool
def check_door_width(door_id: str, min_width_mm: float) -> dict:
    """Check if a door meets minimum width requirement."""
    actual_width = 900  # De datos reales
    return {
        "compliant": actual_width >= min_width_mm,
        "actual_mm": actual_width,
        "required_mm": min_width_mm
    }

tools = [get_room_area, check_door_width]

# 3. Configurar LLM con tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)

# 4. Definir nodos del grafo
def agent_node(state: AgentState):
    """Nodo donde el agente piensa y decide acción"""
    messages = state["messages"]
    iterations = state.get("iterations", 0)
    
    # Límite de iteraciones para evitar loops infinitos
    if iterations >= 10:
        return {
            "messages": [AIMessage(content="STOP - Max iterations reached")],
            "iterations": iterations + 1
        }
    
    # LLM decide qué hacer
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "iterations": iterations + 1
    }

def tools_node(state: AgentState):
    """Nodo que ejecuta las tools solicitadas"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Ejecutar tools si hay tool_calls
    tool_executor = ToolExecutor(tools)
    outputs = []
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_output = tool_executor.invoke(
                ToolInvocation(
                    tool=tool_call["name"],
                    tool_input=tool_call["args"]
                )
            )
            
            # Crear ToolMessage con el resultado
            outputs.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                )
            )
    
    return {"messages": outputs}

def should_continue(state: AgentState):
    """Decide si continuar o terminar"""
    last_message = state["messages"][-1]
    
    # Si el mensaje tiene tool_calls, continuar para ejecutarlas
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    
    # Si no hay tool_calls, el agente ha terminado
    return "end"

# 5. Construir grafo
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")
workflow.set_entry_point("agent")

# 6. Compilar
app = workflow.compile()

# 7. Ejecutar
initial_state = {
    "messages": [HumanMessage(content="Check if door D201 meets the minimum width of 800mm")],
    "iterations": 0
}

for state in app.stream(initial_state):
    print(state)
```

### Visualizar el Grafo

```python
from IPython.display import Image, display

# LangGraph puede generar visualización
display(Image(app.get_graph().draw_mermaid_png()))
```

Genera:
```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────┐     
│  agent  │◄──────┐
└────┬────┘       │
     │            │
     ▼            │
[should_continue] │
     │            │
     ├─continue──►│
     │         ┌──┴───┐
     │         │tools │
     │         └──────┘
     │
     └─end─►┌─────┐
            │ END │
            └─────┘
```

---

## Tools (Herramientas)

### Anatomía de una Tool

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def calculate_egress_distance(
    room_id: str,
    exit_ids: Optional[List[str]] = None
) -> dict:
    """
    Calculate the shortest evacuation distance from a room to the nearest exit.
    
    This tool uses NetworkX to compute the shortest path through the circulation
    network (corridors, doors) from the room's centroid to available exits.
    
    Args:
        room_id: The ID of the room (e.g., 'R201')
        exit_ids: Optional list of exit IDs to consider. If None, uses all exits.
    
    Returns:
        Dictionary containing:
        - distance_m: Shortest distance in meters
        - path: List of nodes in the path
        - target_exit: ID of the nearest exit
        - compliant: Boolean indicating if distance <= 25m (CTE requirement)
    
    Example:
        >>> calculate_egress_distance("R201")
        {"distance_m": 18.5, "path": ["R201", "D201", "C201", "E201"], 
         "target_exit": "E201", "compliant": True}
    """
    # Implementation
    # ... NetworkX shortest path logic ...
    
    return {
        "distance_m": 18.5,
        "path": ["R201", "D201", "C201", "E201"],
        "target_exit": "E201",
        "compliant": True
    }
```

**Elementos clave**:
1. **@tool decorator**: Marca la función como tool
2. **Docstring detallado**: El LLM lo lee para entender cuándo usar la tool
3. **Type hints**: Ayudan al LLM a pasar argumentos correctos
4. **Returns estructurados**: Dict/JSON fácil de parsear
5. **Ejemplo de uso**: Clarifica el formato esperado

### Tipos de Tools Comunes

#### 1. Tools de Lectura (Read)

```python
@tool
def get_room_info(room_id: str) -> dict:
    """Get detailed information about a room."""
    pass

@tool
def list_all_doors() -> List[dict]:
    """List all doors in the project."""
    pass
```

#### 2. Tools de Cálculo (Compute)

```python
@tool
def calculate_area(room_id: str) -> float:
    """Calculate the area of a room using Shapely."""
    pass

@tool
def calculate_distance(point_a: tuple, point_b: tuple) -> float:
    """Calculate Euclidean distance between two points."""
    pass
```

#### 3. Tools de Verificación (Check)

```python
@tool
def check_door_width_compliance(door_id: str, min_mm: float) -> dict:
    """Check if a door meets minimum width requirement."""
    pass

@tool
def check_fire_rating(element_id: str, required_rating: str) -> dict:
    """Check if an element meets fire rating requirement."""
    pass
```

#### 4. Tools de Consulta Externa (Query)

```python
@tool
def query_normativa(question: str) -> str:
    """Query the building code knowledge base (RAG)."""
    pass

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    pass
```

### Tool Best Practices

#### ✅ DO

```python
@tool
def good_tool(param: str) -> dict:
    """
    Clear description of what the tool does.
    
    Args:
        param: Description of parameter
    
    Returns:
        Structured dict with clear keys
    """
    try:
        result = do_something(param)
        return {
            "success": True,
            "data": result,
            "message": "Operation completed"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Operation failed"
        }
```

#### ❌ DON'T

```python
@tool
def bad_tool(x):
    """Does stuff."""  # Vago
    # Sin type hints
    # Sin manejo de errores
    return x * 2  # Retorno no estructurado
```

### Tools con Side Effects

```python
@tool
def save_verification_result(result: dict) -> bool:
    """
    Save a verification result to the database.
    
    CAUTION: This tool has side effects (writes to DB).
    
    Args:
        result: Dict with verification data
    
    Returns:
        True if saved successfully
    """
    # Write to DB
    db.insert("verifications", result)
    return True
```

⚠️ **Importante**: Tools con side effects deben:
- Documentar claramente el side effect
- Ser idempotentes si es posible
- Logear todas las operaciones

---

## Ejemplos Paso a Paso

### Ejemplo 1: Verificación Simple

**Task**: Verificar si la puerta D201 cumple ancho mínimo

**Iteraciones**:

```
=== Iteration 1 ===
Messages: [HumanMessage("Check if door D201 width is >= 800mm")]

THOUGHT: "Necesito primero obtener la información de la puerta D201"

ACTION: get_door_info("D201")

OBSERVATION: {"id": "D201", "width_mm": 900, "mark": "D-2.01"}

---

=== Iteration 2 ===
Messages: [
    HumanMessage("Check if door D201 width is >= 800mm"),
    AIMessage(tool_calls=[{"name": "get_door_info", "args": {"door_id": "D201"}}]),
    ToolMessage("{"id": "D201", "width_mm": 900, ...}")
]

THOUGHT: "La puerta tiene 900mm de ancho. Necesito verificar contra el requisito de 800mm"

ACTION: check_door_width_compliance("D201", 800)

OBSERVATION: {"compliant": True, "actual_mm": 900, "required_mm": 800, "difference": 100}

---

=== Iteration 3 ===
THOUGHT: "La puerta cumple (900mm > 800mm). Tengo suficiente información para responder"

FINAL ANSWER: "✅ La puerta D201 CUMPLE con el ancho mínimo requerido.
- Ancho actual: 900mm
- Ancho requerido: 800mm
- Excedente: 100mm"
```

### Ejemplo 2: Multi-Step Reasoning

**Task**: Verificar distancia de evacuación de room R201

```
=== Iteration 1 ===
THOUGHT: "Para verificar evacuación, necesito:
1. Saber qué salidas hay disponibles
2. Calcular distancia desde R201 a cada salida
3. Consultar normativa sobre distancia máxima permitida"

ACTION: list_all_exits()

OBSERVATION: [{"id": "E201", "type": "staircase"}, {"id": "E202", "type": "door"}]

---

=== Iteration 2 ===
THOUGHT: "Hay 2 salidas. Ahora calculo la distancia desde R201"

ACTION: calculate_egress_distance("R201")

OBSERVATION: {"distance_m": 18.5, "target_exit": "E201", "path": ["R201", "D201", "C201", "E201"]}

---

=== Iteration 3 ===
THOUGHT: "La distancia es 18.5m. ¿Cuál es el máximo permitido por CTE?"

ACTION: query_normativa("¿Distancia máxima de evacuación con múltiples salidas?")

OBSERVATION: "Según CTE DB-SI Art. 3.3.2, cuando existan dos o más salidas alternativas, 
la distancia máxima hasta alguna salida no debe exceder de 25 metros."

---

=== Iteration 4 ===
THOUGHT: "18.5m < 25m, por lo tanto cumple. Tengo toda la información"

FINAL ANSWER: "✅ Room R201 CUMPLE con requisitos de evacuación.
- Distancia a salida más cercana: 18.5m
- Límite CTE DB-SI Art. 3.3.2: 25m (con múltiples salidas)
- Margen: 6.5m
- Ruta: R201 → D201 → C201 → E201"
```

### Ejemplo 3: Error Recovery

**Task**: Verificar puerta inexistente

```
=== Iteration 1 ===
THOUGHT: "Necesito info de la puerta D999"

ACTION: get_door_info("D999")

OBSERVATION: {"error": "Door D999 not found"}

---

=== Iteration 2 ===
THOUGHT: "La puerta D999 no existe. Debería verificar qué puertas están disponibles"

ACTION: list_all_doors()

OBSERVATION: [{"id": "D201"}, {"id": "D202"}, {"id": "D203"}]

---

=== Iteration 3 ===
THOUGHT: "Las puertas disponibles son D201, D202, D203. D999 no está en el proyecto"

FINAL ANSWER: "❌ No se puede verificar la puerta D999 porque no existe en el proyecto.
Puertas disponibles: D201, D202, D203.
¿Quizás quisiste decir alguna de estas?"
```

---

## Best Practices

### 1. Limitar Iteraciones

```python
def agent_node(state: AgentState):
    iterations = state.get("iterations", 0)
    
    # Límite de seguridad
    if iterations >= 15:
        return {
            "messages": [AIMessage(content="MAX_ITERATIONS - Stopping to prevent infinite loop")],
            "iterations": iterations + 1
        }
    
    # ... rest of logic
```

### 2. System Message Claro

```python
system_message = """You are a building compliance verification agent.

Your task: Verify if a project meets building code requirements.

RULES:
1. Always use tools to get accurate data - DO NOT make assumptions
2. Cite specific code articles when stating requirements
3. Be systematic - check one element at a time
4. If you can't find information, say so clearly
5. Format your final answer with clear ✅/❌ indicators

Available tools:
{tools_description}

Work step by step."""
```

### 3. Structured Output

```python
@tool
def check_compliance(element_id: str) -> dict:
    """Always return structured results"""
    return {
        "element_id": element_id,
        "compliant": True,
        "checks": [
            {"criterion": "Width", "pass": True, "details": "900mm >= 800mm"},
            {"criterion": "Fire Rating", "pass": False, "details": "Missing EI60"}
        ],
        "summary": "Partial compliance - missing fire rating"
    }
```

### 4. Logging para Debugging

```python
def agent_node(state: AgentState):
    messages = state["messages"]
    iterations = state["iterations"]
    
    # Log antes de invocar LLM
    logger.info(f"Iteration {iterations}: Invoking LLM with {len(messages)} messages")
    
    response = llm_with_tools.invoke(messages)
    
    # Log después
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
        for tc in response.tool_calls:
            logger.info(f"  - {tc['name']}({tc['args']})")
    else:
        logger.info("LLM provided final answer")
    
    return {"messages": [response], "iterations": iterations + 1}
```

### 5. Graceful Degradation

```python
@tool
def robust_tool(param: str) -> dict:
    """Tool with error handling"""
    try:
        result = expensive_operation(param)
        return {"success": True, "data": result}
    
    except SpecificError as e:
        logger.warning(f"Expected error: {e}")
        return {
            "success": False,
            "error": "recoverable",
            "message": "Try with different parameters",
            "suggestion": "Use list_available_items() first"
        }
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "success": False,
            "error": "critical",
            "message": "Tool failed unexpectedly",
            "details": str(e)
        }
```

---

## Debugging

### Imprimir Razonamiento del Agente

```python
for i, state in enumerate(app.stream(initial_state), 1):
    print(f"\n{'='*60}")
    print(f"ITERATION {i}")
    print('='*60)
    
    if "agent" in state:
        last_msg = state["agent"]["messages"][-1]
        
        if hasattr(last_msg, 'content'):
            print(f"\n💭 THOUGHT:\n{last_msg.content[:200]}...")
        
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            print(f"\n🔧 ACTIONS:")
            for tc in last_msg.tool_calls:
                print(f"  - {tc['name']}({tc['args']})")
    
    if "tools" in state:
        for msg in state["tools"]["messages"]:
            if isinstance(msg, ToolMessage):
                print(f"\n👁️ OBSERVATION:\n{msg.content[:200]}...")
```

### Callback Handler

```python
from langchain.callbacks.base import BaseCallbackHandler

class DebugCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🔧 Tool Start: {serialized.get('name')}")
        print(f"   Input: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"✅ Tool End: Output = {output[:100]}...")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🤖 LLM Start")
    
    def on_llm_end(self, response, **kwargs):
        print(f"🤖 LLM End: {len(response.generations)} generations")

# Usar
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    callbacks=[DebugCallbackHandler()]
)
```

---

## Comparación con Otros Frameworks

| Framework | Tipo | Pros | Cons | Uso |
|-----------|------|------|------|-----|
| **ReAct (LangGraph)** | Agentic | ✅ Flexible, transparente | Setup complejo | ⭐ POC, custom |
| **Chain of Thought (CoT)** | Prompting | Simple, sin tools | Solo reasoning | Problemas simples |
| **Function Calling** | Native | Rápido, integrado | Menos control | Tareas específicas |
| **MRKL / ReAct (LangChain)** | Deprecated | Legacy | Menos flexible | Legacy code |
| **CrewAI** | Multi-agent | Colaboración entre agentes | Opinado, overhead | Multi-agent systems |
| **AutoGen** | Multi-agent | Features avanzadas | Muy complejo | Research, complex |

---

## Recursos Adicionales

- **Paper ReAct**: https://arxiv.org/abs/2210.03629
- **LangGraph Tutorial**: https://langchain-ai.github.io/langgraph/tutorials/introduction/
- **Tool Use Guide**: https://python.langchain.com/docs/modules/agents/tools/
- **Function Calling**: https://ai.google.dev/docs/function_calling

---

**Versión**: 1.0  
**Próxima revisión**: Post-POC
