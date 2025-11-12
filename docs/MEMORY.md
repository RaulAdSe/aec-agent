# Memory Management System

The AEC Compliance Agent uses a two-layer memory system that combines LangChain's conversation memory with custom session state tracking, optimized for complex AEC compliance workflows.

## Architecture Overview

The memory system consists of three main components:

1. **Short-Term Memory** - Recent conversation history with auto-summarization
2. **Session Memory** - Structured state tracking (goals, files, tasks, building data)
3. **Memory Manager** - Unified coordinator that combines both layers

```
┌─────────────────────────────────────────────────────────┐
│                    Memory Manager                        │
│  (Unified interface for agent integration)               │
└──────────────┬──────────────────────┬──────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌───────▼────────────┐
    │  Short-Term Memory   │  │  Session Memory    │
    │  (Conversation)      │  │  (State Tracking)  │
    │                      │  │                    │
    │  - Recent messages   │  │  - Goals           │
    │  - Auto-summaries    │  │  - Active files    │
    │  - LangChain based   │  │  - Subtasks        │
    │                      │  │  - Tool history    │
    │                      │  │  - Building data   │
    │                      │  │    (summary only)  │
    └──────────────────────┘  └────────────────────┘
```

## Components

### 1. Short-Term Memory (`short_term_memory.py`)

Manages recent conversation history using LangChain's hybrid approach:

- **ConversationBufferWindowMemory** - Keeps recent messages in full
- **ConversationSummaryMemory** - Auto-summarizes older messages
- Configurable window sizes and token limits
- Prevents context window overflow while maintaining conversation flow

**What it stores:**
- Recent user queries and agent responses
- Summarized older conversation history
- Conversation metadata (timestamps, message counts)

### 2. Session Memory (`session_memory.py`)

Tracks structured session state throughout the agent's lifecycle:

**What it stores:**
- **Session goals** - Main objective for the session
- **Active files** - Files currently being worked on
- **Modified files** - Files changed in this session
- **Subtasks** - Task breakdown with status tracking
- **Tool execution history** - Record of all tool calls
- **Building data context** - Summary of loaded building data (NOT full JSON)
- **Compliance rules** - Active compliance rules being checked
- **Analysis results** - Latest analysis outcomes

**Persistence:**
- JSON-based persistence with timestamps
- Session can be resumed by session_id
- Full session history maintained

### 3. Memory Manager (`memory_manager.py`)

Unified coordinator that:

- Combines short-term and session memory
- Provides LangChain-compatible memory variables
- Handles auto-save functionality
- Formats context for LLM prompt injection
- Manages memory lifecycle (clear, update, retrieve)

## Building Data Handling

### Key Design Principle: Full Data to Agent, Summary to Memory

The system is designed to handle large JSON building data files efficiently:

1. **Full JSON data** is returned to the agent when `load_building_data()` is called
2. **Only summary** is stored in memory to keep context window lean
3. **Full data** remains cached in the tool layer (`_building_data` global variable)

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Agent calls: load_building_data("building.json")          │
└───────────────────────┬─────────────────────────────────────┘
                        │
            ┌───────────▼───────────┐
            │  _load_building_data_ │
            │  _wrapper()           │
            └───────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│ Loads full   │ │ Extracts    │ │ Returns to │
│ JSON into    │ │ summary     │ │ agent:     │
│ _building_   │ │ (counts,    │ │ - Full JSON│
│ data cache   │ │ project     │ │ - Summary │
│              │ │ name, etc.) │ │            │
└──────────────┘ └──────┬──────┘ └────────────┘
                        │
            ┌───────────▼───────────┐
            │  Memory stores ONLY:  │
            │  - File path          │
            │  - Project name       │
            │  - Total elements     │
            │  - Element counts:    │
            │    "5 stairs, 2 walls"│
            │  (NOT full JSON)      │
            └───────────────────────┘
```

### What the Agent Receives

When `load_building_data()` is called, the agent receives:

```python
{
    "status": "success",
    "data": {
        # Full JSON building data - agent can read everything
        "file_info": {...},
        "spaces": [...],  # All space elements
        "doors": [...],   # All door elements
        "walls": [...],   # All wall elements
        # ... etc
    },
    "summary": {
        # Quick reference summary
        "file_loaded": "/path/to/building.json",
        "project_name": "Office Building",
        "total_elements": 1250,
        "available_element_types": {
            "spaces": 45,
            "doors": 23,
            "stairs": 5,
            "walls": 2,
            "slabs": 10
        }
    },
    "logs": ["Loaded building data from ..."]
}
```

### What Goes Into Memory

Only the summary is stored in memory context:

```
Loaded Building Data: /path/to/building.json
Building Context: Office Building with 1,250 total elements (45 spaces, 23 doors, 5 stairs, 2 walls, 10 slabs)
```

**Why this design?**
- ✅ Agent can read full JSON when needed (in tool response)
- ✅ Memory stays lean (only metadata, not full data)
- ✅ Context window doesn't overflow with large JSON files
- ✅ Agent remains aware of what's loaded and available
- ✅ Full data cached in tool layer for fast subsequent queries

### Data Caching

The full JSON data is cached in memory at the tool layer:

- **Global variable**: `_building_data` in `building_data_toolkit.py`
- **Purpose**: Fast repeated queries without re-reading from disk
- **Lifecycle**: Persists for the session, cleared when needed
- **Size**: Typically 500KB-2MB per building file (efficient for modern systems)

## Memory Context in LLM Prompts

The memory system injects context into every LLM prompt:

```
=== Session Context ===
Session Goal: Analyze building compliance for fire safety
Current Task: Check fire door requirements (Status: in_progress)
Active Files: building.json, compliance_rules.pdf
Loaded Building Data: /path/to/building.json
Building Context: Office Building with 1,250 total elements (45 spaces, 23 doors, 5 stairs, 2 walls, 10 slabs)
Recent Tools: load_building_data, query_elements

=== Conversation History ===
[Recent conversation messages and summaries...]
```

## Usage Examples

### Basic Usage

```python
from aec_agent.agent import create_agent

# Create agent with memory enabled
agent = create_agent(enable_memory=True)

# Set session goal
agent.set_session_goal("Analyze building compliance for fire safety")

# Add subtasks
task_id = agent.add_subtask("Load building IFC data")

# Use agent normally - memory automatically tracks everything
response = agent.process("Load the building data from project.json")

# Later queries benefit from memory
response = agent.process("How many fire doors are in the building?")
# Agent knows: building is already loaded, can query directly

# Access memory summary
summary = agent.get_memory_summary()
```

### Building Data Workflow

```python
# 1. Load building data
response = agent.process("Load building data from office_tower.json")
# → Full JSON returned to agent
# → Summary stored in memory: "Office Building with 1,250 elements (45 spaces, 23 doors...)"

# 2. Query elements (uses cached data, no re-load needed)
response = agent.process("How many fire-rated doors are there?")
# → Agent knows building is loaded from memory context
# → Calls query_elements() which uses cached _building_data

# 3. Complex analysis
response = agent.process("Check compliance for all exit routes")
# → Agent maintains context across multiple tool calls
# → Memory tracks progress and results
```

### Memory Management

```python
# Get memory statistics
stats = agent.get_memory_usage_stats()

# Get active files
active_files = agent.get_active_files()

# Get pending subtasks
pending = agent.get_pending_subtasks()

# Clear conversation memory (keeps session state)
agent.clear_conversation_memory()

# Clear all memory
agent.clear_all_memory()
```

## Best Practices

### 1. Set Session Goals Early
```python
agent.set_session_goal("Analyze building compliance for fire safety")
```
This helps the agent maintain focus throughout the session.

### 2. Use Subtasks for Complex Workflows
```python
task1 = agent.add_subtask("Load building data")
task2 = agent.add_subtask("Analyze fire doors", dependencies=[task1])
```
The agent tracks progress and dependencies.

### 3. Let Memory Track Tool Usage
The system automatically tracks all tool executions. You don't need to manually log them.

### 4. Trust the Building Data Caching
- Load once, query many times
- The system efficiently caches JSON data
- Memory only stores summaries, not full data
- Agent receives full data when needed

### 5. Use Memory Context for Context-Aware Queries
The agent automatically knows:
- What files are loaded
- What's been analyzed
- Current task status
- Recent tool usage

## Configuration

### Memory Manager Configuration

```python
from aec_agent.memory import MemoryManagerConfig, ShortTermMemoryConfig

config = MemoryManagerConfig(
    short_term_config=ShortTermMemoryConfig(
        window_size=10,  # Keep last 10 messages
        max_token_limit=2000  # Summarize when exceeded
    ),
    session_persistence_path=Path("./sessions"),
    auto_save_interval=10,  # Auto-save every 10 operations
    enable_persistence=True
)

agent = create_agent(enable_memory=True, memory_config=config)
```

## Testing

Comprehensive test suite in `tests/test_memory_system.py`:

- Unit tests for each memory component
- Integration tests with agent functionality
- Persistence and serialization validation
- Building data context tracking tests

## Summary

The memory system provides:

✅ **Short-term conversation memory** - Recent messages + auto-summarization  
✅ **Session state tracking** - Goals, files, tasks, tool history  
✅ **Building data awareness** - Knows what's loaded without storing full JSON  
✅ **Context injection** - Automatic prompt enhancement with memory  
✅ **Persistence** - Sessions can be saved and resumed  
✅ **LangChain integration** - Seamless compatibility with LangChain agents  

Perfect for complex AEC compliance workflows where the agent needs to:
- Remember what building data is loaded
- Track analysis progress across multiple queries
- Maintain context about compliance rules and results
- Work efficiently with large JSON building files
