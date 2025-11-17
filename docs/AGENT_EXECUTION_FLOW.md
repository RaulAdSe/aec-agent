# Agent Execution Flow

Complete step-by-step documentation of the autonomous reasoning agent execution process.

## Overview

The agent follows a **ReAct-enhanced pattern** with **fully LLM-powered reasoning** at every decision point except tool execution itself.

```
User Goal → LLM Analysis → LLM Planning → Tool Execution → LLM Validation → Results
```

## Complete Execution Flow

### 🚀 Entry Point
```
User Input: "Count doors in data/out/FM-ARC_v2.json"
↓
main.py reason command → ReasoningAgent.process_goal()
```

### 📋 Phase 1: Setup & Initialization (~1s)

1. **Memory System Initialization**
   - Session tracking with unique session ID
   - Conversation history setup
   - Building data context tracking

2. **LangSmith Tracing Setup**
   - Project: "AEC-Reasoning-Agent"
   - Full observability enabled
   - Nested trace structure

3. **LLM Initialization**
   - Model: gpt-4o-mini
   - Temperature: 0.1 (deterministic)
   - Max tokens: 4000

4. **Tool Registry Setup**
   - 8 building analysis tools
   - 1 compliance search tool
   - JSON wrapper functions

5. **Reasoning Components Initialization**
   - GoalDecomposer (with LLM)
   - ToolPlanner (with LLM)
   - ToolExecutor (deterministic)
   - ResultValidator (with LLM)

### 🧠 Phase 2: Analyze & Plan (~16s)

#### Step 2.1: Goal Decomposition (LLM Call #1 - 9.3s)

**Input to LLM:**
```
System: You are an expert AEC compliance analyst...
Human: Goal to decompose: Count doors in data/out/FM-ARC_v2.json
```

**LLM Output:**
```json
[
  {"name": "Load building data", "description": "Load and validate building data from the specified file", "priority": "HIGH"},
  {"name": "Retrieve door elements", "description": "Use the get_all_elements tool to retrieve all door elements", "priority": "MEDIUM"},
  {"name": "Count door elements", "description": "Count the total number of door elements", "priority": "MEDIUM"},
  {"name": "Validate door count", "description": "Validate the door count results", "priority": "MEDIUM"},
  {"name": "Document door count", "description": "Document the final door count", "priority": "LOW"}
]
```

**Result:** 5 intelligent tasks created with dependencies

#### Step 2.2: Tool Planning (LLM Calls #2-6 - 6.7s total)

**For each task, sequential LLM calls:**

**Task 1: "Load building data"**
```
LLM Input: Task + Available tools + Context
LLM Output: "load_building_data"
Result: ✅ Correct tool selection
```

**Task 2: "Retrieve door elements"**
```
LLM Input: Task + Available tools + Context
LLM Output: "load_building_data"
Result: ❌ Wrong tool (should be "get_all_elements")
```

**Task 3: "Count door elements"**
```
LLM Input: Task + Available tools + Context
LLM Output: "load_building_data" 
Result: ❌ Wrong tool (should be "calculate_metrics")
```

**Task 4: "Validate door count"**
```
LLM Input: Task + Available tools + Context
LLM Output: Invalid response
Fallback: Pattern matching → "search_compliance_documents"
Result: ⚠️ Fallback used
```

**Task 5: "Document door count"**
```
LLM Input: Task + Available tools + Context
LLM Output: "load_building_data"
Result: ❌ Wrong tool (should be custom summary)
```

### ⚙️ Phase 3: Execution Loop (~3s)

#### Iteration 1: Execute Task 1

**Step 3.1: Tool Execution (6.5ms)**
```
Tool: load_building_data
Input: data/out/FM-ARC_v2.json (extracted from goal)
Action: Load JSON file, parse building data
Output: {"status": "success", "data": {...1026 elements...}}
Execution Time: 6.5ms
```

**Step 3.2: Result Validation (LLM Call #7 - 3.0s)**
```
LLM Input:
- Tool: load_building_data
- Task: Load building data
- Output Status: success
- Output Preview: {"status": "success", "data": {...}}
- Execution Time: 0.0065s

LLM Output: {
  "success": true,
  "message": "Tool executed successfully, building data loaded",
  "confidence": 0.95,
  "issues": []
}
```

**Result:** ✅ Task 1 completed successfully (1/5 tasks)

#### Iteration 2: Execute Task 2

**Step 3.3: Tool Execution (Failed - 0ms)**
```
Tool: load_building_data (wrong tool for this task!)
Expected Tool: get_all_elements
Issue: Task needs element retrieval, not file loading
Error: "No file path found in task: Retrieve door elements"
```

**Step 3.4: Basic Validation (0ms)**
```
Basic Check: Tool execution failed
No LLM call - immediate failure detection
Result: ❌ Task 2 failed
```

**Result:** ❌ Task 2 failed, remaining tasks blocked

### 🔄 Phase 4: Completion (0ms)

**Progress Assessment:**
- Total tasks: 5
- Completed: 1
- Failed: 1  
- Pending: 3 (blocked by dependencies)
- Progress: 20%
- Goal achieved: False

**Final Results:**
```json
{
  "status": "partial",
  "message": "Goal partially achieved", 
  "execution_time": 19.03,
  "tasks_completed": "1/5"
}
```

## Execution Summary

### Timing Breakdown
| Phase | Component | Type | Duration | Percentage |
|-------|-----------|------|----------|------------|
| Setup | System | Non-LLM | 1.0s | 5% |
| Goal Decomposition | GoalDecomposer | **LLM** | 9.3s | 49% |
| Tool Planning (5x) | ToolPlanner | **LLM** | 6.7s | 35% |
| Tool Execution | ToolExecutor | Tool | 6.5ms | 0.03% |
| Result Validation | ResultValidator | **LLM** | 3.0s | 16% |
| **Total** | | | **19.03s** | **100%** |

### LLM Usage Analysis
- **Total LLM calls:** 7
- **LLM reasoning time:** 19.0s (99.97%)
- **Tool execution time:** 6.5ms (0.03%)
- **LLM/Tool ratio:** 2923:1

### Architecture Pattern
```
ReAct Enhanced with Multi-Layer LLM Reasoning:

Query → 🧠 Thinking (LLM Goal Decomposition)
      → 🧠 Thinking (LLM Tool Planning) × 5
      → ⚙️ Action (Tool Execution)
      → 🧠 Observation (LLM Validation)
      → 📋 Answer/Finish
```

## Key Observations

### Strengths
1. **Truly LLM-powered reasoning** - Every decision uses AI
2. **Comprehensive tracing** - Full observability in LangSmith
3. **Intelligent task breakdown** - 5 contextual tasks vs generic patterns
4. **Robust error handling** - Multiple fallback mechanisms

### Areas for Improvement
1. **Tool selection accuracy** - LLM chose wrong tools for 3/5 tasks
2. **Sequential planning overhead** - 5 LLM calls before execution
3. **Dependency management** - Failed tasks block entire pipeline
4. **Context passing** - File paths not properly propagated

### Performance Characteristics
- **Fast tool execution** (6.5ms) - Efficient building data processing
- **Slow reasoning** (19s) - Comprehensive LLM analysis
- **High LLM dependence** - 99.97% of time spent in reasoning
- **Deterministic tools only** - As requested by user

## Trace Structure in LangSmith

```
reasoning_agent_process (19.0s)
├── autonomous_reasoning_process (19.0s)
    ├── analyze_and_plan_phase (16.0s)
    │   ├── llm_goal_decomposition (9.3s)
    │   │   └── ChatOpenAI call (9.2s)
    │   └── tool_planning (6.7s)
    │       ├── llm_tool_planning (1.3s) - Task 1
    │       ├── llm_tool_planning (1.2s) - Task 2  
    │       ├── llm_tool_planning (0.9s) - Task 3
    │       ├── llm_tool_planning (1.2s) - Task 4
    │       └── llm_tool_planning (0.9s) - Task 5
    └── execute_reasoning_loop (3.0s)
        ├── execute_single_task (3.0s)
        │   ├── tool_execution (0.006s)
        │   └── llm_result_validation (3.0s)
        └── execute_single_task (0.0s) - Failed immediately
```

This architecture successfully achieves **100% LLM-powered reasoning** while keeping tool execution deterministic and fast.