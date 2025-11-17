# Git Branch Structure and Merge History

This document visualizes the branch structure and merge history of the aec-compliance-agent repository, with **main-2.0** as the central integration branch.

## ASCII Branch Flow Diagram

```
                    ┌─────────────────────────┐
                    │   feature/pilar-1      │
                    │   feature/pilar-2      │
                    │   feature/pilar-3-rag  │
                    │   feature/pillar-4     │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │       develop           │
                    └─────────────────────────┘
                                │
                                │
                    ┌───────────┴─────────────┐
                    │                         │
                    ▼                         ▼
        ┌──────────────────┐      ┌──────────────────────┐
        │      main        │      │     main-2.0         │ ◄─── CENTRAL BRANCH
        └──────────────────┘      └──────────┬───────────┘
                                             │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
        ┌──────────────────────┐  ┌──────────────────┐  ┌────────────────────┐
        │ clean-display-helper │  │element-data-extr.│  │ rag-implementation │
        │      (PR #5)         │  │     (PR #7)      │  │     (PR #10)       │
        └──────────┬───────────┘  └────────┬─────────┘  └─────────┬──────────┘
                   │                       │                       │
                   └───────────────────────┴───────────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   main-2.0      │
                                    │  (integrated)   │
                                    └─────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │     agent-langchain-minimal                 │
                    │         (base branch)                        │
                    └───────────────┬─────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌──────────────────┐  ┌──────────────┐  ┌──────────────────────┐
        │agent-memory-design│  │task-graph-   │  │agent-memory-         │
        │                  │  │formalization │  │summarization         │
        └──────────────────┘  └──────┬───────┘  └──────────┬───────────┘
                                      │                     │
                                      │  (PR #12)           │  (PR #14)
                                      │                     │
                                      └──────────┬──────────┘
                                                 │
                                                 ▼
                                    ┌──────────────────────┐
                                    │  agent-reasoning     │
                                    │  (current branch)    │
                                    │  ⚠️ NOT MERGED YET    │
                                    └──────────────────────┘
                                    
Recommended Merge Path:
  agent-reasoning → agent-langchain-minimal → main-2.0
```

## Current State & Merge Status

**Important**: Neither `agent-langchain-minimal` nor `agent-reasoning` have been merged into `main-2.0` yet.

### Branch Relationships

- **main-2.0** (commit `5376e29`): Latest is "Add RAG document retrieval system with Gemini File Search (#10)"
- **agent-langchain-minimal** (commit `eab9589`): Branched from main-2.0, adds 5 commits (LangChain agent, memory system, tracing)
- **agent-reasoning** (commit `75b16d9`): Built on top of `agent-langchain-minimal`, adds ~22 additional commits (reasoning system, guardrails, task graph, memory summarization)

### Recommended Merge Strategy

Since `agent-reasoning` already contains all commits from `agent-langchain-minimal`, you have three options:

#### Option 1: Consolidate Then Integrate (Recommended)
```
Step 1: agent-reasoning → agent-langchain-minimal (merge)
Step 2: agent-langchain-minimal → main-2.0 (merge)
```
- **Pros**: 
  - Maintains `agent-langchain-minimal` as the consolidated "agent development" branch
  - Clean separation: agent work stays in agent-langchain-minimal, then integrates to main-2.0
  - Easier to track agent-related changes in one branch
- **Cons**: Two-step process
- **When to use**: When you want to keep agent-langchain-minimal as the primary agent branch

#### Option 2: Direct Merge
```
agent-reasoning → main-2.0
```
- **Pros**: Single merge, includes all features at once
- **Cons**: Bypasses agent-langchain-minimal, which might leave it as an orphaned branch
- **When to use**: When you're ready to integrate the complete reasoning system and don't need to maintain agent-langchain-minimal

#### Option 3: Incremental Integration
```
Step 1: agent-langchain-minimal → main-2.0
Step 2: agent-reasoning → main-2.0
```
- **Pros**: Incremental integration, easier to review in stages
- **Cons**: Two separate merges, and agent-reasoning will need to rebase/merge main-2.0 after step 1
- **When to use**: If you want to stabilize the LangChain base first, then add reasoning features

### Current Branch Status

```
main-2.0 (stable, last updated: PR #10)
  │
  ├─→ agent-langchain-minimal (5 commits ahead, NOT merged)
  │     │
  │     └─→ agent-reasoning (22+ commits ahead, NOT merged)
  │
  └─→ [Other merged branches: PR #5, PR #7, PR #10]
```

## Interactive Mermaid Diagram

```mermaid
gitgraph
    commit id: "Initial"
    
    branch develop
    commit id: "Base"
    
    branch feature/pilar-1-extraction
    commit id: "Pilar 1"
    checkout develop
    merge feature/pilar-1-extraction id: "PR #1"
    
    branch feature/pilar-2-calculations
    commit id: "Pilar 2"
    checkout develop
    merge feature/pilar-2-calculations id: "PR #2"
    
    branch feature/pilar-3-rag
    commit id: "Pilar 3 RAG"
    checkout develop
    merge feature/pilar-3-rag id: "Merge to develop"
    
    branch feature/pillar-4-agent
    commit id: "Pillar 4"
    checkout develop
    merge feature/pillar-4-agent id: "Merge to develop"
    
    checkout main
    branch main-2.0
    commit id: "Clean restart"
    
    branch feature/clean-display-helper
    commit id: "Display helper"
    checkout main-2.0
    merge feature/clean-display-helper id: "PR #5"
    
    branch element-data-extraction
    commit id: "IFC converter"
    checkout main-2.0
    merge element-data-extraction id: "PR #7"
    
    commit id: "Merge building data toolkit"
    
    branch rag-implementation
    commit id: "RAG system"
    checkout main-2.0
    merge rag-implementation id: "PR #10"
    
    branch agent-langchain-minimal
    commit id: "LangChain agent"
    
    branch agent-memory-design
    checkout agent-langchain-minimal
    commit id: "Memory system"
    
    branch task-graph-formalization
    commit id: "TaskGraph"
    
    branch agent-memory-summarization
    commit id: "Memory summarization"
    
    branch agent-reasoning
    checkout agent-langchain-minimal
    commit id: "Reasoning base"
    commit id: "Guardrails"
    commit id: "JIT planning"
    checkout task-graph-formalization
    commit id: "TaskGraph impl"
    checkout agent-reasoning
    merge task-graph-formalization id: "PR #12"
    commit id: "Single-tool constraint"
    checkout agent-memory-summarization
    commit id: "Memory phases"
    checkout agent-reasoning
    merge agent-memory-summarization id: "PR #14"
    commit id: "Current"
```

## Detailed Branch Structure

### Central Branch: main-2.0

**main-2.0** serves as the primary integration branch for the Agent 2.0 architecture.

#### Merged Branches into main-2.0:

1. **feature/clean-display-helper** → main-2.0 (PR #5)
   - Clean up display helper and enhance agent tools
   - Foundation for clean agent architecture

2. **element-data-extraction** → main-2.0 (PR #7)
   - IFC to JSON converter
   - Building element extraction system

3. **rag-implementation** → main-2.0 (PR #10)
   - RAG document retrieval system with Gemini File Search
   - Complete RAG implementation

4. **Building Data Toolkit** (direct merge)
   - Merged building data toolkit functionality

### Development Branch: develop

The **develop** branch was used for early feature development and integration:

1. **feature/pilar-1-extraction** → develop (PR #1)
   - CAD Data Extraction System

2. **feature/pilar-2-calculations** → develop (PR #2)
   - Fundamental calculation functions

3. **feature/pilar-3-rag** → develop
   - RAG documentation and system

4. **feature/pillar-4-agent** → develop
   - Comprehensive testing suite

### Active Feature Branches

#### agent-reasoning (Current Branch)

The **agent-reasoning** branch builds on **agent-langchain-minimal** and includes:

1. **task-graph-formalization** → agent-reasoning (PR #12)
   - TaskGraph-based dependency management system

2. **agent-memory-summarization** → agent-reasoning (PR #14)
   - Memory Summarization Implementation
   - Goal-Based Session Lifecycle

#### Related Branches

- **agent-langchain-minimal**: Base LangChain agent implementation
- **agent-memory-design**: Memory system design (merged into agent-langchain-minimal)
- **task-graph-formalization**: Task graph dependency system
- **agent-memory-summarization**: Memory summarization features

## Branch Relationships

```
main-2.0 (Central Integration)
├── feature/clean-display-helper (PR #5)
├── element-data-extraction (PR #7)
├── rag-implementation (PR #10)
└── building-data-toolkit (direct merge)

develop (Early Development)
├── feature/pilar-1-extraction (PR #1)
├── feature/pilar-2-calculations (PR #2)
├── feature/pilar-3-rag
└── feature/pillar-4-agent

agent-reasoning (Current Active)
├── agent-langchain-minimal (base)
│   └── agent-memory-design
├── task-graph-formalization (PR #12)
└── agent-memory-summarization (PR #14)
```

## Merge Timeline

1. **Early Phase**: Feature branches merged into `develop`
   - Pilar 1, 2, 3, 4 features

2. **Agent 2.0 Foundation**: `main-2.0` created with clean architecture
   - Clean restart from previous codebase

3. **Core Features Merged to main-2.0**:
   - Display helper (PR #5)
   - Element data extraction (PR #7)
   - RAG implementation (PR #10)

4. **Reasoning System Development**: `agent-reasoning` branch (⚠️ NOT YET MERGED)
   - Built on agent-langchain-minimal
   - Integrated task-graph-formalization (PR #12)
   - Integrated agent-memory-summarization (PR #14)
   - **Status**: Ready for merge into main-2.0 when ready

## Notes

- **main-2.0** is the stable integration branch for Agent 2.0 architecture
- **develop** was used for early feature development but is now separate
- **agent-reasoning** is the current active development branch
- **⚠️ Pending Merges**: Neither `agent-langchain-minimal` nor `agent-reasoning` have been merged into `main-2.0` yet
- Most merges were done via Pull Requests for code review
- The repository follows a feature branch workflow with main-2.0 as the central hub

## Next Steps for Integration

To complete the integration of the reasoning system into main-2.0 (recommended approach):

1. **Review and test** `agent-reasoning` branch thoroughly
2. **Step 1: Merge agent-reasoning → agent-langchain-minimal**
   - Create Pull Request from `agent-reasoning` to `agent-langchain-minimal`
   - This consolidates all agent work into `agent-langchain-minimal`
   - After merge, `agent-langchain-minimal` will contain all reasoning features
3. **Step 2: Merge agent-langchain-minimal → main-2.0**
   - Create Pull Request from `agent-langchain-minimal` to `main-2.0`
   - This integrates the complete agent system into the main branch
4. **After both merges**: 
   - `agent-langchain-minimal` becomes the consolidated agent branch
   - `agent-reasoning` can be kept for reference or deleted (its work is now in agent-langchain-minimal)
   - Future agent work can branch from `agent-langchain-minimal`

