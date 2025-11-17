# Memory Management System

## Overview

The AEC Compliance Agent features a sophisticated three-layer memory management system designed for long-running analysis sessions. The system provides intelligent summarization, goal-based lifecycle management, and automatic compaction to prevent unbounded memory growth while preserving essential information.

## Architecture

### Three-Layer Memory Structure

```
┌─────────────────────────────────────────────────────────────┐
│                Memory Management System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: Short-Term Memory (Conversation)                  │
│  ├─ Buffer Window: Last N messages (full detail)            │
│  ├─ Summary Memory: LLM-based recursive summarization       │
│  ├─ Token Triggers: 4K conversation cap                     │
│  └─ Strategy: Async/background summarization                │
│                                                              │
│  Layer 2: Session Memory (Structured State)                 │
│  ├─ Goal-Based Lifecycle: Automatic archiving               │
│  ├─ Tool History: Statistical summarization                 │
│  ├─ Task Dependencies: DAG-based orchestration              │
│  ├─ Token Triggers: 12K session cap                         │
│  └─ Strategy: Rule-based statistical summary                │
│                                                              │
│  Layer 3: Execution Memory (Reactive State)                 │
│  ├─ Execution Steps: Per-task context tracking             │
│  ├─ Discovered Context: Strategic insights                  │
│  ├─ Performance Metrics: Success patterns                   │
│  └─ Lifecycle: Resets per task (100 step FIFO)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: Short-Term Memory (Conversation)

### Purpose
Manages conversational context between user and agent, providing recent interaction history and summarized conversation context.

### Components

#### Conversation Buffer
- **Window Size**: Configurable (default: 10 messages)
- **Content**: Recent user inputs and agent responses in full detail
- **Purpose**: Immediate context for ongoing conversation

#### Summary Memory
- **Strategy**: LLM-based recursive summarization using ConversationSummaryMemory
- **Triggers**: Token-based automatic summarization at 4K conversation cap
- **Content**: Compressed summaries of older conversation history
- **Model**: gpt-4o-mini for cost-effective summarization

### Token-Based Management

```python
class ShortTermMemory:
    def add_conversation_turn(self, user_input: str, ai_output: str) -> None:
        """Add conversation turn with automatic token-based summarization trigger."""
        # Add to buffer
        turn = ConversationTurn(user_input, ai_output, timestamp)
        self.conversation_buffer.append(turn)
        
        # Check token count and trigger summarization if needed
        current_tokens = self._estimate_conversation_tokens()
        
        if current_tokens > self.config.short_term_token_cap:  # 4K default
            self._trigger_async_summarization()
```

### Configuration Options

```python
class ShortTermMemoryConfig:
    window_size: int = 10  # Recent messages to keep
    max_token_limit: int = 2000  # Maximum tokens for summary
    enable_summarization: bool = True  # Enable LLM summarization
    summarization_strategy: str = "async"  # sync, async, background
    short_term_token_cap: int = 4000  # Trigger summarization threshold
    short_term_token_warning_threshold: int = 3000  # Warning level
```

### Performance Characteristics
- **Summarization Time**: 1-3 seconds (async, non-blocking)
- **Memory Reduction**: 60-80% compression of conversation history
- **Token Estimation**: Simple heuristic (~4 chars per token)
- **Cache Support**: Avoids redundant summarization of same content

## Layer 2: Session Memory (Structured State)

### Purpose
Maintains structured state across an entire analysis session, handling multiple goals with automatic lifecycle management and intelligent archiving.

### Goal-Based Lifecycle Management

#### Core Concept
Session memory organizes all data around goals, automatically archiving completed goals and keeping only active goal data in full detail.

```
Goal 1: "Count doors in building"
├─ tool_history: [exec1, exec2, exec3] (full detail)
├─ subtasks: [task1, task2, task3] (full detail)
├─ context: "Building has 3 floors, 45 rooms..."
├─ Status: COMPLETED
└─ Archive: Statistical summary created, detailed data removed

Goal 2: "Check fire compliance" 
├─ tool_history: [exec1, exec2] (full detail, fresh start)
├─ subtasks: [task1, task2] (full detail, fresh start)  
├─ context: "Fire exits: 4 main, 2 emergency..."
├─ Status: ACTIVE
└─ Keep all detailed data for active analysis
```

### Data Models

#### Goal Model
```python
class Goal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_text: str
    status: GoalStatus = GoalStatus.ACTIVE  # ACTIVE, COMPLETED, FAILED
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    # Summary statistics (created when goal completes)
    tool_executions_count: int = 0
    subtasks_count: int = 0
    success_rate: float = 1.0
    
    # Archived summaries (rule-based, no LLM needed)
    tool_history_summary: Optional[Dict[str, Any]] = None
    subtasks_summary: Optional[Dict[str, Any]] = None
    context_summary: Optional[str] = None
```

#### Tool Execution Tracking
```python
class ToolExecution(BaseModel):
    id: str
    goal_id: str  # Associates execution with specific goal
    tool_name: str
    arguments: Dict[str, Any]
    result_summary: Optional[str]
    execution_time: datetime
    success: bool
    error_message: Optional[str]
```

### Automatic Archiving Process

#### When Goal Completes
1. **Create Statistical Summary**: Extract patterns from tool executions
2. **Summarize Subtasks**: Group completed tasks by themes
3. **Archive Context**: Keep first 500 characters as summary
4. **Remove Detailed Data**: Clear from active lists
5. **Store in Completed Goals**: Preserve summaries for reference

#### Statistical Summarization Example

**Before Archiving** (50 tool executions):
```python
tool_history = [
    ToolExecution(tool_name="load_building_data", success=True, ...),
    ToolExecution(tool_name="get_all_elements", success=True, ...),
    # ... 48 more executions
]
```

**After Archiving** (statistical summary):
```python
tool_history_summary = {
    "total_executions": 50,
    "success_rate": 0.94,
    "tool_usage": {
        "load_building_data": 5,
        "get_all_elements": 15, 
        "query_elements": 20,
        "calculate_distances": 10
    },
    "error_patterns": {
        "FileNotFoundError": 2,
        "ValidationError": 1
    },
    "time_range": {
        "first": "2024-01-01T10:00:00Z",
        "last": "2024-01-01T11:30:00Z"
    }
}
```

**Memory Reduction**: ~94% (50 objects → 1 summary)

### Token-Based Automatic Compaction

#### Session Token Management
```python
def check_and_trigger_compaction(self, config: SummarizationConfig) -> bool:
    """Check if memory exceeds token cap and trigger compaction if needed."""
    current_tokens = self.estimate_session_memory_tokens()
    
    if current_tokens > config.session_token_cap:  # 12K default
        self.logger.warning(
            f"Session memory token cap exceeded: {current_tokens} > {config.session_token_cap}. "
            f"Triggering compaction..."
        )
        self.compact_session_memory()
        return True
    
    return False
```

#### Token Estimation Strategy
```python
def estimate_session_memory_tokens(self) -> int:
    """Estimate total token count for session memory components."""
    total = 0
    
    # Tool history (all executions for active goal)
    for exec in self.state.tool_history:
        total += self.estimate_tokens(str(exec.model_dump()))
    
    # Subtasks (all tasks for active goal)  
    for task in self.state.subtasks:
        total += self.estimate_tokens(str(task.model_dump()))
    
    # Context string
    total += self.estimate_tokens(self.state.context)
    
    # Completed goals summaries
    for goal in self.state.completed_goals:
        total += self.estimate_tokens(str(goal.model_dump()))
    
    return total
```

### Configuration Options

```python
class SessionMemoryConfig:
    # Goal lifecycle
    enable_goal_based_archiving: bool = True
    auto_archive_on_goal_change: bool = True
    
    # Token-based triggers  
    session_token_cap: int = 12000  # Trigger compaction threshold
    session_token_warning_threshold: int = 10000  # Warning level
    completed_goals_token_cap: int = 6000  # Limit for archived goals
    
    # Retention settings
    keep_completed_goals: int = 10  # Recent completed goals to keep
    tool_history_keep_recent: int = 20  # Recent executions for active goal
    subtask_keep_failed: int = 5  # Recent failed tasks to keep
    context_max_length: int = 2000  # Maximum context string length
```

## Layer 3: Execution Memory (Reactive State)

### Purpose
Tracks execution context and discovered insights during task execution. Resets for each new task/goal, providing focused context for reactive reasoning.

### Components

#### Execution Steps
- **Content**: Individual reasoning and execution steps
- **Capacity**: 100 steps (FIFO cleanup when exceeded)
- **Lifecycle**: Reset per task - no summarization needed
- **Purpose**: Immediate execution context for replanning

#### Discovered Context  
```python
class ExecutionStep(BaseModel):
    id: str
    step_type: str  # reasoning, tool_execution, validation, replanning
    task_name: str
    description: str
    success: bool
    discovered_context: Dict[str, Any]  # Key insights found
    execution_time: datetime
    confidence: Optional[float]
```

#### Performance Metrics
- **Success Patterns**: Track successful tool/task combinations
- **Error Patterns**: Common failure modes and solutions  
- **Plan Modifications**: History of replanning events
- **Context Insights**: Strategic discoveries about building/compliance

### Memory Management
```python
class ExecutionMemory:
    def cleanup_if_needed(self) -> None:
        """FIFO cleanup when steps exceed capacity."""
        if len(self.execution_steps) > self.config.max_execution_steps:
            # Remove oldest steps (simple FIFO)
            excess = len(self.execution_steps) - self.config.max_execution_steps
            self.execution_steps = self.execution_steps[excess:]
```

**Note**: No sophisticated compaction needed - execution memory resets per task, so FIFO cleanup is sufficient.

## Summarization Service

### Centralized Summarization Engine

```python
class SummarizationService:
    """Centralized service for all memory summarization needs."""
    
    @traceable(name="summarize_conversation_async")
    async def summarize_conversation_async(self, messages: List[str]) -> str:
        """LLM-based conversation summarization."""
        
    @traceable(name="summarize_tool_history") 
    def summarize_tool_history(self, executions: List[ToolExecution]) -> Dict[str, Any]:
        """Rule-based statistical summarization of tool executions."""
        
    @traceable(name="summarize_subtasks")
    def summarize_subtasks(self, tasks: List[SubTask]) -> Dict[str, Any]:
        """Rule-based thematic summarization of completed subtasks."""
```

### LangSmith Tracing Integration
All summarization operations include @traceable decorators for complete observability:

- **SummarizationService**: 4 traced methods
- **SessionMemory.compact_session_memory**: Full compaction process 
- **SessionMemory._archive_goal_data**: Goal archiving process
- **MemoryManager.get_memory_usage_stats**: Memory analytics

## Memory Analytics

### Real-Time Metrics

```python
def get_memory_usage_stats(self) -> Dict[str, Any]:
    """Get comprehensive memory usage statistics."""
    return {
        # Short-term memory
        "conversation": {
            "buffer_messages": len(self.short_term.conversation_buffer),
            "has_summary": bool(self.short_term.conversation_summary),
            "estimated_tokens": self.short_term._estimate_conversation_tokens()
        },
        
        # Session memory  
        "session": {
            "current_goal": self.session.state.current_goal_id,
            "tool_executions": len(self.session.state.tool_history),
            "subtasks": len(self.session.state.subtasks),
            "completed_goals": len(self.session.state.completed_goals),
            "estimated_tokens": self.session.estimate_session_memory_tokens()
        },
        
        # Execution memory
        "execution": {
            "execution_steps": len(self.execution.execution_steps),
            "discovered_context_entries": len(self.execution.discovered_context),
            "current_task": self.execution.current_task_name
        }
    }
```

### Performance Monitoring
- **Memory Reduction**: Track compression ratios (typically 40-60%)
- **Token Usage**: Monitor against caps to prevent overruns
- **Summarization Performance**: Track LLM call duration and success rates
- **Cache Hit Rates**: Monitor summarization cache effectiveness

## Integration Points

### Memory Manager Coordination

```python
class MemoryManager:
    """Coordinates all memory layers with automatic management."""
    
    def _increment_operation_count(self) -> None:
        """Automatic triggers for memory management."""
        self._operation_count += 1
        
        # Token-based automatic compaction (PRIMARY)
        if self.config.enable_session_compaction:
            self.session.check_and_trigger_compaction(self.config)
        
        # Short-term memory token check
        if self.config.enable_short_term_summarization:
            current_tokens = self.short_term._estimate_conversation_tokens()
            if current_tokens > self.config.short_term_token_cap:
                self.short_term._trigger_async_summarization()
```

### Reasoning Agent Integration

#### Goal Lifecycle Integration
```python
def process_goal(self, goal: str) -> Dict[str, Any]:
    """Process goal with automatic memory lifecycle management."""
    # Set new goal (archives previous goal automatically)
    if self.memory_manager:
        goal_id = self.memory_manager.set_session_goal(goal)
    
    # Execute reasoning...
    
    # Mark goal as complete when achieved
    if goal_achieved:
        self.memory_manager.complete_current_goal(success=True)
```

## Configuration System

### Comprehensive Memory Configuration

```python
@dataclass
class MemoryConfig:
    # Short-term memory
    short_term_buffer_size: int = 10
    short_term_token_cap: int = 4000
    enable_short_term_summarization: bool = True
    summarization_strategy: str = "async"  # sync, async, background
    
    # Session memory  
    session_token_cap: int = 12000
    session_token_warning_threshold: int = 10000
    enable_goal_based_archiving: bool = True
    tool_history_keep_recent: int = 20
    completed_goals_token_cap: int = 6000
    
    # Execution memory
    max_execution_steps: int = 100
    enable_context_deduplication: bool = True
    
    # Summarization service
    summarization_model: str = "gpt-4o-mini"
    summarization_temperature: float = 0.1
    enable_summarization_cache: bool = True
    cache_ttl_hours: int = 24
```

### Environment Variable Support
```bash
# Memory configuration via environment variables
export AEC_MEMORY_SHORT_TERM_TOKEN_CAP="5000"
export AEC_MEMORY_SESSION_TOKEN_CAP="15000" 
export AEC_MEMORY_ENABLE_GOAL_ARCHIVING="true"
export AEC_MEMORY_SUMMARIZATION_MODEL="gpt-4o-mini"
```

## Testing Strategy

### Comprehensive Test Coverage

#### Unit Tests
- **SummarizationService**: LLM and rule-based summarization
- **ShortTermMemory**: Conversation management with token triggers
- **SessionMemory**: Goal lifecycle and archiving processes
- **ExecutionMemory**: FIFO cleanup and context tracking

#### Integration Tests  
- **Memory Lifecycle**: End-to-end goal archiving workflow
- **Token Management**: Automatic compaction trigger verification
- **Cross-Layer Coordination**: MemoryManager integration testing

#### Performance Tests
- **Memory Reduction**: Verify 40-60% compression targets
- **Summarization Speed**: Non-blocking async operation
- **Token Estimation**: Accuracy of token counting heuristics

### Test Scenarios

#### Long-Running Session Test
```python
def test_long_running_session():
    """Test memory management across multiple goals."""
    # Goal 1: Door analysis (50 operations)
    # Goal 2: Fire compliance (75 operations)  
    # Goal 3: Window analysis (30 operations)
    # Verify: Goal 1&2 archived, Goal 3 active, memory under caps
```

#### Token Cap Management Test
```python
def test_token_cap_management():
    """Test automatic compaction at token caps."""
    # Add operations until token cap exceeded
    # Verify: Automatic compaction triggered
    # Verify: Memory reduced below cap
    # Verify: Essential information preserved
```

## Performance Characteristics

### Memory Efficiency
- **Goal-Based Archiving**: 50-70% memory reduction per completed goal
- **Token-Based Triggers**: Prevents unbounded growth with hard caps
- **Statistical Summarization**: Preserves patterns without LLM overhead
- **FIFO Cleanup**: O(1) execution memory management

### Summarization Performance
- **Rule-Based**: <10ms for statistical summaries (tool history, subtasks)
- **LLM-Based**: 1-3 seconds for conversation summarization (async)
- **Cache Benefits**: 80%+ cache hit rate for repeated content
- **Background Processing**: Non-blocking summarization queue

### Token Management
- **Estimation Accuracy**: ±20% using character-based heuristics
- **Conservative Caps**: Leave room for prompts and responses
- **Early Warning**: Alerts at 75% of cap (configurable)
- **Automatic Triggers**: No manual intervention required

## Best Practices

### Memory Configuration
1. **Start Conservative**: Use default token caps initially
2. **Monitor Usage**: Track memory analytics in production
3. **Adjust Gradually**: Increase caps based on real usage patterns
4. **Enable Tracing**: Use LangSmith for memory operation visibility

### Goal Management
1. **Clear Goal Boundaries**: Ensure each goal has distinct completion criteria
2. **Complete Goals Explicitly**: Call `complete_current_goal()` when done
3. **Monitor Completed Goals**: Periodically review archived goal summaries
4. **Archive Old Goals**: Clean up very old completed goals if needed

### Performance Optimization
1. **Enable Caching**: Use summarization cache for repeated content
2. **Async Summarization**: Use async strategy for non-blocking operation
3. **Batch Operations**: Group memory operations when possible
4. **Monitor Metrics**: Track compression ratios and performance

## Future Enhancements

### Planned Features
1. **Hierarchical Summarization**: Summary of summaries for very old data
2. **Importance-Based Retention**: Keep high-importance items longer
3. **Cross-Session Learning**: Preserve insights across session boundaries
4. **Advanced Analytics**: Memory usage predictions and recommendations

### Research Directions  
1. **Semantic Summarization**: Content-aware compression beyond statistics
2. **Adaptive Token Caps**: Dynamic caps based on usage patterns
3. **Memory Sharing**: Share insights between related analysis sessions
4. **Explainable Memory**: Understand what information is preserved/lost

## Summary

The three-layer memory management system delivers:

✅ **Intelligent Growth Prevention**: Token-based caps prevent unbounded memory usage  
✅ **Goal-Based Organization**: Clean separation and archiving of analysis sessions  
✅ **Preservation of Insights**: Statistical summaries maintain essential patterns  
✅ **Performance Efficiency**: 40-60% memory reduction with minimal overhead  
✅ **Full Observability**: Complete LangSmith tracing of all memory operations  
✅ **Zero Manual Management**: Automatic lifecycle and compaction triggers  

This system enables long-running building analysis sessions while maintaining optimal performance and preserving critical insights for decision-making.