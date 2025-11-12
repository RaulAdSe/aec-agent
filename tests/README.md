# Test Structure

## Current Organization

### tests/memory_system/
Tests for agent memory and persistence.

#### tests/memory_system/unit/
Memory component testing:
- `test_memory_system.py` - Memory managers and conversation history

### tests/fixtures/
Shared test data and utilities:
- `sample_compliance_docs.py` - Sample building data and compliance documents

### tests/utils/
Testing utilities and helpers:
- `test_helpers.py` - Common test functions and utilities

### tests/legacy/
Previously used test files that are no longer active:

#### tests/legacy/unit/
Old unit tests:
- `test_basic.py`, `test_config.py`, `test_document_retrieval.py`

#### tests/legacy/integration/  
Old integration tests:
- `test_agent_workflow.py`, `test_knowledge_base_workflow.py`

#### tests/legacy/agentic/
Old agentic system tests:
- `performance/test_performance.py` - Performance testing
- `workflows/test_compliance_workflow.py` - Compliance workflows

## Running Tests

```bash
# Run all active tests
pytest tests/memory_system/ tests/utils/ tests/fixtures/

# Run memory system tests
pytest tests/memory_system/

# Run with coverage (excluding legacy)
pytest --cov=aec_agent tests/memory_system/ tests/utils/

# Run legacy tests if needed
pytest tests/legacy/
```

## Adding New Tests

When implementing new reasoning agent features, create test directories as needed:
- `tests/reasoning_agent/` for autonomous reasoning tests
- `tests/tools/` for agent tool tests  
- `tests/core/` for core functionality tests

## Test Guidelines

1. **Keep clean structure**: Only create test directories when you have actual tests
2. **Use fixtures**: Leverage shared test data from `tests/fixtures/`
3. **Clean isolation**: Each test should be independent and clean up after itself
4. **Legacy separation**: Keep old unused tests in `tests/legacy/`