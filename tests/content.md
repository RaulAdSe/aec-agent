# Test Structure Documentation

## tests/unit/
Test individual components in isolation
- Mock external dependencies
- Fast execution
- One component per test file

## tests/integration/
Test component interactions
- Real dependencies where possible
- End-to-end workflows
- Database/API integrations

## tests/agentic/workflows/
Test agent decision-making processes
- Multi-step agent workflows
- Tool selection and usage
- Memory and state management

## tests/agentic/mocks/
Mock AI services for testing
- Fake LLM responses
- Deterministic agent behavior
- No external API calls

## tests/agentic/performance/
Performance and load testing
- Response time measurements
- Token usage tracking
- Memory usage monitoring

## tests/fixtures/
Test data and sample inputs
- Sample building data
- Expected outputs
- Configuration files

## tests/utils/
Testing helper functions
- Common test utilities
- Setup/teardown helpers
- Test data generators