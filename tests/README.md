# Tests

This directory contains the test suite for the `image-builder-mcp` project.

## Test Files

- `test_auth.py`: Tests for authentication functionality
- `test_get_blueprints.py`: Tests for the blueprint retrieval functionality
- `test_llm_integration.py`: Integration tests with OpenAI-compatible LLM endpoints

## Running Tests

### Standard Tests

Run all tests:
```bash
make test
```

for debug (logging) output of the tests, run:

```bash
make test-verbose
```

### LLM Integration Tests

The LLM integration tests require an OpenAI-compatible endpoint and will be skipped unless the required environment variables are set.

#### Required Environment Variables

- `MODEL_API`: The base URL of the OpenAI-compatible API (e.g., `https://api.openai.com/v1`)
- `MODEL_ID`: The model identifier (e.g., `gpt-4`, `llama-3.1-70b-versatile`)
- `USER_KEY`: The API key for authentication

#### What the LLM Integration Tests Do

These tests:
1. Extract tool definitions from the MCP server
2. Send test questions to the LLM with the tool definitions
3. Verify that the LLM selects appropriate tools for the given questions
4. Check that the responses contain expected tool calls

Test questions include:
- "can you create a RHEL 9 image for me?"
- "what is the status of my latest image build"

The tests verify that the LLM correctly identifies relevant tools like `create_blueprint`, `get_openapi`, `get_composes`, and `get_compose_details` based on the user's intent.

## Test Coverage

Run tests with coverage:
```bash
make test-coverage
```

This will generate an HTML coverage report in the `htmlcov/` directory.
