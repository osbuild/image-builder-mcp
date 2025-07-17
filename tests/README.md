# Testing Documentation

This directory contains test suites for the Image Builder MCP server.

## Test Structure

- `test_auth.py` - Authentication and OAuth tests
- `test_get_blueprints.py` - Blueprint retrieval tests
- `test_llm_integration.py` - LLM integration tests using deepeval
- `test_utils.py` - Shared testing utilities and helper functions
- `conftest.py` - Pytest fixtures and configuration

## LLM Integration Testing

The LLM integration tests support matrix testing across multiple LLM configurations using deepeval framework.

### Configuration

Create a `test_config.json` file in the project root with the following structure:

```json
{
  "llm_configurations": [
    {
      "name": "Primary Model",
      "MODEL_API": "${MODEL_API}",
      "MODEL_ID": "${MODEL_ID}",
      "USER_KEY": "${USER_KEY}"
    },
    {
      "name": "Alternative Model 1",
      "MODEL_API": "${MODEL_API_ALT1}",
      "MODEL_ID": "${MODEL_ID_ALT1}",
      "USER_KEY": "${USER_KEY_ALT1}"
    },
    {
      "name": "Alternative Model 2",
      "MODEL_API": "${MODEL_API_ALT2}",
      "MODEL_ID": "${MODEL_ID_ALT2}",
      "USER_KEY": "${USER_KEY_ALT2}"
    }
  ]
}
```

The `${VARIABLE}` syntax allows environment variable substitution. Set the corresponding environment variables for each model configuration you want to test.

### Environment Variables

For each LLM configuration, you need to set:
- `MODEL_API` - API endpoint URL (e.g., `https://api.example.com/v1`)
- `MODEL_ID` - Model identifier (e.g., `gpt-4`, `claude-3`)
- `USER_KEY` - API authentication key

For multiple configurations, use suffixed variables:
- Primary: `MODEL_API`, `MODEL_ID`, `USER_KEY`
- Alternative 1: `MODEL_API_ALT1`, `MODEL_ID_ALT1`, `USER_KEY_ALT1`
- Alternative 2: `MODEL_API_ALT2`, `MODEL_ID_ALT2`, `USER_KEY_ALT2`

### Running Matrix Tests

The LLM tests will automatically run against all valid configurations:

```bash
# Run all LLM integration tests across all configured models
pytest tests/test_llm_integration.py -v

# Run specific test across all models
pytest tests/test_llm_integration.py::TestLLMIntegration::test_rhel_image_creation_behavioral_rules -v

# Run tests for a specific model configuration
pytest tests/test_llm_integration.py -k "Primary Model" -v
```

### Test Output

Each test run clearly indicates which model is being tested:

```
🧪 Testing model: Primary Model (gpt-4-turbo)
✓ Behavioral rules working for Primary Model - tools intended: ['get_openapi']
```

### Fallback Behavior

If `test_config.json` is not found or contains no valid configurations, the tests will fall back to using environment variables directly (`MODEL_API`, `MODEL_ID`, `USER_KEY`) for backward compatibility.

## Test Requirements

- Python packages: `pytest`, `deepeval`, `requests`
- For LLM tests: Valid API credentials for at least one LLM service
- For server tests: Image Builder API access (optional, can use mocks)
