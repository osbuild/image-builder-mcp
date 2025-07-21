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

### Setup

1. **Copy the example configuration:**
   ```bash
   cp test_config.json.example test_config.json
   ```

2. **Configure your models** by editing `test_config.json` with your API credentials:
   ```json
   {
     "llm_configurations": [
       {
         "name": "Primary Model",
         "MODEL_ID": "granite-3.1",
         "MODEL_API": "https://your-vLLM-server",
         "USER_KEY": "your-api-key"
       }
     ],
     "guardian_llm": {
       "name": "Optional model for Test evaluation",
       "MODEL_ID": "granite-3.2",
       "MODEL_API": "https://your-vLLM-server2",
       "USER_KEY": "your-api-key"
     }
   }
   ```

### Running Tests

```bash
# Run all LLM integration tests across all configured models
pytest tests/test_llm_integration.py -v

# Run specific test across all models
pytest tests/test_llm_integration.py::TestLLMIntegration::test_rhel_image_creation_behavioral_rules -v

# Run tests for a specific model configuration
pytest tests/test_llm_integration.py -k "Primary Model" -v
```

### Test Output

Each test run indicates which model is being tested:
```
🧪 Testing model: Primary Model (gpt-4-turbo)
✓ Behavioral rules working for Primary Model - tools intended: ['get_openapi']
```

### Fallback

If `test_config.json` is missing, tests fall back to environment variables: `MODEL_API`, `MODEL_ID`, `USER_KEY`.

## Requirements

- Python packages: `pytest`, `deepeval`, `requests`
- Valid API credentials for LLM services
- Image Builder API access (optional, can use mocks)
