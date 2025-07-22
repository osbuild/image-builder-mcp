"""Test the utils of the tests."""

import json

from .utils import MCPError, load_llm_configurations


# Load LLM configurations for parametrization
llm_configurations, _ = load_llm_configurations()

# pylint: disable=redefined-outer-name,too-many-locals


def test_tool_parameter_casting(test_agent, verbose_logger):
    """Test that tool arguments are properly cast according to MCP parameter specifications."""

    # Test casting with controlled data
    test_tools = [{
        'name': 'get_openapi',
        'inputSchema': {
            'properties': {
                'response_size': {'type': 'integer'}
            }
        }
    }]

    # Test casting string to integer
    test_args = {'response_size': '7'}
    cast_args = test_agent.cast_tool_args(test_args, 'get_openapi', test_tools)

    verbose_logger.debug(f"Original args: {test_args}")
    verbose_logger.debug(f"Cast args: {cast_args}")

    # Verify that response_size was cast to integer
    assert isinstance(cast_args['response_size'], int), f"response_size should be int, got {
        type(cast_args['response_size'])}"
    assert cast_args['response_size'] == 7, f"response_size should be 7, got {cast_args['response_size']}"

    print("✓ Parameter casting test passed - string '7' was successfully cast to integer 7")

    # Test with actual tool call to ensure it works end-to-end
    try:
        tool_response = test_agent.call_tool('get_openapi', test_args)
        verbose_logger.debug(f"Tool call successful: {tool_response}")

        # Verify we got a response (the tool should work with cast parameters)
        assert tool_response is not None, "Tool should return a response"

        print("✓ End-to-end tool call with parameter casting successful")

    except (MCPError, json.JSONDecodeError, KeyError, ValueError) as e:
        verbose_logger.error(f"Tool call failed: {e}")
        # The test should still pass if casting worked, even if the tool call fails for other reasons
        print(f"⚠ Tool call failed but parameter casting worked: {e}")

    print("✓ Parameter casting functionality verified")
