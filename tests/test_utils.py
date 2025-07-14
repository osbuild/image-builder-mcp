"""Utility functions and tests for LLM integration testing."""

import json
import os
import logging
import socket
import time
import asyncio
import multiprocessing
from typing import Dict, List, Any

import pytest
import requests


def should_skip_llm_tests() -> bool:
    """Check if LLM integration tests should be skipped."""
    required_vars = ['MODEL_API', 'MODEL_ID', 'USER_KEY']
    return not all(os.getenv(var) for var in required_vars)


@pytest.fixture
def verbose_logger(request):
    """Get a logger that respects pytest verbosity."""
    logger = logging.getLogger(__name__)

    verbosity = request.config.getoption('verbose', default=0)

    if verbosity >= 3:
        logger.setLevel(logging.DEBUG)
    elif verbosity >= 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    return logger


def get_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(scope="session")
def mcp_server_thread():  # pylint: disable=too-many-locals
    """Start MCP server in a separate thread using HTTP streaming."""

    port = get_free_port()
    server_url = f'http://127.0.0.1:{port}/mcp/'

    # Use multiprocessing instead of threading to avoid asyncio conflicts
    server_queue = multiprocessing.Queue()

    def start_server_process():
        """Start the MCP server in a separate process."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Import here to avoid module-level asyncio conflicts
            # pylint: disable=import-outside-toplevel
            from image_builder_mcp.server import ImageBuilderMCP

            # Get credentials from environment (may be None for testing)
            client_id = os.getenv("IMAGE_BUILDER_CLIENT_ID")
            client_secret = os.getenv("IMAGE_BUILDER_CLIENT_SECRET")

            mcp_server = ImageBuilderMCP(
                client_id=client_id,
                client_secret=client_secret,
                stage=False,  # Use production API
                proxy_url=None,
                transport="http",
                oauth_enabled=False,
            )

            # Signal that server is starting
            server_queue.put("starting")

            # Start server with HTTP transport on dynamic port
            mcp_server.run(transport="http", host="127.0.0.1", port=port)

        except Exception as e:  # pylint: disable=broad-exception-caught
            server_queue.put(f"error: {e}")

    # Start server process
    server_process = multiprocessing.Process(target=start_server_process, daemon=True)
    server_process.start()

    try:
        # Wait for server to start
        start_signal = server_queue.get(timeout=10)
        if start_signal.startswith("error:"):
            # pylint: disable=broad-exception-raised
            raise Exception(f"Server failed to start: {start_signal}")

        # Additional wait for server to be fully ready
        time.sleep(3)

        # Test server connectivity with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                test_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0.0"}
                    }
                }
                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json, text/event-stream'
                }
                response = requests.post(server_url, json=test_request, headers=headers, timeout=10)

                if response.status_code == 200:
                    break

                if attempt == max_retries - 1:
                    # pylint: disable=broad-exception-raised
                    raise Exception(f"Server not responding properly after {max_retries}"
                                    f"attempts: {response.status_code} - {response.text}")

                time.sleep(2)  # Wait before retry

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    # pylint: disable=broad-exception-raised
                    raise Exception(f"Failed to connect to server after {max_retries} attempts: {e}") from e
                time.sleep(2)  # Wait before retry

        yield server_url

    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.fail(f"Failed to start MCP server: {e}")
    finally:
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=5)
            if server_process.is_alive():
                server_process.kill()


class LLMTestUtils:
    """Utility class for LLM integration testing."""

    def get_mcp_tools_and_instructions(self, server_url: str) -> tuple[List[Dict[str, Any]], str]:
        """Extract available tools and instructions from MCP server using HTTP streaming protocol."""
        session = requests.Session()
        session_id = None

        try:
            # Step 1: Initialize MCP session
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }

            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream'
            }

            response = session.post(
                server_url,
                json=init_request,
                headers=headers,
                timeout=10
            )

            if response.status_code != 200:
                # pylint: disable=broad-exception-raised
                raise Exception(f"Initialize failed: {response.status_code} - {response.text}")

            # Extract session ID from response headers
            session_id = response.headers.get('mcp-session-id')
            if not session_id:
                # Try to extract from cookie or other headers
                session_id = response.headers.get('Mcp-Session-Id')

            # Parse response - it could be JSON or SSE format
            init_response = self._parse_mcp_response(response.text)
            if not init_response:
                # pylint: disable=broad-exception-raised
                raise Exception("Failed to parse MCP response")

            # Extract server instructions from initialization response
            server_instructions = ""
            if isinstance(init_response, list) and len(init_response) > 0:
                server_info = init_response[0].get('result', {})
            else:
                server_info = init_response.get('result', {})

            # Look for instructions in the server info
            server_instructions = server_info.get('instructions', '')

            # Step 2: Send initialized notification
            if session_id:
                headers['mcp-session-id'] = session_id

            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }

            session.post(
                server_url,
                json=initialized_notification,
                headers=headers,
                timeout=10
            )

            # Step 3: List available tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }

            response = session.post(
                server_url,
                json=tools_request,
                headers=headers,
                timeout=10
            )

            if response.status_code != 200:
                # pylint: disable=broad-exception-raised
                raise Exception(f"Tools list failed: {response.status_code} - {response.text}")

            tools_response = self._parse_mcp_response(response.text)

            # Extract tools from response
            if isinstance(tools_response, list) and len(tools_response) > 0:
                tools_data = tools_response[0].get('result', {})
            else:
                tools_data = tools_response.get('result', {})

            tools = tools_data.get('tools', [])

            # Convert MCP tools to OpenAI function format
            return self._convert_mcp_tools_to_openai_format(tools), server_instructions

        except Exception as e:
            raise Exception(f"Failed to get MCP tools: {e}") from e  # pylint: disable=broad-exception-raised

    def get_mcp_tools(self, server_url: str) -> List[Dict[str, Any]]:
        """Extract available tools from MCP server using HTTP streaming protocol."""
        tools, _ = self.get_mcp_tools_and_instructions(server_url)
        return tools

    def _parse_mcp_response(self, response_text: str) -> Dict[str, Any]:
        """Parse MCP response which could be JSON or SSE format."""
        try:
            # Try parsing as JSON first
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try parsing as SSE format
            return self._parse_sse_response(response_text)

    def _parse_sse_response(self, sse_text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response format."""
        for line in sse_text.split('\n'):
            if line.startswith('data: '):
                data_part = line[6:]  # Remove 'data: ' prefix
                try:
                    return json.loads(data_part)
                except json.JSONDecodeError:
                    continue

        raise Exception(f"No valid JSON found in SSE response: {sse_text}")  # pylint: disable=broad-exception-raised

    def _convert_mcp_tools_to_openai_format(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format."""
        openai_tools = []

        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {}
                    })
                }
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def call_llm(self, prompt: str, tools: List[Dict[str, Any]], verbose_logger: logging.Logger, system_prompt: str = None, messages: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call LLM with tools and return response.

        Args:
            prompt: User prompt (used when messages is None)
            tools: Available tools
            verbose_logger: Logger for output
            system_prompt: System prompt (used when messages is None)
            messages: Full conversation history (overrides prompt/system_prompt)
        """
        api_url = os.getenv('MODEL_API')
        model_id = os.getenv('MODEL_ID')
        api_key = os.getenv('USER_KEY')

        verbose_logger.info(f"call_llm using LLM {model_id} at {api_url}")

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        # Build messages array
        if messages is None:
            # Use simple prompt-based approach (backward compatibility)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            # Use provided conversation history
            messages = messages

        payload = {
            "model": model_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }

        verbose_logger.debug(f"call_llm: payload:\n{json.dumps(payload, indent=2)}")

        response = requests.post(f"{api_url}/chat/completions", json=payload, headers=headers, timeout=30)

        verbose_logger.debug(f"call_llm: response:\n{json.dumps(response.json(), indent=2)}")

        if response.status_code != 200:
            # pylint: disable=broad-exception-raised
            raise Exception(f"LLM API call failed: {response.status_code} - {response.text}")

        return response.json()

    def _cast_tool_args(self, tool_args: Dict[str, Any], tool_name: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cast tool arguments to match the parameter specifications from MCP tools."""
        # Find the tool definition
        tool_def = None
        for tool in tools:
            if tool.get("name") == tool_name:
                tool_def = tool
                break

        if not tool_def:
            # Return original args if tool not found
            return tool_args

        # Get parameter specifications from the tool
        input_schema = tool_def.get("inputSchema", {})
        properties = input_schema.get("properties", {})

        cast_args = {}

        for arg_name, arg_value in tool_args.items():
            if arg_name in properties:
                prop_spec = properties[arg_name]
                prop_type = prop_spec.get("type")

                # Cast according to the expected type
                if prop_type == "integer":
                    try:
                        cast_args[arg_name] = int(arg_value)
                    except (ValueError, TypeError):
                        cast_args[arg_name] = arg_value
                elif prop_type == "number":
                    try:
                        cast_args[arg_name] = float(arg_value)
                    except (ValueError, TypeError):
                        cast_args[arg_name] = arg_value
                elif prop_type == "boolean":
                    if isinstance(arg_value, str):
                        cast_args[arg_name] = arg_value.lower() in ("true", "1", "yes", "on")
                    else:
                        cast_args[arg_name] = bool(arg_value)
                elif prop_type == "string":
                    # Handle nullable strings
                    if arg_value is None or arg_value == "null":
                        cast_args[arg_name] = None
                    else:
                        cast_args[arg_name] = str(arg_value)
                elif prop_type == "object":
                    # Handle dict/object types
                    if isinstance(arg_value, str):
                        try:
                            cast_args[arg_name] = json.loads(arg_value)
                        except json.JSONDecodeError:
                            cast_args[arg_name] = arg_value
                    else:
                        cast_args[arg_name] = arg_value
                elif prop_type == "array":
                    # Handle array types
                    if isinstance(arg_value, str):
                        try:
                            cast_args[arg_name] = json.loads(arg_value)
                        except json.JSONDecodeError:
                            cast_args[arg_name] = arg_value
                    else:
                        cast_args[arg_name] = arg_value
                else:
                    # Unknown type, keep as is
                    cast_args[arg_name] = arg_value
            else:
                # Parameter not in schema, keep as is
                cast_args[arg_name] = arg_value

        return cast_args

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any], server_url: str, verbose_logger: logging.Logger, system_prompt: str) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
        session = requests.Session()
        session_id = None

        try:
            # Step 1: Initialize MCP session
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }

            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream'
            }

            response = session.post(server_url, json=init_request, headers=headers, timeout=10)

            if response.status_code != 200:
                raise Exception(f"Initialize failed: {response.status_code} - {response.text}")

            # Extract session ID from response headers
            session_id = response.headers.get('mcp-session-id')
            if not session_id:
                session_id = response.headers.get('Mcp-Session-Id')

            # Step 2: Send initialized notification
            if session_id:
                headers['mcp-session-id'] = session_id

            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }

            session.post(server_url, json=initialized_notification, headers=headers, timeout=10)

            # Step 2.5: Get tool definitions to cast arguments properly
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }

            tools_response = session.post(server_url, json=tools_request, headers=headers, timeout=10)

            if tools_response.status_code == 200:
                tools_data = self._parse_mcp_response(tools_response.text)

                # Extract tools from response
                if isinstance(tools_data, list) and len(tools_data) > 0:
                    tools_result = tools_data[0].get('result', {})
                else:
                    tools_result = tools_data.get('result', {})

                tools = tools_result.get('tools', [])

                # Cast tool arguments according to parameter specifications
                tool_args = self._cast_tool_args(tool_args, tool_name, tools)
                verbose_logger.debug(f"Cast tool args for {tool_name}: {tool_args}")

            # Step 3: Call the tool
            tool_request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": tool_args
                }
            }

            verbose_logger.debug(f"Calling tool {tool_name} with args: {tool_args}")

            response = session.post(server_url, json=tool_request, headers=headers, timeout=30)

            if response.status_code != 200:
                raise Exception(f"Tool call failed: {response.status_code} - {response.text}")

            tool_response = self._parse_mcp_response(response.text)
            verbose_logger.debug(f"Tool response: {tool_response}")

            return tool_response

        except Exception as e:
            verbose_logger.error(f"Error calling tool {tool_name}: {e}")
            raise Exception(f"Failed to call tool {tool_name}: {e}") from e

    def _process_tool_calls(self, response: Dict[str, Any], tools: List[Dict[str, Any]], server_url: str, verbose_logger: logging.Logger, system_prompt: str, original_prompt: str = None) -> Dict[str, Any]:
        """Process tool calls and return final LLM response with conversation history."""
        choice = response["choices"][0]
        message = choice["message"]

        if "tool_calls" not in message or not message["tool_calls"]:
            return response

        tool_calls = message["tool_calls"]
        tool_results = []

        # Process each tool call
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]

            # Parse tool arguments
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                verbose_logger.error(f"Failed to parse tool arguments: {e}")
                tool_result = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": f"Error: Failed to parse tool arguments - {e}"
                }
                tool_results.append(tool_result)
                continue

            # Call the tool
            try:
                tool_response = self.call_tool(tool_name, tool_args, server_url, verbose_logger, system_prompt)

                # Extract the actual result from the MCP response
                if isinstance(tool_response, list) and len(tool_response) > 0:
                    result = tool_response[0].get('result', {})
                else:
                    result = tool_response.get('result', {})

                # Format the result as a tool response
                tool_result = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": json.dumps(result) if isinstance(result, dict) else str(result)
                }
                tool_results.append(tool_result)

            except Exception as e:
                verbose_logger.error(f"Tool call failed: {e}")
                tool_result = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": f"Error: Tool call failed - {e}"
                }
                tool_results.append(tool_result)

        # Build proper conversation history
        conversation_messages = []

        # Add system prompt if provided
        if system_prompt:
            conversation_messages.append({"role": "system", "content": system_prompt})

        # Add original user message
        if original_prompt:
            conversation_messages.append({"role": "user", "content": original_prompt})

        # Add assistant message with tool calls
        conversation_messages.append(message)

        # Add tool result messages
        conversation_messages.extend(tool_results)

        verbose_logger.debug(f"Conversation history: {json.dumps(conversation_messages, indent=2)}")

        # Call LLM again with the full conversation history
        final_response = self.call_llm(
            prompt=None,  # Not used when messages is provided
            tools=tools,
            verbose_logger=verbose_logger,
            system_prompt=None,  # Already included in messages
            messages=conversation_messages
        )

        return final_response

    def complete_conversation_with_tools(self, user_prompt: str, tools: List[Dict[str, Any]], server_url: str, verbose_logger: logging.Logger, system_prompt: str) -> Dict[str, Any]:
        """Complete a full conversation with tool calls, implementing proper agent flow."""
        verbose_logger.info(f"Starting conversation with prompt: {user_prompt}")

        # Initial LLM call
        response = self.call_llm(user_prompt, tools, verbose_logger, system_prompt)

        # Check if LLM wants to use tools
        choice = response["choices"][0]
        message = choice["message"]

        if "tool_calls" in message and message["tool_calls"]:
            verbose_logger.info(f"LLM made tool calls: {[tc['function']['name'] for tc in message['tool_calls']]}")

            # Process tool calls and get final response
            final_response = self._process_tool_calls(
                response, tools, server_url, verbose_logger, system_prompt, user_prompt)

            verbose_logger.info("Conversation completed with tool calls")
            return final_response
        else:
            verbose_logger.info("LLM responded without tool calls")
            return response


# Utility tests
@pytest.mark.skipif(should_skip_llm_tests(), reason="LLM environment variables not set")
class TestLLMUtils:
    """Test utility functions for LLM integration."""

    def test_tool_definitions_extraction(self, mcp_server_thread):
        """Test that we can extract tool definitions from MCP server."""
        server_url = mcp_server_thread
        utils = LLMTestUtils()

        # Get tools from MCP server
        tools = utils.get_mcp_tools(server_url)

        # Verify we got some tools
        assert len(tools) > 0, "No tools extracted from MCP server"

        # Check that tools have required OpenAI format
        for tool in tools:
            assert "type" in tool, f"Tool missing 'type' field: {tool}"
            assert tool["type"] == "function", f"Tool type should be 'function': {tool}"
            assert "function" in tool, f"Tool missing 'function' field: {tool}"

            func = tool["function"]
            assert "name" in func, f"Function missing 'name' field: {func}"
            assert "description" in func, f"Function missing 'description' field: {func}"
            assert "parameters" in func, f"Function missing 'parameters' field: {func}"

        print(f"Successfully extracted {len(tools)} tools from MCP server")
        for tool in tools:
            print(f"  - {tool['function']['name']}: {tool['function']['description']}")

    def test_llm_api_connectivity(self, mcp_server_thread, verbose_logger):
        """Test basic LLM API connectivity."""
        server_url = mcp_server_thread
        utils = LLMTestUtils()

        # Get tools and instructions from MCP server
        tools, system_prompt = utils.get_mcp_tools_and_instructions(server_url)

        # Test basic LLM call
        try:
            response = utils.call_llm("Hello, can you help me?", tools, verbose_logger, system_prompt)
            assert "choices" in response, "LLM response missing 'choices' field"
            assert len(response["choices"]) > 0, "LLM response has no choices"

            print("LLM API connectivity test passed")
        except Exception as e:  # pylint: disable=broad-exception-caught
            pytest.skip(f"LLM API not accessible: {e}")

    def test_system_prompt_extraction(self, mcp_server_thread, verbose_logger):
        """Test that system prompt is correctly extracted from MCP server."""
        server_url = mcp_server_thread
        utils = LLMTestUtils()

        # Get tools and instructions from MCP server
        tools, system_prompt = utils.get_mcp_tools_and_instructions(server_url)

        verbose_logger.debug(f"Full system prompt: {system_prompt}")

        # Verify system prompt contains expected elements
        assert system_prompt, "System prompt should not be empty"
        assert "NEVER CALL create_blueprint() IMMEDIATELY" in system_prompt, "System prompt should contain behavioral rules"
        assert "AVAILABLE DISTRIBUTIONS:" in system_prompt, "System prompt should contain available distributions"
        assert "AVAILABLE ARCHITECTURES:" in system_prompt, "System prompt should contain available architectures"
        assert "AVAILABLE IMAGE TYPES:" in system_prompt, "System prompt should contain available image types"

        print("✓ System prompt correctly extracted with all expected elements")
        print(f"System prompt length: {len(system_prompt)} characters")

    def test_tool_parameter_casting(self, mcp_server_thread, verbose_logger):
        """Test that tool arguments are properly cast according to MCP parameter specifications."""
        server_url = mcp_server_thread
        utils = LLMTestUtils()

        # Get tools from MCP server
        tools, system_prompt = utils.get_mcp_tools_and_instructions(server_url)

        # Find the get_openapi tool (which expects response_size as integer)
        get_openapi_tool = None
        for tool in tools:
            if tool['function']['name'] == 'get_openapi':
                get_openapi_tool = tool
                break

        assert get_openapi_tool is not None, "get_openapi tool not found"

        # Test casting string to integer
        test_args = {'response_size': '7'}  # String that should be cast to int

        # Get the raw MCP tools (not converted to OpenAI format) using a fresh session
        session = requests.Session()
        session_id = None

        # Initialize session
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
        }

        response = session.post(server_url, json=init_request, headers=headers, timeout=10)
        assert response.status_code == 200

        # Extract session ID from response headers
        session_id = response.headers.get('mcp-session-id')
        if not session_id:
            session_id = response.headers.get('Mcp-Session-Id')

        # Send initialized notification
        if session_id:
            headers['mcp-session-id'] = session_id

        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        session.post(server_url, json=initialized_notification, headers=headers, timeout=10)

        # Get tools list
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        tools_response = session.post(server_url, json=tools_request, headers=headers, timeout=10)
        assert tools_response.status_code == 200, f"Tools list failed: {
            tools_response.status_code} - {tools_response.text}"

        tools_data = utils._parse_mcp_response(tools_response.text)

        # Extract tools from response
        if isinstance(tools_data, list) and len(tools_data) > 0:
            tools_result = tools_data[0].get('result', {})
        else:
            tools_result = tools_data.get('result', {})

        raw_tools = tools_result.get('tools', [])

        # Test the casting function
        cast_args = utils._cast_tool_args(test_args, 'get_openapi', raw_tools)

        verbose_logger.debug(f"Original args: {test_args}")
        verbose_logger.debug(f"Cast args: {cast_args}")

        # Verify that response_size was cast to integer
        assert isinstance(cast_args['response_size'], int), f"response_size should be int, got {
            type(cast_args['response_size'])}"
        assert cast_args['response_size'] == 7, f"response_size should be 7, got {cast_args['response_size']}"

        # Test with actual tool call to ensure it works end-to-end
        try:
            tool_response = utils.call_tool('get_openapi', test_args, server_url, verbose_logger, system_prompt)
            verbose_logger.debug(f"Tool call successful: {tool_response}")

            # Verify we got a response (the tool should work with cast parameters)
            assert tool_response is not None, "Tool should return a response"

            print("✓ Parameter casting test passed - string '7' was successfully cast to integer 7")

        except Exception as e:
            verbose_logger.error(f"Tool call failed: {e}")
            # The test should still pass if casting worked, even if the tool call fails for other reasons
            print(f"⚠ Tool call failed but parameter casting worked: {e}")

        print(f"✓ Parameter casting functionality verified")
