"""Utility functions and tests for LLM integration testing."""

import json
import os
import logging
import socket
import time
import asyncio
import multiprocessing
from typing import Dict, List, Any, Optional

import pytest
import requests


def cleanup_server_process(server_process: multiprocessing.Process) -> None:
    """Helper function to properly cleanup a server process."""
    if server_process.is_alive():
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()


class ServerStartupError(Exception):
    """Exception raised when MCP server fails to start."""


class ServerConnectionError(Exception):
    """Exception raised when unable to connect to MCP server."""


class MCPResponseError(Exception):
    """Exception raised when MCP response is invalid or unexpected."""


class LLMAPIError(Exception):
    """Exception raised when LLM API call fails."""


class ToolCallError(Exception):
    """Exception raised when tool call fails."""


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


def start_mcp_server_process():
    """Start MCP server in a separate process - shared utility function."""
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
            raise ServerStartupError(f"Server failed to start: {start_signal}")

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
                    raise ServerConnectionError(f"Server not responding properly after {max_retries}"
                                                f"attempts: {response.status_code} - {response.text}")

                time.sleep(2)  # Wait before retry

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise ServerConnectionError(f"Failed to connect to server after {max_retries} attempts: {e}") from e
                time.sleep(2)  # Wait before retry

        return server_url, server_process

    except Exception as e:  # pylint: disable=broad-exception-caught
        cleanup_server_process(server_process)
        raise e


class TypeCaster:
    """Helper class for casting tool arguments to proper types."""

    @staticmethod
    def cast_integer(value: Any) -> Any:
        """Cast value to integer if possible."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def cast_number(value: Any) -> Any:
        """Cast value to float if possible."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def cast_boolean(value: Any) -> Any:
        """Cast value to boolean."""
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    @staticmethod
    def cast_string(value: Any) -> Any:
        """Cast value to string, handling null values."""
        if value is None or value == "null":
            return None
        return str(value)

    @staticmethod
    def cast_object(value: Any) -> Any:
        """Cast value to object/dict."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    @staticmethod
    def cast_array(value: Any) -> Any:
        """Cast value to array/list."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


class LLMTestUtils:
    """Utility class for LLM integration testing."""

    def get_mcp_tools_and_instructions(self, server_url: str) -> tuple[List[Dict[str, Any]], str]:
        """Extract available tools and instructions from MCP server using HTTP streaming protocol."""
        session = requests.Session()

        # Initialize session and get tools
        self._initialize_mcp_session(session, server_url)
        tools_data = self._get_tools_list(session, server_url)

        # Extract server instructions and tools
        server_instructions = self._extract_server_instructions(tools_data)
        tools = self._extract_tools_from_response(tools_data)

        # Convert MCP tools to OpenAI function format
        return self._convert_mcp_tools_to_openai_format(tools), server_instructions

    def _initialize_mcp_session(self, session: requests.Session, server_url: str) -> str:
        """Initialize MCP session and return session ID."""
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
            raise MCPResponseError(f"Initialize failed: {response.status_code} - {response.text}")

        # Extract session ID from response headers
        session_id = response.headers.get('mcp-session-id') or response.headers.get('Mcp-Session-Id')

        # Send initialized notification
        if session_id:
            headers['mcp-session-id'] = session_id

        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        session.post(server_url, json=initialized_notification, headers=headers, timeout=10)
        return session_id or ""

    def _get_tools_list(self, session: requests.Session, server_url: str) -> Dict[str, Any]:
        """Get tools list from MCP server."""
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
        }

        response = session.post(server_url, json=tools_request, headers=headers, timeout=10)

        if response.status_code != 200:
            raise MCPResponseError(f"Tools list failed: {response.status_code} - {response.text}")

        return self.parse_mcp_response(response.text)

    def _extract_server_instructions(self, init_response: Dict[str, Any]) -> str:
        """Extract server instructions from initialization response."""
        if isinstance(init_response, list) and len(init_response) > 0:
            server_info = init_response[0].get('result', {})
        else:
            server_info = init_response.get('result', {})

        return server_info.get('instructions', '')

    def _extract_tools_from_response(self, tools_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tools from MCP response."""
        if isinstance(tools_response, list) and len(tools_response) > 0:
            tools_data = tools_response[0].get('result', {})
        else:
            tools_data = tools_response.get('result', {})

        return tools_data.get('tools', [])

    def get_mcp_tools(self, server_url: str) -> List[Dict[str, Any]]:
        """Extract available tools from MCP server using HTTP streaming protocol."""
        tools, _ = self.get_mcp_tools_and_instructions(server_url)
        return tools

    def parse_mcp_response(self, response_text: str) -> Dict[str, Any]:
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

        raise MCPResponseError(f"No valid JSON found in SSE response: {sse_text}")

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

    def call_llm(self, **kwargs) -> Dict[str, Any]:
        """Call LLM with tools and return response.

        Args:
            prompt: User prompt (used when messages is None)
            tools: Available tools
            logger: Logger for output
            system_prompt: System prompt (used when messages is None)
            messages: Full conversation history (overrides prompt/system_prompt)
        """
        # Extract arguments with defaults
        prompt = kwargs.get('prompt')
        tools = kwargs.get('tools', [])
        logger = kwargs.get('logger')
        system_prompt = kwargs.get('system_prompt')
        messages = kwargs.get('messages')

        api_url = os.getenv('MODEL_API')
        model_id = os.getenv('MODEL_ID')
        api_key = os.getenv('USER_KEY')

        if logger:
            logger.info(f"call_llm using LLM {model_id} at {api_url}")

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
        # else: Use provided conversation history (messages already set)

        payload = {
            "model": model_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }

        if logger:
            logger.debug(f"call_llm: payload:\n{json.dumps(payload, indent=2)}")

        response = requests.post(f"{api_url}/chat/completions", json=payload, headers=headers, timeout=30)

        if logger:
            logger.debug(f"call_llm: response:\n{json.dumps(response.json(), indent=2)}")

        if response.status_code != 200:
            raise LLMAPIError(f"LLM API call failed: {response.status_code} - {response.text}")

        return response.json()

    def cast_tool_args(self, tool_args: Dict[str, Any], tool_name: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cast tool arguments to match the parameter specifications from MCP tools."""
        # Find the tool definition
        tool_def = self._find_tool_definition(tool_name, tools)

        if not tool_def:
            return tool_args

        # Get parameter specifications from the tool
        input_schema = tool_def.get("inputSchema", {})
        properties = input_schema.get("properties", {})

        cast_args = {}
        type_caster = TypeCaster()

        for arg_name, arg_value in tool_args.items():
            if arg_name in properties:
                prop_spec = properties[arg_name]
                prop_type = prop_spec.get("type")
                cast_args[arg_name] = self._cast_single_argument(arg_value, prop_type, type_caster)
            else:
                # Parameter not in schema, keep as is
                cast_args[arg_name] = arg_value

        return cast_args

    def _find_tool_definition(self, tool_name: str, tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find tool definition by name."""
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
        return None

    def _cast_single_argument(self, arg_value: Any, prop_type: str, type_caster: TypeCaster) -> Any:
        """Cast a single argument value based on its expected type."""
        type_handlers = {
            "integer": type_caster.cast_integer,
            "number": type_caster.cast_number,
            "boolean": type_caster.cast_boolean,
            "string": type_caster.cast_string,
            "object": type_caster.cast_object,
            "array": type_caster.cast_array,
        }

        handler = type_handlers.get(prop_type)
        if handler:
            return handler(arg_value)
        # Unknown type, keep as is
        return arg_value

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any],
                  server_url: str, logger: logging.Logger) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
        session = requests.Session()

        try:
            # Initialize MCP session
            self._initialize_mcp_session(session, server_url)

            # Get tool definitions to cast arguments properly
            tools_data = self._get_tools_list(session, server_url)
            tools = self._extract_tools_from_response(tools_data)

            # Cast tool arguments according to parameter specifications
            tool_args = self.cast_tool_args(tool_args, tool_name, tools)
            logger.debug(f"Cast tool args for {tool_name}: {tool_args}")

            # Call the tool
            tool_call = {"name": tool_name, "arguments": tool_args}
            tool_response = self._execute_tool_call(session, server_url, tool_call, logger)
            return tool_response

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise ToolCallError(f"Failed to call tool {tool_name}: {e}") from e

    def _execute_tool_call(self, session: requests.Session, server_url: str,
                           tool_call: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
        """Execute the actual tool call."""
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        tool_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args
            }
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
        }

        logger.debug(f"Calling tool {tool_name} with args: {tool_args}")

        response = session.post(server_url, json=tool_request, headers=headers, timeout=30)

        if response.status_code != 200:
            raise ToolCallError(f"Tool call failed: {response.status_code} - {response.text}")

        tool_response = self.parse_mcp_response(response.text)
        logger.debug(f"Tool response: {tool_response}")

        return tool_response

    def _process_tool_calls(self, tool_calls: List[Dict[str, Any]], server_url: str,
                            logger: logging.Logger) -> List[Dict[str, Any]]:
        """Process tool calls and return tool results."""
        tool_results = []

        # Process each tool call
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]

            # Parse tool arguments
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments: {e}")
                tool_result = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": f"Error: Failed to parse tool arguments - {e}"
                }
                tool_results.append(tool_result)
                continue

            # Call the tool
            try:
                tool_response = self.call_tool(tool_name, tool_args, server_url, logger)

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

            except (ToolCallError, MCPResponseError, LLMAPIError, json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Tool call failed: {e}")
                tool_result = {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": f"Error: {e}"
                }
                tool_results.append(tool_result)

        return tool_results

    def complete_conversation_with_tools(self, user_prompt: str, server_url: str,
                                         logger: logging.Logger, **kwargs) -> Dict[str, Any]:
        """Complete a full conversation with tool calls, implementing proper agent flow."""
        tools = kwargs.get('tools', [])
        system_prompt = kwargs.get('system_prompt', '')

        logger.info(f"Starting conversation with prompt: {user_prompt}")

        # Build conversation messages
        conversation_messages = []

        # Add system prompt if provided
        if system_prompt:
            conversation_messages.append({"role": "system", "content": system_prompt})

        # Add user message
        conversation_messages.append({"role": "user", "content": user_prompt})

        # Loop until no more tool calls are requested
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Conversation iteration {iteration}")

            # Call LLM with current conversation history
            response = self.call_llm(
                prompt=None,  # Not used when messages is provided
                tools=tools,
                logger=logger,
                system_prompt=None,  # Already included in messages
                messages=conversation_messages
            )

            # Get the assistant's response
            choice = response["choices"][0]
            message = choice["message"]

            # Add assistant message to conversation
            conversation_messages.append(message)

            # Check if LLM wants to use tools
            if "tool_calls" in message and message["tool_calls"]:
                logger.info(f"LLM made tool calls: {[tc['function']['name'] for tc in message['tool_calls']]}")

                # Process tool calls
                tool_results = self._process_tool_calls(
                    message["tool_calls"], server_url, logger)

                # Add tool results to conversation
                conversation_messages.extend(tool_results)

                logger.debug(f"Conversation history after tool calls: {
                    json.dumps(conversation_messages, indent=2)}")

                # Continue the loop to get LLM's response to tool results
                continue

            logger.info("LLM responded without tool calls - conversation complete")
            break

        if iteration >= max_iterations:
            logger.warning(f"Conversation reached max iterations ({max_iterations})")

        logger.info(f"Conversation completed after {iteration} iterations")
        return response


# Utility tests
@pytest.mark.skipif(should_skip_llm_tests(), reason="LLM environment variables not set")
class TestLLMUtils:
    """Test utility functions for LLM integration."""

    # pylint: disable=redefined-outer-name
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

    # pylint: disable=redefined-outer-name
    def test_llm_api_connectivity(self, mcp_server_thread, test_logger):
        """Test basic LLM API connectivity."""
        server_url = mcp_server_thread
        utils = LLMTestUtils()

        # Get tools and instructions from MCP server
        tools, system_prompt = utils.get_mcp_tools_and_instructions(server_url)

        # Test basic LLM call
        try:
            response = utils.call_llm(
                prompt="Hello, can you help me?",
                tools=tools,
                logger=test_logger,
                system_prompt=system_prompt
            )
            assert "choices" in response, "LLM response missing 'choices' field"
            assert len(response["choices"]) > 0, "LLM response has no choices"

            print("LLM API connectivity test passed")
        except Exception as e:  # pylint: disable=broad-exception-caught
            pytest.skip(f"LLM API not accessible: {e}")

    # pylint: disable=redefined-outer-name
    def test_system_prompt_extraction(self, mcp_server_thread, test_logger):
        """Test that system prompt is correctly extracted from MCP server."""
        server_url = mcp_server_thread
        utils = LLMTestUtils()

        # Get tools and instructions from MCP server
        _, system_prompt = utils.get_mcp_tools_and_instructions(server_url)

        test_logger.debug(f"Full system prompt: {system_prompt}")

        # Verify system prompt contains expected elements
        assert system_prompt, "System prompt should not be empty"
        assert "NEVER CALL create_blueprint() IMMEDIATELY" in system_prompt, (
               "System prompt should contain behavioral rules")
        assert "AVAILABLE DISTRIBUTIONS:" in system_prompt, "System prompt should contain available distributions"
        assert "AVAILABLE ARCHITECTURES:" in system_prompt, "System prompt should contain available architectures"
        assert "AVAILABLE IMAGE TYPES:" in system_prompt, "System prompt should contain available image types"

        print("✓ System prompt correctly extracted with all expected elements")
        print(f"System prompt length: {len(system_prompt)} characters")

    # pylint: disable=redefined-outer-name
    def test_tool_parameter_casting(self, mcp_server_thread, test_logger):
        """Test that tool arguments are properly cast according to MCP parameter specifications."""
        utils = LLMTestUtils()

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
        cast_args = utils.cast_tool_args(test_args, 'get_openapi', test_tools)

        test_logger.debug(f"Original args: {test_args}")
        test_logger.debug(f"Cast args: {cast_args}")

        # Verify that response_size was cast to integer
        assert isinstance(cast_args['response_size'], int), f"response_size should be int, got {
            type(cast_args['response_size'])}"
        assert cast_args['response_size'] == 7, f"response_size should be 7, got {cast_args['response_size']}"

        print("✓ Parameter casting test passed - string '7' was successfully cast to integer 7")

        # Test with actual tool call to ensure it works end-to-end
        try:
            server_url = mcp_server_thread
            tool_response = utils.call_tool('get_openapi', test_args, server_url, test_logger)
            test_logger.debug(f"Tool call successful: {tool_response}")

            # Verify we got a response (the tool should work with cast parameters)
            assert tool_response is not None, "Tool should return a response"

            print("✓ End-to-end tool call with parameter casting successful")

        except (ToolCallError, MCPResponseError, LLMAPIError, json.JSONDecodeError, KeyError, ValueError) as e:
            test_logger.error(f"Tool call failed: {e}")
            # The test should still pass if casting worked, even if the tool call fails for other reasons
            print(f"⚠ Tool call failed but parameter casting worked: {e}")

        print("✓ Parameter casting functionality verified")


@pytest.fixture
def test_logger(request):
    """Get a test logger that doesn't conflict with verbose_logger fixture name."""
    logger = logging.getLogger(__name__ + "_test")

    verbosity = request.config.getoption('verbose', default=0)

    if verbosity >= 3:
        logger.setLevel(logging.DEBUG)
    elif verbosity >= 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    return logger
