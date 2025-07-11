"""Test suite for LLM integration with the MCP server."""

import json
import os
import logging
import socket
import asyncio
import multiprocessing
import time
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

    if verbosity >= 2:
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


@pytest.mark.skipif(should_skip_llm_tests(), reason="LLM environment variables not set")
class TestLLMIntegration:
    """Test LLM integration with MCP server using HTTP streaming protocol."""

    @pytest.fixture(scope="session")
    def mcp_server_thread(self):  # pylint: disable=too-many-locals
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

    def get_mcp_tools(self, server_url: str) -> List[Dict[str, Any]]:
        """Extract available tools from MCP server using HTTP streaming protocol."""
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
            return self._convert_mcp_tools_to_openai_format(tools)

        except Exception as e:
            raise Exception(f"Failed to get MCP tools: {e}") from e  # pylint: disable=broad-exception-raised

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

    def call_llm(self, prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call LLM with tools and return response."""
        api_url = os.getenv('MODEL_API')
        model_id = os.getenv('MODEL_ID')
        api_key = os.getenv('USER_KEY')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "tools": tools,
            "tool_choice": "auto"
        }

        response = requests.post(f"{api_url}/chat/completions", json=payload, headers=headers, timeout=30)

        if response.status_code != 200:
            # pylint: disable=broad-exception-raised
            raise Exception(f"LLM API call failed: {response.status_code} - {response.text}")

        return response.json()

    def test_tool_definitions_extraction(self, mcp_server_thread):
        """Test that we can extract tool definitions from MCP server."""
        server_url = mcp_server_thread

        # Get tools from MCP server
        tools = self.get_mcp_tools(server_url)

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

    def test_llm_api_connectivity(self, mcp_server_thread):
        """Test basic LLM API connectivity."""
        server_url = mcp_server_thread

        # Get tools from MCP server
        tools = self.get_mcp_tools(server_url)

        # Test basic LLM call
        try:
            response = self.call_llm("Hello, can you help me?", tools)
            assert "choices" in response, "LLM response missing 'choices' field"
            assert len(response["choices"]) > 0, "LLM response has no choices"

            print("LLM API connectivity test passed")
        except Exception as e:  # pylint: disable=broad-exception-caught
            pytest.skip(f"LLM API not accessible: {e}")

    def test_rhel_image_creation_question(self, mcp_server_thread, verbose_logger):  # pylint: disable=redefined-outer-name
        """Test LLM tool selection for RHEL image creation."""
        server_url = mcp_server_thread

        # Get tools from MCP server
        tools = self.get_mcp_tools(server_url)

        verbose_logger.info(f"test_rhel_image_creation_question: tools:\n{json.dumps(tools, indent=2)}")

        # Ask about creating RHEL image
        prompt = "Can you create a RHEL 9 image for me?"

        response = self.call_llm(prompt, tools)

        verbose_logger.info(f"test_rhel_image_creation_question: LLM response:\n{json.dumps(response, indent=2)}")

        # Check if LLM wants to use tools
        choice = response["choices"][0]
        message = choice["message"]

        # Look for tool calls in response
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            tool_names = [call["function"]["name"] for call in tool_calls]

            # Check if LLM selected relevant tools
            expected_tools = ["create_blueprint", "get_openapi"]
            found_relevant = any(tool in tool_names for tool in expected_tools)

            assert "create_blueprint" not in tool_names, (
                "LLM should not select create_blueprint tool, but rather ask for more information"
            )

            assert found_relevant, f"LLM didn't select relevant tools. Selected: {tool_names}"
            print(f"LLM correctly selected tools: {tool_names}")
        else:
            print("LLM responded without tool calls")
            # TBD check if LLM asks for more information

    def test_image_build_status_question(self, mcp_server_thread):
        """Test LLM tool selection for image build status."""
        server_url = mcp_server_thread

        # Get tools from MCP server
        tools = self.get_mcp_tools(server_url)

        # Ask about image build status
        prompt = "What is the status of my latest image build?"

        response = self.call_llm(prompt, tools)

        # Check if LLM wants to use tools
        choice = response["choices"][0]
        message = choice["message"]

        # Look for tool calls in response
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            tool_names = [call["function"]["name"] for call in tool_calls]

            # Check if LLM selected relevant tools
            expected_tools = ["get_composes", "get_compose_details"]
            found_relevant = any(tool in tool_names for tool in expected_tools)

            assert found_relevant, f"LLM didn't select relevant tools. Selected: {tool_names}"
            print(f"LLM correctly selected tools: {tool_names}")
        else:
            print("LLM responded without tool calls - this might be expected behavior")
