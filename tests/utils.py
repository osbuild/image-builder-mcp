"""Utility functions for testing."""

import os
import json
import logging
import socket
import time
import asyncio
import multiprocessing
from typing import Dict, List, Any, Tuple, Optional

import requests


# Constants
DEFAULT_JSON_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream'
}


def should_skip_llm_tests() -> bool:
    """Check if LLM integration tests should be skipped."""
    required_vars = ['MODEL_API', 'MODEL_ID', 'USER_KEY']
    return not all(os.getenv(var) for var in required_vars)


def load_llm_configurations() -> Tuple[List[Dict[str, Optional[str]]], Optional[Dict[str, str]]]:
    """Load LLM configurations from test_config.json file."""
    config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_config.json')

    if not os.path.exists(config_file):
        # Fallback to environment variables for backward compatibility
        if not should_skip_llm_tests():
            return [{
                'name': 'Default Model',
                'MODEL_API': os.getenv('MODEL_API'),
                'MODEL_ID': os.getenv('MODEL_ID'),
                'USER_KEY': os.getenv('USER_KEY')
            }], None
        return [], None

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        configurations = []
        for llm_config in config.get('llm_configurations', []):
            # Substitute environment variables in configuration
            resolved_config: Dict[str, Optional[str]] = {}
            for key, value in llm_config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]  # Remove ${ and }
                    resolved_value = os.getenv(env_var)
                    if resolved_value:
                        resolved_config[key] = resolved_value
                    else:
                        # Skip this configuration if required env var is missing
                        break
                else:
                    resolved_config[key] = value

            # Only add configuration if all required variables are present
            if all(key in resolved_config and resolved_config[key]
                   for key in ['MODEL_API', 'MODEL_ID', 'USER_KEY']):
                configurations.append(resolved_config)
        guardian_llm: Optional[Dict[str, str]] = config.get('guardian_llm')
        return configurations, guardian_llm

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        logging.warning("Error loading test_config.json: %s. Falling back to environment variables.", e)
        # Fallback to environment variables
        if not should_skip_llm_tests():
            return [{
                'name': 'Default Model',
                'MODEL_API': os.getenv('MODEL_API'),
                'MODEL_ID': os.getenv('MODEL_ID'),
                'USER_KEY': os.getenv('USER_KEY')
            }], None
        return [], None


def should_skip_llm_matrix_tests() -> bool:
    """Check if LLM matrix tests should be skipped."""
    configurations, _ = load_llm_configurations()
    return len(configurations) == 0


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


class MCPError(Exception):
    """Exception raised for MCP-related errors."""


def get_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def create_mcp_init_request() -> dict:
    """Create standard MCP initialization request."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }


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
                test_request = create_mcp_init_request()
                response = requests.post(server_url, json=test_request, headers=DEFAULT_JSON_HEADERS, timeout=10)

                if response.status_code == 200:
                    break

                if attempt == max_retries - 1:
                    raise ServerConnectionError(
                        (f"Server not responding properly after {max_retries} "
                         f"attempts: {response.status_code} - {response.text}")
                    )

                time.sleep(2)  # Wait before retry

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise ServerConnectionError(f"Failed to connect to server after {max_retries} attempts: {e}") from e
                time.sleep(2)  # Wait before retry

        return server_url, server_process

    except Exception as e:  # pylint: disable=broad-exception-caught
        cleanup_server_process(server_process)
        raise e


def parse_mcp_response(response_text: str) -> Dict[str, Any]:
    """Parse MCP response which could be JSON or SSE format."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        # Try parsing as SSE format
        for line in response_text.split('\n'):
            if line.startswith('data: '):
                data_part = line[6:]  # Remove 'data: ' prefix
                try:
                    return json.loads(data_part)
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"No valid JSON found in response: {response_text}") from exc


def get_system_prompt_from_server(server_url: str) -> str:
    """Get system prompt from MCP server."""
    try:
        # Initialize with server
        init_request = create_mcp_init_request()
        response = requests.post(server_url, json=init_request, headers=DEFAULT_JSON_HEADERS, timeout=10)

        if response.status_code == 200:
            # Parse response to get instructions
            response_data = parse_mcp_response(response.text)
            if isinstance(response_data, dict) and 'result' in response_data:
                result = response_data['result']
                return result.get('instructions', '')
        return ""
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to get system prompt: %s", e)
        return ""


def make_llm_api_request(api_url: str, api_key: str, payload: Dict[str, Any]) -> str:
    """Make HTTP request to LLM API and return response content."""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        raise MCPError(f"LLM query failed: {e}") from e
    except (KeyError, IndexError) as e:
        raise MCPError(f"Unexpected LLM response format: {e}") from e


def call_llm_api(api_url: str, model_id: str, api_key: str, messages: List[Dict[str, str]],
                 temperature: float = 0.1) -> str:
    """Call LLM API with messages and return response content."""
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }

    return make_llm_api_request(api_url, api_key, payload)


class AgentWrapper:
    """Wrapper for agent functionality using raw HTTP requests."""

    def __init__(self, server_url: str, api_url: str, model_id: str, api_key: str):
        self.server_url = server_url
        self.api_url = api_url
        self.model_id = model_id
        self.api_key = api_key
        self.tools: List[Dict[str, Any]] = []
        self.system_prompt = ""
        self._get_server_info()

    def _get_server_info(self):
        """Get tools and system prompt from MCP server."""
        try:
            # Get system prompt using shared function
            self.system_prompt = get_system_prompt_from_server(self.server_url)

            # Get available tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            response = requests.post(self.server_url, json=tools_request, headers=DEFAULT_JSON_HEADERS, timeout=10)
            if response.status_code == 200:
                response_data = parse_mcp_response(response.text)
                if isinstance(response_data, dict) and 'result' in response_data:
                    self.tools = response_data['result'].get('tools', [])

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to get server info: %s", e)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse MCP response which could be JSON or SSE format."""
        return parse_mcp_response(response_text)

    def cast_tool_args(self, tool_args: Dict[str, Any], tool_name: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cast tool arguments according to their schema specifications."""
        # Find the tool schema
        tool_schema = None
        for tool in tools:
            if tool['name'] == tool_name:
                tool_schema = tool.get('inputSchema', {})
                break

        if not tool_schema:
            return tool_args

        # Cast arguments according to schema
        casted_args = {}
        properties = tool_schema.get('properties', {})

        for arg_name, arg_value in tool_args.items():
            if arg_name in properties:
                prop_type = properties[arg_name].get('type', 'string')
                casted_args[arg_name] = self._cast_by_type(arg_value, prop_type)
            else:
                casted_args[arg_name] = arg_value

        return casted_args

    def _cast_by_type(self, arg_value: Any, prop_type: str) -> Any:
        """Cast a value to the specified type."""
        try:
            if prop_type == 'integer':
                return int(float(arg_value))  # Handle string numbers like "5.0"
            if prop_type == 'number':
                return float(arg_value)
            if prop_type == 'boolean':
                if isinstance(arg_value, bool):
                    return arg_value
                return str(arg_value).lower() in ('true', '1', 'yes', 'on')
            return str(arg_value)
        except (ValueError, TypeError):
            # If casting fails, keep original value
            return arg_value

    def call_tool(self, tool_name: str, tool_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
        request_data = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args if tool_args else {}
            }
        }

        try:
            response = requests.post(self.server_url, json=request_data, headers=DEFAULT_JSON_HEADERS, timeout=30)
            response.raise_for_status()

            response_data = parse_mcp_response(response.text)
            if isinstance(response_data, dict) and 'result' in response_data:
                return response_data['result']
            if isinstance(response_data, dict) and 'error' in response_data:
                raise MCPError(f"Tool call error: {response_data['error']}")
            raise MCPError(f"Unexpected response format: {response_data}")

        except requests.exceptions.RequestException as e:
            raise MCPError(f"HTTP request failed: {e}") from e
        except Exception as e:
            raise MCPError(f"Tool call failed: {e}") from e

    def query_llm(self, user_msg: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Query the LLM with a message and optional conversation history."""
        # Build messages from conversation history and current message
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_msg})

        # Use shared LLM API function
        return call_llm_api(self.api_url, self.model_id, self.api_key, messages)


def pretty_print_conversation_history(conversation_history: List[Dict[str, Any]], llm_config: str) -> str:
    """Pretty print conversation history."""
    ret = ""
    for idx, message in enumerate(conversation_history):
        ret += f"-------- {llm_config}: Message {idx} ({message.get('role', 'unknown')}) --------\n"
        ret += f"Content: {message.get('content', '')}\n\n"
    return ret
