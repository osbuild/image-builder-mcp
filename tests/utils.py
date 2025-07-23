"""Utility functions for testing."""

import os
import json
import logging
import socket
import time
import asyncio
import multiprocessing
from typing import Dict, List, Any, Tuple, Optional
import warnings

import requests
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ToolCall


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
