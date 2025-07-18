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


class CustomVLLMModel(DeepEvalBaseLLM):
    """Custom LLM model for deepeval that uses vLLM with OpenAI-compatible API."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ):
        if not api_url:
            raise ValueError("api_url must be provided for CustomVLLMModel")
        if not model_id:
            raise ValueError("model_id must be provided for CustomVLLMModel")

        self.api_url = api_url
        self.model_id = model_id or "default"
        self.api_key = api_key or ""

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        super().__init__(self.model_id)

    # pylint: disable=arguments-differ
    def generate(  # type: ignore[override]
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> str:

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        ret = content
        if schema:
            try:
                # remove markdown code block markers
                content = content.replace("```json", "").replace("```", "")
                ret = schema.model_validate_json(content)
                print(f"Model {self.model_id} replied for {payload}\ņwith {ret}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_message = (f"The LLM {self.model_id} was expected to return a valid JSON object "
                                 f"compatible with the schema {schema}. but it returned {content}."
                                 f"Error: {e}")
                raise ValueError(error_message) from e

            return ret

        return content

    # pylint: disable=arguments-differ
    async def a_generate(  # type: ignore[override]
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> str:
        # For simplicity, reuse sync version
        return self.generate(prompt, schema)

    def load_model(self):
        # For API-based models, we don't need to load anything
        return None

    def get_model_name(self):
        return f"{self.model_id} (vLLM)"


class MCPAgentWrapper:
    """Wrapper for MCP agent functionality to work with deepeval."""

    def __init__(self, server_url: str, api_url: str, model_id: str, api_key: str):
        self.server_url = server_url
        self.session: requests.Session = requests.Session()
        self.tools: List[Dict[str, Any]] = []
        self.system_prompt = "You are a helpful assistant that can use the given tools to help the user."
        self.session_id: Optional[str] = None
        # Initialize custom LLM for agent interactions
        self.custom_llm = CustomVLLMModel(api_url=api_url, model_id=model_id, api_key=api_key)
        self._initialize()

    def _initialize(self):
        """Initialize MCP session and get available tools."""
        self._init_mcp_session()
        self._get_tools_list()
        self._log_final_state()

    def _init_mcp_session(self):
        """Initialize the MCP session."""
        init_request = create_mcp_init_request()
        init_request["params"]["clientInfo"]["name"] = "deepeval-test-client"

        headers = self._get_default_headers()
        logging.info("Sending MCP init request: %s", init_request)
        response = self.session.post(self.server_url, json=init_request, headers=headers, timeout=10)

        self._log_response("Init", response)
        if response.status_code != 200:
            raise ServerConnectionError(f"Failed to initialize MCP session: {response.status_code}")

        self._extract_session_id(response)
        self._extract_system_prompt(response)

    def _send_initialized_notification(self):
        """Send initialized notification."""
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        headers = self._get_headers_with_session()

        logging.info("Sending initialized notification: %s", notification)
        response = self.session.post(self.server_url, json=notification, headers=headers, timeout=10)
        self._log_response("Notification", response)
        return response

    def _get_tools_list(self):
        """Get the list of available tools."""
        notify_response = self._send_initialized_notification()

        if notify_response.status_code in [200, 202]:
            tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            headers = self._get_headers_with_session()

            logging.info("Sending tools/list request: %s", tools_request)
            response = self.session.post(self.server_url, json=tools_request, headers=headers, timeout=10)
            self._log_response("Tools", response)

            if response.status_code == 200:
                tools_data = self._parse_response(response.text)
                logging.info("Parsed tools data: %s", tools_data)
                self._extract_tools_and_instructions(tools_data)
            else:
                logging.error("Failed to get tools list: %d - %s", response.status_code, response.text)
        else:
            logging.warning("Notification failed, skipping tools/list request. Status: %d, Response: %s",
                            notify_response.status_code, notify_response.text)

    def _get_default_headers(self):
        """Get default headers for requests."""
        return {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}

    def _get_headers_with_session(self):
        """Get headers with session ID if available."""
        headers = self._get_default_headers()
        if self.session_id:
            headers['mcp-session-id'] = self.session_id
        return headers

    def _log_response(self, prefix, response):
        """Log response details."""
        logging.info("%s response status: %d", prefix, response.status_code)
        logging.info("%s response text: %s", prefix, response.text)
        if hasattr(response, 'headers'):
            logging.info("%s response headers: %s", prefix, dict(response.headers))
        logging.info("Session cookies after %s: %s", prefix.lower(), dict(self.session.cookies))

    def _extract_session_id(self, response):
        """Extract session ID from response."""
        self.session_id = response.headers.get('mcp-session-id')
        if self.session_id:
            logging.info("Extracted session ID: %s", self.session_id)
            self.session.headers.update({'mcp-session-id': self.session_id})
        else:
            logging.warning("No session ID found in init response headers")

    def _extract_system_prompt(self, response):
        """Extract system prompt from init response."""
        init_data = self._parse_response(response.text)
        logging.info("Parsed init data: %s", init_data)

        if isinstance(init_data, dict) and 'result' in init_data:
            result = init_data['result']
            self.system_prompt = result.get('instructions', '')
            prompt_preview = (self.system_prompt[:200] + "..."
                              if len(self.system_prompt) > 200 else self.system_prompt)
            logging.info("Extracted system prompt from init: %s", prompt_preview)

    def _log_final_state(self):
        """Log final initialization state."""
        prompt_preview = (self.system_prompt[:200] + "..."
                          if len(self.system_prompt) > 200 else self.system_prompt)
        logging.info("Final system prompt: %s", prompt_preview)
        logging.info("Number of tools discovered: %d", len(self.tools))
        for i, tool in enumerate(self.tools):
            logging.info("Tool %d: %s", i, tool)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
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

    def _extract_tools_and_instructions(self, response: Dict[str, Any]):
        """Extract tools and system instructions from MCP response."""
        if isinstance(response, list) and len(response) > 0:
            result = response[0].get('result', {})
        else:
            result = response.get('result', {})

        # Extract tools
        raw_tools = result.get('tools', [])
        self.tools = []
        for tool in raw_tools:
            self.tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {}
                    })
                }
            })

        # Only extract system instructions if we don't already have them
        # (tools/list response may not have instructions, so preserve init instructions)
        if not self.system_prompt and 'instructions' in result:
            self.system_prompt = result.get('instructions', '')

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
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

        # Add session ID if available
        if self.session_id:
            headers['mcp-session-id'] = self.session_id

        response = self.session.post(self.server_url, json=tool_request, headers=headers, timeout=30)

        if response.status_code != 200:
            raise MCPError(f"Tool call failed: {response.status_code} - {response.text}")

        return self._parse_response(response.text)

    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the LLM API with the given messages using custom vLLM endpoint."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.custom_llm.api_key}'
        }

        payload = {
            "model": self.custom_llm.model_id,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
            "temperature": 0.1
        }

        response = requests.post(f"{self.custom_llm.api_url}/chat/completions",
                                 json=payload, headers=headers, timeout=30)
        print(f"LLM payload: {payload}")
        print(f"LLM response: {response.text}")

        if response.status_code != 200:
            raise MCPError(f"LLM API call failed: {response.status_code} - {response.text}")

        return response.json()

    def _process_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[ToolCall]:
        """Process tool calls and return ToolCall objects."""
        tools_called = []

        for tool_call in tool_calls_data:
            tool_name = tool_call["function"]["name"]
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            # Actually call the tool
            try:
                tool_response = self.call_tool(tool_name, tool_args)
                if isinstance(tool_response, list) and len(tool_response) > 0:
                    result = tool_response[0].get('result', {})
                else:
                    result = tool_response.get('result', {})

                tools_called.append(ToolCall(
                    name=tool_name,
                    input_parameters=tool_args,
                    output=result
                ))
            except MCPError as exc:
                tools_called.append(ToolCall(
                    name=tool_name,
                    input_parameters=tool_args,
                    output=f"Error: {exc}"
                ))

        return tools_called

    def _extract_tool_intentions(self, tool_calls_data: List[Dict[str, Any]]) -> List[ToolCall]:
        """Extract tool intentions without executing them - for behavioral testing."""
        tools_intended = []

        for tool_call in tool_calls_data:
            tool_name = tool_call["function"]["name"]
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            # Create ToolCall object without executing the tool
            tools_intended.append(ToolCall(
                name=tool_name,
                input_parameters=tool_args,
                output="[Tool not executed - intention only]"
            ))

        return tools_intended

    def query_with_tools(self, user_input: str) -> Tuple[str, List[ToolCall]]:
        """Query the LLM with available tools and return response and tools used."""
        # Prepare messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})

        # Call LLM API
        llm_response = self._call_llm_api(messages)
        choice = llm_response["choices"][0]
        message = choice["message"]

        final_content = message.get("content", "")
        tools_called = []

        # Process tool calls if any
        if "tool_calls" in message and message["tool_calls"]:
            tools_called = self._process_tool_calls(message["tool_calls"])
            # If there's no content but tools were called, provide a meaningful response
            if not final_content:
                tool_names = [tool.name for tool in tools_called]
                final_content = f"I've called the following tools to help answer your question: {', '.join(tool_names)}"

        # Ensure we always have some content for deepeval
        if not final_content:
            final_content = "I understand your request, but I cannot provide a specific response at this time."

        return final_content, tools_called

    def check_tool_intentions(self, user_input: str) -> Tuple[str, List[ToolCall]]:
        """Check LLM tool intentions without executing tools - for behavioral testing."""
        # Prepare messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_input})

        # Call LLM API
        llm_response = self._call_llm_api(messages)
        choice = llm_response["choices"][0]
        message = choice["message"]

        final_content = message.get("content", "")
        tools_intended = []

        # Extract tool intentions without executing them
        if "tool_calls" in message and message["tool_calls"]:
            tools_intended = self._extract_tool_intentions(message["tool_calls"])
            # If there's no content but tools were intended, provide a meaningful response
            if not final_content:
                tool_names = [tool.name for tool in tools_intended]
                final_content = f"I would call the following tools to help answer your question: {
                    ', '.join(tool_names)}"

        # Ensure we always have some content for deepeval
        if not final_content:
            final_content = ("I understand your request, but I cannot determine the appropriate "
                             "response or tools to use.")

        return final_content, tools_intended
