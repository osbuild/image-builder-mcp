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
        self.system_prompt = ""
        self.session_id: Optional[str] = None
        # Initialize custom LLM for agent interactions
        self.custom_llm = CustomVLLMModel(api_url=api_url, model_id=model_id, api_key=api_key)
        # NOTE: this is only for manual testing if LLMs just have the types wrong
        # rather note which LLMs don't provide proper types or improve system prompt
        # to convince the LLM to use the correct types
        self.enable_tool_casting = False
        self._initialize()

    def _initialize(self):
        """Initialize MCP session and get available tools."""
        self._init_mcp_session()
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
        raise MCPError(f"No valid SSE compatible JSON found in response: {sse_text}")

    def _get_tools_list(self) -> Dict[str, Any]:
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

        response = self.session.post(self.server_url, json=tools_request, headers=headers, timeout=10)

        if response.status_code != 200:
            raise MCPError(f"Tools list failed: {response.status_code} - {response.text}")

        return self.parse_mcp_response(response.text)

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

    def cast_tool_args(self,
                       tool_args: Dict[str, Any],
                       tool_name: str,
                       tools: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
        """Cast tool arguments to match the parameter specifications from MCP tools.

        Returns:
            Tuple of (casted_args, any_casted)
        """
        # Find the tool definition
        tool_def = self._find_tool_definition(tool_name, tools)

        if not tool_def:
            return tool_args, False

        # Get parameter specifications from the tool
        input_schema = tool_def.get("inputSchema", {})
        properties = input_schema.get("properties", {})

        cast_args = {}
        type_caster = TypeCaster()

        any_casted = False
        for arg_name, arg_value in tool_args.items():
            if arg_name in properties:
                prop_spec = properties[arg_name]
                prop_type = prop_spec.get("type")
                cast_args[arg_name] = self._cast_single_argument(arg_value, prop_type, type_caster)
                if cast_args[arg_name] != arg_value:
                    any_casted = True
            else:
                # Parameter not in schema, keep as is
                cast_args[arg_name] = arg_value

        return cast_args, any_casted

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

    def _extract_tools_from_response(self, tools_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tools from MCP response."""
        if isinstance(tools_response, list) and len(tools_response) > 0:
            tools_data = tools_response[0].get('result', {})
        else:
            tools_data = tools_response.get('result', {})

        return tools_data.get('tools', [])

    def call_tool(self, tool_name: str, tool_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

        if self.enable_tool_casting:
            # Get tool definitions to cast arguments properly
            tools_data = self._get_tools_list()
            tools = self._extract_tools_from_response(tools_data)
            # Cast tool arguments according to parameter specifications
            tool_args, any_casted = self.cast_tool_args(tool_args or {}, tool_name, tools)
            if any_casted:
                warnings.warn(f"Tool {tool_name} arguments needed to be casted.")

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
        # print(f"LLM payload: {payload}")
        # print(f"LLM response: {response.text}")

        if response.status_code != 200:
            raise MCPError(f"LLM API call failed: {response.status_code} - {response.text}")

        return response.json()

    def _process_tool_calls(self, tool_calls_data: List[ToolCall]) -> List[ToolCall]:
        """Process tool calls and return ToolCall objects."""
        tools_called = []

        for tool_call in tool_calls_data:
            tool_name = tool_call.name
            tool_args = tool_call.input_parameters

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

    def query_with_messages(self, role_conent_map: List[Dict[str, str]],
                            conversation_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[
                                str, List[ToolCall], List[Dict[str, Any]]]:
        """Check LLM tool intentions without executing tools - for behavioral testing.

        Args:
            role_conent_map: List of role-content mappings for the current messages
            conversation_history: Previous conversation history (optional)
            answer_mode: If True, the LLM should consume the conversation history containing
                         to tool call results and return the final answer.

        Returns:
            Tuple of (response_content, tools_intended, updated_conversation_history)
        """
        # Initialize conversation history if not provided
        if conversation_history is None:
            conversation_history = []

        # Prepare messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add conversation history
        messages.extend(conversation_history)

        for message in role_conent_map:
            # although there should only be one message
            # for clarity on the order - we iterate not to miss any messages
            for role, content in message.items():
                messages.append({"role": role, "content": content})

        # Call LLM API
        llm_response = self._call_llm_api(messages)
        choice = llm_response["choices"][0]
        message = choice["message"]

        # The LLM call can return None for content
        # Tool metrics will fail if we return None
        final_content = message.get("content") or ""
        tools_intended = []

        # Extract tool intentions without executing them
        if "tool_calls" in message and message["tool_calls"]:
            # the return type is fine but comming from the generic interface
            tools_intended = self._extract_tool_intentions(message["tool_calls"])  # type: ignore[arg-type]

        # Update conversation history with the new exchange
        updated_history = conversation_history.copy()

        # Add the user message(s) to history
        for msg in role_conent_map:
            for role, content in msg.items():
                updated_history.append({"role": role, "content": content})

        # Add the complete assistant response to history (preserving tool_calls format)
        assistant_message = {
            "role": "assistant",
            "content": final_content
        }

        # Add tool calls to the assistant message if any
        if "tool_calls" in message and message["tool_calls"]:
            assistant_message["tool_calls"] = message["tool_calls"]

        updated_history.append(assistant_message)

        return final_content, tools_intended, updated_history

    def execute_tools_with_messages(self, role_conent_map: List[Dict[str, str]],
                                    conversation_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[
                                        str, List[ToolCall], List[Dict[str, Any]]]:
        """Query the LLM with available tools, execute them and return the final answer.

        Args:
            role_conent_map: A dictionary of role to content mappings aka prompts.
                             The role can be "system", "user", "assistant" or "tool".
            conversation_history: Previous conversation history (optional)

        Returns:
            Tuple of (response_content, tools_called, updated_conversation_history)
        """

        final_content, tools_intended, updated_history = self.query_with_messages(role_conent_map, conversation_history)

        tools_called = self._process_tool_calls(tools_intended)

        # Add tool responses to conversation history in OpenAI format
        if tools_called and "tool_calls" in updated_history[-1]:
            # Get the tool_calls from the last assistant message
            assistant_tool_calls = updated_history[-1]["tool_calls"]

            # Add tool response messages for each tool call
            for i, tool_call in enumerate(tools_called):
                tool_call_id = assistant_tool_calls[i]["id"] if i < len(assistant_tool_calls) else f"tool_call_{i}"
                content = json.dumps(tool_call.output) if not isinstance(tool_call.output, str) else tool_call.output
                tool_response_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content
                }
                updated_history.append(tool_response_msg)

            # we need to call LLM again with tool content to get the final answer
            final_content, tools_intended, updated_history = self.query_with_messages([], updated_history)
        else:
            warnings.warn("No tools called, but this was expected")

        return final_content, tools_called, updated_history


def pretty_print_conversation_history(conversation_history: List[Dict[str, Any]], llm_config: str) -> str:
    """Pretty print the conversation history."""
    ret = ""
    for message in conversation_history:
        ret += f"-------- {llm_config}: {message['role']} --------\n"
        ret += f"{message}\n"
        ret += f"{message['content']}\n\n"
    return ret
