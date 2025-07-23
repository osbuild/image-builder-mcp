"""Utility functions for testing using LlamaIndex agent implementation."""

import os
import json
import logging
import socket
import time
import asyncio
import multiprocessing
from typing import Dict, List, Any, Tuple, Optional, Callable
import warnings

import requests
from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ToolCall

# LlamaIndex imports
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec, aget_tools_from_mcp_url
from llama_index.core.base.llms.types import ChatResponseGen, LLMMetadata
from llama_index.core.tools import FunctionTool


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
                # print(f"Model {self.model_id} replied for {payload}\ņwith {ret}")
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


class CustomLlamaIndexLLM(OpenAI):
    """Custom LlamaIndex LLM that wraps vLLM with OpenAI-compatible API."""

    def __init__(self, api_url: str, model_id: str, api_key: str, **kwargs):
        # Configure OpenAI client to use custom endpoint
        super().__init__(
            model=model_id,
            api_key=api_key,
            api_base=api_url,
            temperature=kwargs.get('temperature', 0.1),
            **kwargs
        )
        self._custom_model_id = model_id

    @property
    def metadata(self):
        """Override metadata to provide context window for custom models."""
        # Use a reasonable default context window for custom models
        return LLMMetadata(
            context_window=8192,  # Common context window size
            num_output=2048,
            is_chat_model=True,
            is_function_calling_model=True,  # Mark as function calling capable
            model_name=self._custom_model_id,
        )


class MCPAgentWrapper:
    """Wrapper for MCP agent functionality using LlamaIndex."""

    def __init__(self, server_url: str, api_url: str, model_id: str, api_key: str):
        self.server_url = server_url
        self.api_url = api_url
        self.model_id = model_id
        self.api_key = api_key
        self.tools: List[FunctionTool] = []
        self.system_prompt = ""
        self.agent: Optional[FunctionAgent] = None
        self.context: Optional[Context] = None

        # Initialize custom LLM for agent interactions (for deepeval compatibility)
        self.custom_llm = CustomVLLMModel(api_url=api_url, model_id=model_id, api_key=api_key)

        # Initialize LlamaIndex LLM
        self.llama_llm = CustomLlamaIndexLLM(api_url=api_url, model_id=model_id, api_key=api_key)

        # Run async initialization
        asyncio.run(self._initialize())

    async def _initialize(self):
        """Initialize MCP session and get available tools."""
        await self._init_mcp_tools()
        await self._setup_agent()

    async def _init_mcp_tools(self):
        """Initialize MCP tools using LlamaIndex MCP support."""
        try:
            # Create MCP client
            mcp_client = BasicMCPClient(self.server_url)

            # Get tools from MCP server using McpToolSpec
            mcp_tool_spec = McpToolSpec(client=mcp_client)
            self.tools = await mcp_tool_spec.to_tool_list_async()

            # Extract system prompt if available
            # Note: LlamaIndex MCP tools might not expose system prompt directly
            # We may need to make a direct call to get it
            self.system_prompt = await self._get_system_prompt(mcp_client)

            logging.info(f"Initialized {len(self.tools)} tools from MCP server")

        except Exception as e:
            logging.error(f"Failed to initialize MCP tools: {e}")
            raise

    async def _get_system_prompt(self, mcp_client: BasicMCPClient) -> str:
        """Get system prompt from MCP server."""
        try:
            # Make a direct initialization request to get system prompt
            init_request = create_mcp_init_request()

            # Use requests for direct HTTP call
            response = requests.post(self.server_url, json=init_request, headers=DEFAULT_JSON_HEADERS, timeout=10)

            if response.status_code == 200:
                # Parse response to get system prompt
                response_data = self._parse_response(response.text)
                if isinstance(response_data, dict) and 'result' in response_data:
                    result = response_data['result']
                    return result.get('instructions', '')

            return ""

        except Exception as e:
            logging.warning(f"Failed to get system prompt: {e}")
            return ""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse MCP response which could be JSON or SSE format."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try parsing as SSE format
            for line in response_text.split('\n'):
                if line.startswith('data: '):
                    data_part = line[6:]  # Remove 'data: ' prefix
                    try:
                        return json.loads(data_part)
                    except json.JSONDecodeError:
                        continue
            raise ValueError(f"No valid JSON found in response: {response_text}")

    async def _setup_agent(self):
        """Setup LlamaIndex agent with MCP tools."""
        self.agent = FunctionAgent(
            name="MCP Agent",
            description="Agent with MCP tools",
            llm=self.llama_llm,
            tools=self.tools,
            system_prompt=self.system_prompt if self.system_prompt else None,
        )
        self.context = Context(self.agent)

    def conversation_history_to_chat_messages(self, conversation_history: Optional[List[Dict[str, Any]]]) -> List[ChatMessage]:
        """Convert conversation history dict format to ChatMessage list."""
        if not conversation_history:
            return []

        messages = []
        for msg in conversation_history:
            messages.append(ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))
        return messages

    def _wrap_tools_with_tracker(self, track_calls: bool = True) -> Tuple[List[FunctionTool], List[ToolCall]]:
        """Wrap tools to either track calls or mock execution.

        Args:
            track_calls: If True, execute tools and track calls. If False, mock execution.

        Returns:
            Tuple of (wrapped_tools, tool_calls_list)
        """
        tools_called = []
        wrapped_tools = []

        for tool in self.tools:
            if track_calls:
                # Create tracked version that executes
                original_fn = tool.fn

                def create_tracked_fn(tool_name, orig_fn):
                    def tracked_fn(**kwargs):
                        result = orig_fn(**kwargs)
                        tools_called.append(ToolCall(
                            name=tool_name,
                            input_parameters=kwargs,
                            output=result
                        ))
                        return result
                    return tracked_fn

                wrapped_fn = create_tracked_fn(tool.metadata.name, original_fn)
            else:
                # Create mock version that doesn't execute
                def create_mock_fn(tool_name):
                    def mock_fn(**kwargs):
                        tools_called.append(ToolCall(
                            name=tool_name,
                            input_parameters=kwargs,
                            output="[Tool not executed - intention only]"
                        ))
                        return "[Tool not executed - intention only]"
                    return mock_fn

                wrapped_fn = create_mock_fn(tool.metadata.name)

            wrapped_tool = FunctionTool.from_defaults(
                fn=wrapped_fn,
                name=tool.metadata.name,
                description=tool.metadata.description
            )
            wrapped_tools.append(wrapped_tool)

        return wrapped_tools, tools_called

    def cast_tool_args(self, tool_args: Dict[str, Any], tool_name: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cast tool arguments - compatibility method for tests.

        Note: LlamaIndex handles type casting internally, so this just returns the args as-is.
        """
        return tool_args

    def call_tool(self, tool_name: str, tool_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Call a specific tool on the MCP server."""
        # Find the tool
        tool = None
        for t in self.tools:
            if t.metadata.name == tool_name:
                tool = t
                break

        if not tool:
            raise MCPError(f"Tool {tool_name} not found")

        try:
            # Call the tool directly
            result = tool.call(**tool_args) if tool_args else tool.call()
            return {"result": result}
        except Exception as e:
            raise MCPError(f"Tool call failed: {str(e)}") from e

    def query_with_messages(self, user_msg: str,
                            chat_history: Optional[List[ChatMessage]] = None) -> Tuple[
                                str, List[ToolCall], List[ChatMessage]]:
        """Check LLM tool intentions without executing tools - for behavioral testing.

        Args:
            user_msg: The user's message
            chat_history: Previous messages as ChatMessage objects
        """
        if chat_history is None:
            chat_history = []
        response, tools_intended, updated_history = asyncio.run(
            self._query_with_messages_async(user_msg, chat_history)
        )
        return response, tools_intended, updated_history

    async def _query_with_messages_async(self, user_msg: str,
                                         chat_history: List[ChatMessage]) -> Tuple[
                                             str, List[ToolCall], List[ChatMessage]]:
        """Async version of query_with_messages using ChatMessage list."""
        # Wrap tools for mock execution
        mock_tools, tools_intended = self._wrap_tools_with_tracker(track_calls=False)

        # Temporarily set mock tools
        original_tools = self.agent.tools
        self.agent.tools = mock_tools

        try:
            # Run agent with chat history
            response = await self.agent.run(user_msg, ctx=self.context, chat_history=chat_history)

            # Build updated chat history
            updated_history = chat_history + [
                ChatMessage(role="user", content=user_msg),
                ChatMessage(role="assistant", content=str(response))
            ]

            return str(response), tools_intended, updated_history

        finally:
            # Restore original tools
            self.agent.tools = original_tools

    def execute_tools_with_messages(self, user_msg: str,
                                    chat_history: Optional[List[ChatMessage]] = None) -> Tuple[
                                        str, List[ToolCall], List[ChatMessage]]:
        """Query the LLM with available tools, execute them and return the final answer.

        Args:
            user_msg: The user's message
            chat_history: Previous messages as ChatMessage objects
        """
        if chat_history is None:
            chat_history = []
        response, tools_called, updated_history = asyncio.run(
            self._execute_tools_with_messages_async(user_msg, chat_history)
        )
        return response, tools_called, updated_history

    async def _execute_tools_with_messages_async(self, user_msg: str,
                                                 chat_history: List[ChatMessage]) -> Tuple[
                                                     str, List[ToolCall], List[ChatMessage]]:
        """Async version of execute_tools_with_messages using ChatMessage list."""
        # Wrap tools for tracked execution
        tracked_tools, tools_called = self._wrap_tools_with_tracker(track_calls=True)

        # Temporarily set tracked tools
        self.agent.tools = tracked_tools

        try:
            # Run agent with chat history
            response = await self.agent.run(user_msg, ctx=self.context, chat_history=chat_history)

            # Build updated chat history
            updated_history = chat_history + [
                ChatMessage(role="user", content=user_msg),
                ChatMessage(role="assistant", content=str(response))
            ]

            return str(response), tools_called, updated_history

        finally:
            # Restore original tools
            self.agent.tools = self.tools

    def query_with_chat_messages(self, user_msg: str,
                                 chat_history: Optional[List[ChatMessage]] = None) -> Tuple[
                                     str, List[ToolCall], List[ChatMessage]]:
        """Query with ChatMessage objects directly - preferred method.

        Args:
            user_msg: The user's message
            chat_history: Previous messages as ChatMessage objects

        Returns:
            Tuple of (response, tools_intended, updated_chat_history)
        """
        if chat_history is None:
            chat_history = []

        return asyncio.run(self._query_with_chat_messages_async(user_msg, chat_history))

    async def _query_with_chat_messages_async(self, user_msg: str,
                                              chat_history: List[ChatMessage]) -> Tuple[
                                                  str, List[ToolCall], List[ChatMessage]]:
        """Async version using ChatMessage objects directly."""
        # Wrap tools for mock execution
        mock_tools, tools_intended = self._wrap_tools_with_tracker(track_calls=False)

        # Temporarily set mock tools
        original_tools = self.agent.tools
        self.agent.tools = mock_tools

        try:
            # Run agent with chat history
            response = await self.agent.run(user_msg, ctx=self.context, chat_history=chat_history)

            # Build updated chat history
            updated_history = chat_history + [
                ChatMessage(role="user", content=user_msg),
                ChatMessage(role="assistant", content=str(response))
            ]

            return str(response), tools_intended, updated_history

        finally:
            # Restore original tools
            self.agent.tools = original_tools

    def execute_with_chat_messages(self, user_msg: str,
                                   chat_history: Optional[List[ChatMessage]] = None) -> Tuple[
                                       str, List[ToolCall], List[ChatMessage]]:
        """Execute tools with ChatMessage objects directly - preferred method.

        Args:
            user_msg: The user's message
            chat_history: Previous messages as ChatMessage objects

        Returns:
            Tuple of (response, tools_called, updated_chat_history)
        """
        if chat_history is None:
            chat_history = []

        return asyncio.run(self._execute_with_chat_messages_async(user_msg, chat_history))

    async def _execute_with_chat_messages_async(self, user_msg: str,
                                                chat_history: List[ChatMessage]) -> Tuple[
                                                    str, List[ToolCall], List[ChatMessage]]:
        """Async version using ChatMessage objects directly."""
        # Wrap tools for tracked execution
        tracked_tools, tools_called = self._wrap_tools_with_tracker(track_calls=True)

        # Temporarily set tracked tools
        self.agent.tools = tracked_tools

        try:
            # Run agent with chat history
            response = await self.agent.run(user_msg, ctx=self.context, chat_history=chat_history)

            # Build updated chat history
            updated_history = chat_history + [
                ChatMessage(role="user", content=user_msg),
                ChatMessage(role="assistant", content=str(response))
            ]

            return str(response), tools_called, updated_history

        finally:
            # Restore original tools
            self.agent.tools = self.tools


def pretty_print_chat_history(chat_history: List[ChatMessage], llm_config: str) -> str:
    """Pretty print ChatMessage history - preferred method."""
    ret = ""
    for message in chat_history:
        ret += f"-------- {llm_config}: {message.role} --------\n"
        ret += f"Content: {message.content}\n\n"
    return ret
