"""Utility functions for testing using LlamaIndex agent implementation."""

import logging
import asyncio
from typing import Dict, List, Any, Tuple, Optional

from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ToolCall

# LlamaIndex imports
# pylint: disable=import-error
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.tools import FunctionTool

# Import shared utilities from utils
from .utils import (
    MCPError,
    parse_mcp_response,
    get_system_prompt_from_server,
    call_llm_api,
    make_llm_api_request
)


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
    ) -> Any:

        # For simple cases without schema, use shared function
        if not schema:
            messages = [{"role": "user", "content": prompt}]
            return call_llm_api(self.api_url, self.model_id, self.api_key, messages, self.temperature)

        # For schema validation, use shared HTTP request function
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        content = make_llm_api_request(self.api_url, self.api_key, payload)

        if schema:
            try:
                # remove markdown code block markers
                content = content.replace("```json", "").replace("```", "")
                return schema.model_validate_json(content)
                # print(f"Model {self.model_id} replied for {payload}\ņwith {ret}")
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_message = (f"The LLM {self.model_id} was expected to return a valid JSON object "
                                 f"compatible with the schema {schema}. but it returned {content}."
                                 f"Error: {e}")
                raise ValueError(error_message) from e

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


class CustomLlamaIndexLLM(OpenAI):  # pylint: disable=too-few-public-methods
    """Custom LlamaIndex LLM that wraps vLLM with OpenAI-compatible API."""

    def __init__(self, api_url: str, model_id: str, api_key: str, system_prompt: str = "", **kwargs):
        # Configure OpenAI client to use custom endpoint
        super().__init__(
            model=model_id,
            api_key=api_key,
            api_base=api_url,
            temperature=kwargs.get('temperature', 0.1),
            **kwargs
        )
        self._custom_model_id = model_id
        self._system_prompt = system_prompt

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


class MCPAgentWrapper:  # pylint: disable=too-many-instance-attributes
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

        # Re-initialize LlamaIndex LLM with the system prompt
        self.llama_llm = CustomLlamaIndexLLM(
            api_url=self.api_url,
            model_id=self.model_id,
            api_key=self.api_key,
            system_prompt=self.system_prompt
        )

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

            logging.info("Initialized %d tools from MCP server", len(self.tools))

        except Exception as e:
            logging.error("Failed to initialize MCP tools: %s", e)
            raise

    async def _get_system_prompt(self, mcp_client: BasicMCPClient) -> str:  # pylint: disable=unused-argument
        """Get system prompt from MCP server."""
        return get_system_prompt_from_server(self.server_url)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse MCP response which could be JSON or SSE format."""
        return parse_mcp_response(response_text)

    async def _setup_agent(self):
        """Setup LlamaIndex agent with MCP tools."""
        self.agent = FunctionAgent(
            name="MCP Agent",
            description="Agent with MCP tools",
            llm=self.llama_llm,
            tools=self.tools,
        )
        self.context = Context(self.agent)

    def conversation_history_to_chat_messages(
        self, conversation_history: Optional[List[Dict[str, Any]]]
    ) -> List[ChatMessage]:
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

    def cast_tool_args(
        self, tool_args: Dict[str, Any], _tool_name: str, _tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
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
        if self.agent is None:
            raise ValueError("Agent not initialized")

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
        if self.agent is None:
            raise ValueError("Agent not initialized")

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
        if self.agent is None:
            raise ValueError("Agent not initialized")

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
        if self.agent is None:
            raise ValueError("Agent not initialized")

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
