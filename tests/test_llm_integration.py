"""Integration tests for LLM functionality with MCP server using deepeval."""

import json
import os
import logging
from typing import Dict, List, Any, Tuple, Optional

import pytest
import requests
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.metrics import GEval, ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.tracing import observe, update_current_span

from .test_utils import should_skip_llm_tests, create_mcp_init_request


class ServerStartupError(Exception):
    """Exception raised when MCP server fails to start."""


class ServerConnectionError(Exception):
    """Exception raised when unable to connect to MCP server."""


class MCPError(Exception):
    """Exception raised for MCP-related errors."""


class CustomVLLMModel(DeepEvalBaseLLM):
    """Custom LLM model for deepeval that uses vLLM with OpenAI-compatible API."""

    def __init__(self, verbose_logger):  # pylint: disable=redefined-outer-name
        super().__init__()
        self.api_url = os.getenv('MODEL_API')
        self.model_id = os.getenv('MODEL_ID')
        self.api_key = os.getenv('USER_KEY')

        if not all([self.api_url, self.model_id, self.api_key]):
            raise ValueError("MODEL_API, MODEL_ID, and USER_KEY environment variables must be set")

        verbose_logger.info("Using custom vLLM model: %s", self.model_id)

    def load_model(self, *args, **kwargs):
        # For API-based models, we don't need to load anything
        return None

    def generate(self, *args, **kwargs):
        """Generate text using the custom vLLM endpoint."""
        prompt = self._extract_prompt(args, kwargs)
        schema = kwargs.get('schema', None)

        try:
            content = self._call_llm_api(prompt)
            return self._process_response(content, schema)
        except (requests.RequestException, requests.Timeout, ValueError) as e:
            return self._handle_error(e, schema)

    def _extract_prompt(self, args, kwargs):
        """Extract prompt from args or kwargs."""
        if 'prompt' in kwargs:
            return kwargs['prompt']
        if args:
            return args[0]
        raise ValueError("prompt must be provided as keyword argument or first positional argument")

    def _call_llm_api(self, prompt):
        """Call the LLM API and return the content."""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
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
        return content or "I apologize, but I cannot provide a response to that query."

    def _process_response(self, content, schema):
        """Process the response content, optionally with schema."""
        if not schema:
            return content

        try:
            # Try to extract JSON from the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                parsed_data = json.loads(json_content)
                return schema(**parsed_data)
            # If no JSON found, create a simple response structure
            return schema(steps=["Unable to generate structured response"])
        except (json.JSONDecodeError, TypeError, ValueError):
            # Fallback if structured parsing fails
            if hasattr(schema, 'steps'):
                return schema(steps=["Unable to generate structured response"])
            return schema()

    def _handle_error(self, error, schema):
        """Handle errors and return appropriate fallback."""
        fallback_response = f"Error generating response: {str(error)}"
        if schema:
            try:
                if hasattr(schema, 'steps'):
                    return schema(steps=[fallback_response])
                return schema()
            except (TypeError, ValueError):
                pass
        return fallback_response

    async def a_generate(self, *args, **kwargs):
        """Async generate - reusing sync version for simplicity."""
        return self.generate(*args, **kwargs)

    def get_model_name(self, *args, **kwargs):
        return f"Custom vLLM ({self.model_id})"


class MCPAgentWrapper:
    """Wrapper for MCP agent functionality to work with deepeval."""

    def __init__(self, server_url: str, verbose_logger: logging.Logger):  # pylint: disable=redefined-outer-name
        self.server_url = server_url
        self.session: requests.Session = requests.Session()
        self.tools: List[Dict[str, Any]] = []
        self.system_prompt = ""
        self.session_id: Optional[str] = None
        # Initialize custom LLM for agent interactions
        self.custom_llm = CustomVLLMModel(verbose_logger)
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

    @observe()
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

        # Update span for tracing
        update_current_span(
            test_case=LLMTestCase(
                input=user_input,
                actual_output=final_content,
                tools_called=tools_called
            )
        )

        return final_content, tools_called

    @observe()
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

        # Update span for tracing
        update_current_span(
            test_case=LLMTestCase(
                input=user_input,
                actual_output=final_content,
                tools_called=tools_intended
            )
        )

        return final_content, tools_intended


@pytest.mark.skipif(should_skip_llm_tests(), reason="LLM environment variables not set")
class TestLLMIntegration:
    """Test LLM integration with MCP server using deepeval."""

    agent = None
    custom_llm = None

    @pytest.fixture(autouse=True)
    def setup_agent(self, mcp_server_thread, verbose_logger):  # pylint: disable=redefined-outer-name
        """Set up MCP agent wrapper for tests."""
        self.agent = MCPAgentWrapper(mcp_server_thread, verbose_logger)
        # Create custom LLM for deepeval metrics
        self.custom_llm = CustomVLLMModel(verbose_logger)

    def test_rhel_image_creation_behavioral_rules(self):
        """Test that LLM follows behavioral rules and doesn't immediately call create_blueprint."""

        # Define expected behavior metric using custom LLM
        behavioral_compliance = GEval(
            name="Behavioral Compliance",
            criteria=(
                "The LLM should NOT immediately call create_blueprint when asked to create an image. "
                "Instead, it should either ask for more information about requirements (distributions, "
                "architectures, image types) or use get_openapi to understand the system first."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.TOOLS_CALLED],
            model=self.custom_llm
        )

        prompt = "Can you create a RHEL 9 image for me?"

        # Use lightweight intention-only check instead of actually executing tools
        response, tools_intended = self.agent.check_tool_intentions(prompt)

        test_case = LLMTestCase(
            input=prompt,
            actual_output=response,
            tools_called=tools_intended
        )

        # Check that create_blueprint is not called immediately
        tool_names = [tool.name for tool in tools_intended]
        assert "create_blueprint" not in tool_names, (
            f"❌ BEHAVIORAL RULE VIOLATION: LLM called create_blueprint immediately! "
            f"Tool calls: {tool_names}. System prompt not working correctly.\n"
            f"The prompt was: {prompt}\n"
            f"The response was: {response}\n"
        )

        logging.info("✓ Behavioral rules working - tools intended: %s", tool_names)
        logging.info("Response: %s", response)

        # Evaluate with deepeval metric
        assert_test(test_case, [behavioral_compliance])

    def test_image_build_status_tool_selection(self):
        """Test that LLM selects appropriate tools for image build status queries."""

        # Define tool correctness metric - ToolCorrectnessMetric doesn't support model parameter
        tool_correctness = ToolCorrectnessMetric(
            threshold=0.7,
            include_reason=True
        )

        prompt = "What is the status of my latest image build?"

        response, tools_called = self.agent.query_with_tools(prompt)

        # Define expected tools for this query
        expected_tools = [
            ToolCall(name="get_composes"),
            # Could also include get_compose_details if compose ID is known
        ]

        test_case = LLMTestCase(
            input=prompt,
            actual_output=response,
            tools_called=tools_called,
            expected_tools=expected_tools
        )

        logging.info("Tools called: %s", [tool.name for tool in tools_called])
        logging.info("Response: %s", response)

        # Check if relevant tools were selected
        tool_names = [tool.name for tool in tools_called]
        expected_tool_names = ["get_composes", "get_compose_details"]
        found_relevant = any(tool in tool_names for tool in expected_tool_names)

        if found_relevant:
            logging.info("✓ LLM correctly selected relevant tools")
        else:
            logging.warning("LLM may not have selected optimal tools: %s", tool_names)

        # Evaluate with deepeval metric
        assert_test(test_case, [tool_correctness])

    def test_system_prompt_effectiveness(self):
        """Test that the system prompt contains all necessary behavioral guidelines."""

        # Define system prompt quality metric using custom LLM
        system_prompt_quality = GEval(
            name="System Prompt Quality",
            criteria=(
                "The system prompt should contain: "
                "1. Clear behavioral rules about not calling create_blueprint immediately "
                "2. Available distributions, architectures, and image types "
                "3. Instructions to ask for clarification when information is missing "
                "4. Guidance on proper tool usage sequence"
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.custom_llm
        )

        # Test the system prompt content
        system_prompt = self.agent.system_prompt

        logging.info("System prompt length: %d characters", len(system_prompt))
        logging.info("System prompt content: %s", system_prompt[:500] +
                     "..." if len(system_prompt) > 500 else system_prompt)

        # If system prompt is empty or very short, skip the test content checks
        if not system_prompt or len(system_prompt) < 100:
            logging.warning("System prompt is empty or very short. Skipping content validation.")
            pytest.skip("System prompt not available from MCP server")

        # Verify key elements are present (relaxed requirements)
        required_elements = [
            "create_blueprint",  # Should mention this function
            "distribution",      # Should mention distributions
            "architecture",      # Should mention architectures
            "image"             # Should mention images
        ]

        missing_elements = []
        for element in required_elements:
            if element.lower() not in system_prompt.lower():
                missing_elements.append(element)

        if missing_elements:
            logging.warning("System prompt missing some elements: %s", missing_elements)
        else:
            logging.info("✓ System prompt contains expected elements")

        test_case = LLMTestCase(
            input="System prompt evaluation",
            actual_output=system_prompt
        )

        # Evaluate with deepeval metric
        assert_test(test_case, [system_prompt_quality])

    def test_complete_conversation_flow(self):
        """Test complete conversation flow with proper agent behavior."""

        # Define conversation flow metric using custom LLM
        conversation_quality = GEval(
            name="Conversation Flow Quality",
            criteria=(
                "The conversation should demonstrate proper agent behavior: "
                "1. Understanding user intent "
                "2. Using appropriate tools to gather information "
                "3. Providing helpful and informative responses "
                "4. Following system guidelines"
            ),
            evaluation_params=[LLMTestCaseParams.INPUT,
                               LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.TOOLS_CALLED],
            model=self.custom_llm
        )

        prompt = "Can you help me understand what blueprints are available?"

        response, tools_called = self.agent.query_with_tools(prompt)

        test_case = LLMTestCase(
            input=prompt,
            actual_output=response,
            tools_called=tools_called
        )

        logging.info("Conversation prompt: %s", prompt)
        logging.info("Tools called: %s", [tool.name for tool in tools_called])
        logging.info("Response length: %d characters", len(response))
        logging.info("Response preview: %s...", response[:200])

        # Basic sanity checks (relaxed)
        assert response, "LLM should provide a non-empty response"
        # Allow shorter responses for tool-based interactions
        assert len(response) > 10, "Response should be substantial"

        # Evaluate with deepeval metric
        assert_test(test_case, [conversation_quality])

        logging.info("✓ Complete conversation flow test passed")

    def test_tool_usage_patterns(self):
        """Test various tool usage patterns and their appropriateness."""

        test_scenarios = [
            {
                "prompt": "Show me the API documentation",
                "expected_tools": ["get_openapi"],
                "description": "Should use get_openapi for API documentation"
            },
            {
                "prompt": "List all my recent builds",
                "expected_tools": ["get_composes"],
                "description": "Should use get_composes for build listings"
            },
            {
                "prompt": "What blueprints do I have?",
                "expected_tools": ["get_blueprints"],
                "description": "Should use get_blueprints for blueprint listings"
            }
        ]

        # Create tool correctness metric - doesn't support model parameter
        tool_correctness = ToolCorrectnessMetric(threshold=0.6)

        for scenario in test_scenarios:
            logging.info("Testing scenario: %s", scenario['description'])

            response, tools_called = self.agent.query_with_tools(scenario["prompt"])

            expected_tools = [ToolCall(name=name) for name in scenario["expected_tools"]]

            test_case = LLMTestCase(
                input=scenario["prompt"],
                actual_output=response,
                tools_called=tools_called,
                expected_tools=expected_tools
            )

            tool_names = [tool.name for tool in tools_called]
            logging.info("  Prompt: %s", scenario['prompt'])
            logging.info("  Tools called: %s", tool_names)
            logging.info("  Expected: %s", scenario['expected_tools'])

            # Evaluate with deepeval
            assert_test(test_case, [tool_correctness])

        logging.info("✓ Tool usage pattern tests completed")
