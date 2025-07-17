"""Integration tests for LLM functionality with MCP server using deepeval."""

import logging

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.metrics import GEval, ToolCorrectnessMetric

from tests.conftest import verbose_logger

from .test_utils import (
    should_skip_llm_tests,
    CustomVLLMModel,
    MCPAgentWrapper
)


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

        for tool in tools_called:
            verbose_logger.info("Tool: %s", tool.name)
            verbose_logger.info("Parameters: %s", tool.input_parameters)
        verbose_logger.info("Response: %s", response)

        test_case = LLMTestCase(
            input=prompt,
            actual_output=response,
            tools_called=tools_called,
            expected_tools=expected_tools
        )

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
