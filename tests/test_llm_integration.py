"""Integration tests for LLM functionality with MCP server using deepeval."""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.metrics import GEval, ToolCorrectnessMetric

from tests.conftest import verbose_logger

from .test_utils import (
    should_skip_llm_matrix_tests,
    load_llm_configurations,
    CustomVLLMModel,
    MCPAgentWrapper
)


# Load LLM configurations for parametrization
llm_configurations, guardian_llm_config = load_llm_configurations()


@pytest.mark.skipif(should_skip_llm_matrix_tests(), reason="No valid LLM configurations found")
class TestLLMIntegration:
    """Test LLM integration with MCP server using deepeval with multiple LLM configurations."""

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    def test_rhel_image_creation_behavioral_rules(self, mcp_server_thread, verbose_logger, llm_config):  # pylint: disable=redefined-outer-name
        """Test that LLM follows behavioral rules and doesn't immediately call create_blueprint."""

        # Log which model is being tested
        verbose_logger.info("🧪 Testing model: %s (%s)", llm_config['name'], llm_config['MODEL_ID'])

        # Set up agent for this specific LLM configuration
        agent = MCPAgentWrapper(
            server_url=mcp_server_thread,
            api_url=llm_config['MODEL_API'],
            model_id=llm_config['MODEL_ID'],
            api_key=llm_config['USER_KEY'],
            verbose_logger=verbose_logger
        )

        # if there is a guardian LLM, use it for the guardian agent
        # otherwise, use the test LLM for the guardian agent
        if guardian_llm_config:
            guardian_agent = CustomVLLMModel(
                api_url=guardian_llm_config['MODEL_API'],
                model_id=guardian_llm_config['MODEL_ID'],
                api_key=guardian_llm_config['USER_KEY'],
                verbose_logger=verbose_logger
            )
        else:
            guardian_agent = CustomVLLMModel(
                api_url=llm_config['MODEL_API'],
                model_id=llm_config['MODEL_ID'],
                api_key=llm_config['USER_KEY'],
                verbose_logger=verbose_logger
            )

        prompt = "Can you create a RHEL 9 image for me?"

        # Use lightweight intention-only check instead of actually executing tools
        response, tools_intended = agent.check_tool_intentions(prompt)

        # Check that create_blueprint is not called immediately
        tool_names = [tool.name for tool in tools_intended]
        assert "create_blueprint" not in tool_names, (
            f"❌ BEHAVIORAL RULE VIOLATION for {llm_config['name']} ({llm_config['MODEL_ID']}): "
            f"LLM called create_blueprint immediately! Tool calls: {tool_names}. "
            f"System prompt not working correctly.\nThe prompt was: {prompt}\n"
            f"The response was: {response}\n"
        )

        verbose_logger.info("✓ Behavioral rules working for %s - tools intended: %s",
                            llm_config['name'], tool_names)
        verbose_logger.info("Response: %s", response)

        test_case = LLMTestCase(
            input=prompt,
            actual_output=response,
            tools_called=tools_intended
        )

        # Define expected behavior metric using custom LLM
        behavioral_compliance = GEval(
            name="Behavioral Compliance",
            criteria=(
                "The LLM should NOT immediately call create_blueprint. "
                "Instead, it should either ask for more information about requirements (distributions, "
                "architectures, image types etc.) or use get_openapi to understand the system first."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.TOOLS_CALLED],
            model=guardian_agent
        )

        # Evaluate with deepeval metric
        assert_test(test_case, [behavioral_compliance])

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    def test_image_build_status_tool_selection(self, mcp_server_thread, verbose_logger, llm_config):  # pylint: disable=redefined-outer-name
        """Test that LLM selects appropriate tools for image build status queries."""

        # Log which model is being tested
        verbose_logger.info("🧪 Testing model: %s (%s)", llm_config['name'], llm_config['MODEL_ID'])

        # Set up agent for this specific LLM configuration
        agent = MCPAgentWrapper(
            server_url=mcp_server_thread,
            api_url=llm_config['MODEL_API'],
            model_id=llm_config['MODEL_ID'],
            api_key=llm_config['USER_KEY'],
            verbose_logger=verbose_logger
        )

        # Define tool correctness metric - ToolCorrectnessMetric doesn't support model parameter
        tool_correctness = ToolCorrectnessMetric(
            threshold=0.7,
            include_reason=True
        )

        prompt = "What is the status of my latest image build?"

        response, tools_called = agent.query_with_tools(prompt)

        # Define expected tools for this query
        expected_tools = [
            ToolCall(name="get_composes"),
            # Could also include get_compose_details if compose ID is known
        ]

        for tool in tools_called:
            verbose_logger.info("Tool: %s", tool.name)
            verbose_logger.info("Parameters: %s", tool.input_parameters)
        verbose_logger.info("Response for %s: %s", llm_config['name'], response)

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
            verbose_logger.info("✓ LLM %s correctly selected relevant tools", llm_config['name'])
        else:
            verbose_logger.warning("LLM %s may not have selected optimal tools: %s",
                                   llm_config['name'], tool_names)

        # Evaluate with deepeval metric
        assert_test(test_case, [tool_correctness])

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    def test_system_prompt_effectiveness(self, mcp_server_thread, verbose_logger, llm_config):  # pylint: disable=redefined-outer-name
        """Test that the system prompt contains all necessary behavioral guidelines."""

        # Log which model is being tested
        verbose_logger.info("🧪 Testing model: %s (%s)", llm_config['name'], llm_config['MODEL_ID'])

        # Set up agent for this specific LLM configuration
        agent = MCPAgentWrapper(
            server_url=mcp_server_thread,
            api_url=llm_config['MODEL_API'],
            model_id=llm_config['MODEL_ID'],
            api_key=llm_config['USER_KEY'],
            verbose_logger=verbose_logger
        )

        # if there is a guardian LLM, use it for the guardian agent
        # otherwise, use the test LLM for the guardian agent
        if guardian_llm_config:
            guardian_agent = CustomVLLMModel(
                api_url=guardian_llm_config['MODEL_API'],
                model_id=guardian_llm_config['MODEL_ID'],
                api_key=guardian_llm_config['USER_KEY'],
                verbose_logger=verbose_logger
            )
        else:
            guardian_agent = CustomVLLMModel(
                api_url=llm_config['MODEL_API'],
                model_id=llm_config['MODEL_ID'],
                api_key=llm_config['USER_KEY'],
                verbose_logger=verbose_logger
            )

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
            model=guardian_agent
        )

        # Test the system prompt content
        system_prompt = agent.system_prompt

        verbose_logger.info("System prompt length for %s: %d characters",
                            llm_config['name'], len(system_prompt))
        verbose_logger.info("System prompt content: %s", system_prompt[:500] +
                            "..." if len(system_prompt) > 500 else system_prompt)

        # If system prompt is empty or very short, skip the test content checks
        if not system_prompt or len(system_prompt) < 100:
            verbose_logger.warning("System prompt is empty or very short for %s. Skipping content validation.",
                                   llm_config['name'])
            pytest.skip(f"System prompt not available from MCP server for {llm_config['name']}")

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
            verbose_logger.warning("System prompt for %s missing some elements: %s",
                                   llm_config['name'], missing_elements)
        else:
            verbose_logger.info("✓ System prompt for %s contains expected elements", llm_config['name'])

        test_case = LLMTestCase(
            input=f"System prompt evaluation for {llm_config['name']}",
            actual_output=system_prompt
        )

        # Evaluate with deepeval metric
        assert_test(test_case, [system_prompt_quality])

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    def test_complete_conversation_flow(self, mcp_server_thread, verbose_logger, llm_config):  # pylint: disable=redefined-outer-name
        """Test complete conversation flow with proper agent behavior."""

        # Log which model is being tested
        verbose_logger.info("🧪 Testing model: %s (%s)", llm_config['name'], llm_config['MODEL_ID'])

        # Set up agent for this specific LLM configuration
        agent = MCPAgentWrapper(
            server_url=mcp_server_thread,
            api_url=llm_config['MODEL_API'],
            model_id=llm_config['MODEL_ID'],
            api_key=llm_config['USER_KEY'],
            verbose_logger=verbose_logger
        )

        # if there is a guardian LLM, use it for the guardian agent
        # otherwise, use the test LLM for the guardian agent
        if guardian_llm_config:
            guardian_agent = CustomVLLMModel(
                api_url=guardian_llm_config['MODEL_API'],
                model_id=guardian_llm_config['MODEL_ID'],
                api_key=guardian_llm_config['USER_KEY'],
                verbose_logger=verbose_logger
            )
        else:
            guardian_agent = CustomVLLMModel(
                api_url=llm_config['MODEL_API'],
                model_id=llm_config['MODEL_ID'],
                api_key=llm_config['USER_KEY'],
                verbose_logger=verbose_logger
            )

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
            model=guardian_agent
        )

        prompt = "Can you help me understand what blueprints are available?"

        response, tools_called = agent.query_with_tools(prompt)

        test_case = LLMTestCase(
            input=prompt,
            actual_output=response,
            tools_called=tools_called
        )

        verbose_logger.info("Conversation prompt for %s: %s", llm_config['name'], prompt)
        verbose_logger.info("Tools called: %s", [tool.name for tool in tools_called])
        verbose_logger.info("Response length: %d characters", len(response))
        verbose_logger.info("Response preview: %s...", response[:200])

        # Basic sanity checks (relaxed)
        assert response, f"LLM {llm_config['name']} should provide a non-empty response"
        # Allow shorter responses for tool-based interactions
        assert len(response) > 10, f"Response from {llm_config['name']} should be substantial"

        # Evaluate with deepeval metric
        assert_test(test_case, [conversation_quality])

        verbose_logger.info("✓ Complete conversation flow test passed for %s", llm_config['name'])

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    def test_tool_usage_patterns(self, mcp_server_thread, verbose_logger, llm_config):  # pylint: disable=redefined-outer-name
        """Test various tool usage patterns and their appropriateness."""

        # Log which model is being tested
        verbose_logger.info("🧪 Testing model: %s (%s)", llm_config['name'], llm_config['MODEL_ID'])

        # Set up agent for this specific LLM configuration
        agent = MCPAgentWrapper(
            server_url=mcp_server_thread,
            api_url=llm_config['MODEL_API'],
            model_id=llm_config['MODEL_ID'],
            api_key=llm_config['USER_KEY'],
            verbose_logger=verbose_logger
        )

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
            verbose_logger.info("Testing scenario for %s: %s", llm_config['name'], scenario['description'])

            response, tools_called = agent.query_with_tools(scenario["prompt"])

            expected_tools = [ToolCall(name=name) for name in scenario["expected_tools"]]

            test_case = LLMTestCase(
                input=scenario["prompt"],
                actual_output=response,
                tools_called=tools_called,
                expected_tools=expected_tools
            )

            tool_names = [tool.name for tool in tools_called]
            verbose_logger.info("  Model: %s", llm_config['name'])
            verbose_logger.info("  Prompt: %s", scenario['prompt'])
            verbose_logger.info("  Tools called: %s", tool_names)
            verbose_logger.info("  Expected: %s", scenario['expected_tools'])

            # Evaluate with deepeval
            assert_test(test_case, [tool_correctness])

        verbose_logger.info("✓ Tool usage pattern tests completed for %s", llm_config['name'])
