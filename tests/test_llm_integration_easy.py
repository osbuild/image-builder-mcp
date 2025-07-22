"""Integration tests for LLM functionality with MCP server using deepeval.
This includes easy questions to the LLM, that should work out of the box.
"""

from typing import Any, Dict, List
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.metrics import GEval, ToolCorrectnessMetric

from .utils import (
    should_skip_llm_matrix_tests,
    load_llm_configurations,
    pretty_print_conversation_history
)


# Load LLM configurations for parametrization
llm_configurations, _ = load_llm_configurations()

# Test scenarios for tool usage patterns
# not sure why mypy needs Any here
TOOL_USAGE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "prompt": "List all my recent builds",
        "expected_tools": ["get_composes"],
        "description": "Should use get_composes for build listings"
    },
    {
        "prompt": "What blueprints do I have?",
        "expected_tools": ["get_blueprints"],
        "description": "Should use get_blueprints for blueprint listings"
    },
    {
        "prompt": "Please show my blueprints",
        "expected_tools": ["get_blueprints"],
        "description": "Should use get_blueprints for blueprint listings"
    }
]


@pytest.mark.skipif(should_skip_llm_matrix_tests(), reason="No valid LLM configurations found")
class TestLLMIntegrationEasy:
    """Test LLM integration with MCP server using deepeval with multiple LLM configurations."""

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    # pylint: disable=redefined-outer-name
    def test_rhel_initial_question(self, test_agent, guardian_agent, llm_config, verbose_logger):
        """Test that LLM follows behavioral rules and doesn't immediately call create_blueprint."""

        prompt = "Can you create a RHEL 9 image for me?"

        messages = [{
            "user": prompt
        }]

        # Use lightweight intention-only check instead of actually executing tools
        response, tools_intended, _ = test_agent.query_with_messages(messages)

        # Check that create_blueprint is not called immediately
        tool_names = [tool.name for tool in tools_intended]
        assert "create_blueprint" not in tool_names, (
            f"❌ BEHAVIORAL RULE VIOLATION for {llm_config['name']} ({llm_config['MODEL_ID']}): "
            f"LLM called create_blueprint immediately! Tool calls: {tool_names}. "
            f"System prompt not working correctly.\nThe prompt was: {prompt}\n"
            f"The response was: {response}\n"
        )

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

        verbose_logger.info("Test passed for %s", prompt)
        verbose_logger.info("Response: %s", response)

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    # pylint: disable=redefined-outer-name,too-many-locals
    def test_image_build_status_tool_selection(self, test_agent, verbose_logger, llm_config, guardian_agent):
        """Test that LLM selects appropriate tools for image build status queries."""

        # Define tool correctness metric - ToolCorrectnessMetric doesn't support model parameter
        tool_correctness = ToolCorrectnessMetric(
            threshold=0.7,
            include_reason=True
        )

        prompt = "What is the status of my latest image build?"

        messages = [{
            "user": prompt
        }]
        response, tools_intended, conversation_history = test_agent.query_with_messages(messages)

        verbose_logger.info("Prompt: %s", prompt)
        verbose_logger.info("Response: %s", response)
        verbose_logger.info("Tools called: %s", [tool.name for tool in tools_intended])
        verbose_logger.info("Full conversation history:\n%s",
                            pretty_print_conversation_history(conversation_history, llm_config['name']))

        # first we check if there is a question in the response for the name or UUID of the compose
        contains_question = GEval(
            name="Contains Question",
            criteria=(
                "The response should contain a question for the name or UUID of the compose"
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=guardian_agent
        )

        question_test_case = LLMTestCase(
            input=prompt,
            actual_output=response,
        )

        answered_with_question = None
        # if this fails that's ok, we can continue
        try:
            assert_test(question_test_case, [contains_question])
            verbose_logger.info("✓ LLM %s correctly answered with a question", llm_config['name'])
        except AssertionError as e:
            answered_with_question = e
            verbose_logger.info("Question test case failed, continuing...")

        # Define expected tools for this query
        expected_tools = [
            ToolCall(name="get_composes"),
            # Could also include get_compose_details if compose ID is known
        ]

        test_case = LLMTestCase(input=prompt, actual_output=response,
                                tools_called=tools_intended, expected_tools=expected_tools)

        # Check if relevant tools were selected
        tool_names = [tool.name for tool in tools_intended]
        expected_tool_names = ["get_composes", "get_compose_details"]
        found_relevant = any(tool in tool_names for tool in expected_tool_names)

        if found_relevant:
            verbose_logger.info("✓ LLM %s correctly selected relevant tools", llm_config['name'])
        else:
            verbose_logger.warning("LLM %s may not have selected optimal tools: %s",
                                   llm_config['name'], tool_names)

        answered_with_tools = None
        try:
            assert_test(test_case, [tool_correctness])
            verbose_logger.info("✓ LLM %s correctly used the tools", llm_config['name'])
        except AssertionError as e:
            answered_with_tools = e
            verbose_logger.info("Tool correctness test case failed, continuing...")

        assert answered_with_question is None or answered_with_tools is None, "One of the tests have to succeed"

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    @pytest.mark.parametrize("scenario", TOOL_USAGE_SCENARIOS,
                             ids=[scenario['prompt'] for scenario in TOOL_USAGE_SCENARIOS])
    def test_tool_usage_patterns(self, test_agent, verbose_logger, llm_config, scenario):  # pylint: disable=redefined-outer-name
        """Test various tool usage patterns and their appropriateness."""

        messages = [{
            "user": scenario["prompt"]
        }]
        response, tools_intended, _ = test_agent.query_with_messages(messages)

        expected_tools = [ToolCall(name=name) for name in scenario["expected_tools"]]

        test_case = LLMTestCase(
            input=scenario["prompt"],
            actual_output=response,
            tools_called=tools_intended,
            expected_tools=expected_tools
        )

        tool_names = [tool.name for tool in tools_intended]
        verbose_logger.info("  Model: %s", llm_config['name'])
        verbose_logger.info("  Prompt: %s", scenario['prompt'])
        verbose_logger.info("  Expected: %s", scenario['expected_tools'])
        verbose_logger.info("  Tools called: %s", tool_names)
        verbose_logger.info("  Response: %s", response)

        # Create tool correctness metric - doesn't support model parameter
        tool_correctness = ToolCorrectnessMetric(threshold=0.6)
        # Evaluate with deepeval
        assert_test(test_case, [tool_correctness])

        verbose_logger.info("✓ Tool usage pattern test passed for %s with prompt: %s",
                            llm_config['name'], scenario['prompt'])

    @pytest.mark.parametrize("llm_config", llm_configurations,
                             ids=[config['name'] for config in llm_configurations])
    def test_llm_paging(self, test_agent, verbose_logger, llm_config):  # pylint: disable=redefined-outer-name,too-many-locals
        """Test that the LLM can page through results."""

        prompt = "List my latest 2 blueprints"

        messages = [{
            "user": prompt
        }]
        response, tools_called, conversation_history = test_agent.execute_tools_with_messages(messages)

        verbose_logger.info("Model: %s", llm_config['name'])
        verbose_logger.info("Prompt: %s", prompt)
        verbose_logger.info("Tools called: %s", [tool.name for tool in tools_called])
        verbose_logger.info("Full conversation history:\n%s",
                            pretty_print_conversation_history(conversation_history, llm_config['name']))

        expected_tools = [ToolCall(name="get_blueprints")]

        test_case_initial = LLMTestCase(
            input=prompt,
            actual_output=response,
            tools_called=tools_called,
            expected_tools=expected_tools
        )
        tool_correctness = ToolCorrectnessMetric(threshold=0.6)

        assert_test(test_case_initial, [tool_correctness])

        # Now ask for more with conversation context
        follow_up_prompt = "Can you show me the next 3 blueprints?"
        messages = [{
            "user": follow_up_prompt
        }]
        response, tools_intended, updated_history = test_agent.query_with_messages(messages, conversation_history)

        verbose_logger.info("Follow-up Prompt: %s", follow_up_prompt)
        verbose_logger.info("Full conversation history:\n%s",
                            pretty_print_conversation_history(updated_history, llm_config['name']))

        expected_tools = [ToolCall(name="get_blueprints", arguments={"limit": 3, "offset": 2})]

        test_case_subsequent = LLMTestCase(
            input=follow_up_prompt,
            actual_output=response,
            tools_called=tools_intended,
            expected_tools=expected_tools
        )
        tool_correctness = ToolCorrectnessMetric(threshold=0.6)

        assert_test(test_case_subsequent, [tool_correctness])
