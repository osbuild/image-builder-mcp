"""Integration tests for LLM functionality with MCP server."""

import json

import pytest

from .test_utils import should_skip_llm_tests, LLMTestUtils


class ServerStartupError(Exception):
    """Exception raised when MCP server fails to start."""


class ServerConnectionError(Exception):
    """Exception raised when unable to connect to MCP server."""


@pytest.mark.skipif(should_skip_llm_tests(), reason="LLM environment variables not set")
class TestLLMIntegration:
    """Test LLM integration with MCP server using HTTP streaming protocol."""

    def __init__(self):
        self.utils = None

    @pytest.fixture(autouse=True)
    def setup_utils(self):
        """Set up utilities for tests."""
        self.utils = LLMTestUtils()

    # pylint: disable=redefined-outer-name, too-many-locals

    def test_rhel_image_creation_question(self, mcp_server_thread, verbose_logger):
        """Test LLM tool selection for RHEL image creation."""
        server_url = mcp_server_thread

        # Get tools and instructions from MCP server
        tools, system_prompt = self.utils.get_mcp_tools_and_instructions(server_url)

        verbose_logger.debug(f"Extracted system prompt: {system_prompt[:200]}...")

        # Ask about creating RHEL image
        prompt = "Can you create a RHEL 9 image for me?"

        response = self.utils.call_llm(
            prompt=prompt,
            tools=tools,
            logger=verbose_logger,
            system_prompt=system_prompt
        )

        # Check if LLM wants to use tools
        choice = response["choices"][0]
        message = choice["message"]

        # Look for tool calls in response
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            tool_names = [call["function"]["name"] for call in tool_calls]

            assert "create_blueprint" not in tool_names, (
                "LLM should not select create_blueprint tool, but rather ask for more information"
            )

            if "get_openapi" in tool_names:
                # Use the complete conversation flow instead of _process_tool_calls directly
                final_response = self.utils.complete_conversation_with_tools(
                    prompt, server_url, verbose_logger, tools=tools, system_prompt=system_prompt)
                verbose_logger.info(f"Final LLM response after tool calls: {json.dumps(final_response, indent=2)}")

                # Check the final response content
                final_choice = final_response["choices"][0]
                final_message = final_choice["message"]
                final_content = final_message.get("content", "")

                # The LLM should now provide a more informative response using the tool results
                verbose_logger.info(f"Final response content: {final_content}")

                # Update response for further assertions
                response = final_response

        else:
            print("LLM responded without tool calls")
            # Check if LLM asks for more information in its response text
            response_text = message.get("content", "").lower()
            asking_questions = any(word in response_text for word in ["what", "which", "do you", "would you", "?"])

            assert asking_questions, f"LLM should be asking for more information, but it didn't: {response_text}"

    # pylint: disable=redefined-outer-name
    def test_image_build_status_question(self, mcp_server_thread, verbose_logger):
        """Test LLM tool selection for image build status."""
        server_url = mcp_server_thread

        # Get tools and instructions from MCP server
        tools, system_prompt = self.utils.get_mcp_tools_and_instructions(server_url)

        # Ask about image build status
        prompt = "What is the status of my latest image build?"

        response = self.utils.call_llm(
            prompt=prompt,
            tools=tools,
            logger=verbose_logger,
            system_prompt=system_prompt
        )

        # Check if LLM wants to use tools
        choice = response["choices"][0]
        message = choice["message"]

        # Look for tool calls in response
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            tool_names = [call["function"]["name"] for call in tool_calls]

            # Check if LLM selected relevant tools
            expected_tools = ["get_composes", "get_compose_details"]
            found_relevant = any(tool in tool_names for tool in expected_tools)

            assert found_relevant, f"LLM didn't select relevant tools. Selected: {tool_names}"
            print(f"LLM correctly selected tools: {tool_names}")
        else:
            print("LLM responded without tool calls - this might be expected behavior")

    # pylint: disable=redefined-outer-name
    def test_behavioral_rules_effectiveness(self, mcp_server_thread, verbose_logger):
        """Test that the behavioral rules in system prompt are effective."""
        server_url = mcp_server_thread

        # Get tools and instructions from MCP server
        tools, system_prompt = self.utils.get_mcp_tools_and_instructions(server_url)

        # Ask about creating RHEL image
        prompt = "Can you create a RHEL 9 image for me?"

        response = self.utils.call_llm(
            prompt=prompt,
            tools=tools,
            logger=verbose_logger,
            system_prompt=system_prompt
        )

        # Check LLM response
        choice = response["choices"][0]
        message = choice["message"]

        # The key test: LLM should NOT immediately call create_blueprint
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            tool_names = [call["function"]["name"] for call in tool_calls]

            # This is the critical assertion - system prompt should prevent immediate create_blueprint calls
            assert "create_blueprint" not in tool_names, (
                f"❌ BEHAVIORAL RULE VIOLATION: LLM called create_blueprint immediately! "
                f"Tool calls: {tool_names}. System prompt not working correctly."
            )

            # LLM should select get_openapi to understand the structure first
            if "get_openapi" in tool_names:
                print("✓ LLM correctly selected get_openapi to understand structure first")
            else:
                print(f"LLM selected tools: {tool_names}")
        else:
            # If no tool calls, LLM should be asking questions
            response_text = message.get("content", "")
            print(f"✓ LLM responded with text instead of immediate tool calls: {response_text[:100]}...")

        print("✓ Behavioral rules are working - LLM did not immediately call create_blueprint")

    # pylint: disable=redefined-outer-name
    def test_complete_conversation_flow(self, mcp_server_thread, verbose_logger):
        """Test complete conversation flow demonstrating proper agent behavior."""
        server_url = mcp_server_thread

        # Get tools and instructions from MCP server
        tools, system_prompt = self.utils.get_mcp_tools_and_instructions(server_url)

        # Test the complete conversation flow
        user_prompt = "Can you help me understand what blueprints are available?"

        final_response = self.utils.complete_conversation_with_tools(
            user_prompt, server_url, verbose_logger, tools=tools, system_prompt=system_prompt
        )

        # Verify we got a proper response
        assert "choices" in final_response, "Response should have choices"
        assert len(final_response["choices"]) > 0, "Response should have at least one choice"

        final_choice = final_response["choices"][0]
        final_message = final_choice["message"]
        final_content = final_message.get("content", "")

        verbose_logger.info(f"Final conversation response: {final_content}")

        # The response should contain meaningful information about blueprints
        assert final_content, f"LLM should provide a non-empty response {final_response}"
        assert len(final_content) > 50, "Response should be substantial"

        print("✓ Complete conversation flow test passed")
        print(f"Final response length: {len(final_content)} characters")
