"""Test suite for authentication-related functionality."""

from unittest.mock import patch
import pytest

# Clean import - no sys.path.insert needed with proper package structure!
from image_builder_mcp import ImageBuilderMCP
import image_builder_mcp.server as image_builder_mcp


class TestAuthentication:
    """Test suite for authentication-related functionality."""

    # List of functions to test for authentication (excluding get_openapi)
    AUTH_FUNCTIONS = [
        ('create_blueprint', {'data': {'name': 'test', 'description': 'test'}}),
        ('get_blueprints', {'response_size': 7}),
        ('get_more_blueprints', {'response_size': 7}),
        ('get_blueprint_details', {'blueprint_identifier': 'test-uuid'}),
        ('get_composes', {'response_size': 7}),
        ('get_more_composes', {'response_size': 7}),
        ('get_compose_details', {'compose_identifier': 'test-uuid'}),
        ('blueprint_compose', {'blueprint_uuid': 'test-uuid'}),
    ]

    @pytest.mark.parametrize("function_name,kwargs", AUTH_FUNCTIONS)
    def test_function_no_auth(self, function_name, kwargs):
        """Test that functions without authentication return error."""
        mcp_server = ImageBuilderMCP(
            client_id='test-client-id',
            client_secret='test-client-secret',
            stage=False
        )

        # Setup mocks - no credentials
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {}

            # Call the method
            method = getattr(mcp_server, function_name)
            result = method(**kwargs)

            # Should return authentication error
            # The actual implementation makes API calls and gets 401 errors when no auth is provided
            assert result.startswith("Error:")

    @pytest.mark.parametrize("function_name,kwargs", AUTH_FUNCTIONS)
    def test_function_no_auth_error_message(self, function_name, kwargs):
        """Test that functions return the no_auth_error() message when authentication is missing."""
        # Create MCP server without default credentials
        mcp_server = ImageBuilderMCP(
            client_id=None,
            client_secret=None,
            stage=False
        )

        # Test default transport mode
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {}  # No auth headers

            method = getattr(mcp_server, function_name)
            result = method(**kwargs)

            # Check for relevant parts of the no_auth_error message for default transport
            assert "Tell the user" in result
            assert "IMAGE_BUILDER_CLIENT_ID" in result
            assert "IMAGE_BUILDER_CLIENT_SECRET" in result
            assert "mcp.json config" in result
            assert "Error: Client ID is required to access the Image Builder API" in result

    @pytest.mark.parametrize("function_name,kwargs", AUTH_FUNCTIONS)
    def test_function_no_auth_error_message_sse_transport(self, function_name, kwargs):
        """Test that functions return the no_auth_error() message for SSE transport.

        Tests the case when authentication is missing.
        """
        # Create MCP server with SSE transport
        mcp_server = ImageBuilderMCP(
            client_id=None,
            client_secret=None,
            stage=False,
            transport="sse"
        )

        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {}  # No auth headers

            method = getattr(mcp_server, function_name)
            result = method(**kwargs)

            # Check for relevant parts of the no_auth_error message for SSE transport
            assert "Tell the user" in result
            assert "header variables" in result
            assert "image-builder-client-id" in result
            assert "image-builder-client-secret" in result
            assert "Error: Client ID is required to access the Image Builder API" in result

    @pytest.mark.parametrize("function_name,kwargs", AUTH_FUNCTIONS)
    def test_function_no_auth_error_message_http_transport(self, function_name, kwargs):
        """Test that functions return the no_auth_error() message for HTTP transport.

        Tests the case when authentication is missing.
        """
        # Create MCP server with HTTP transport
        mcp_server = ImageBuilderMCP(
            client_id=None,
            client_secret=None,
            stage=False,
            transport="http"
        )

        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {}  # No auth headers

            method = getattr(mcp_server, function_name)
            result = method(**kwargs)

            # Check for relevant parts of the no_auth_error message for HTTP transport
            assert "Tell the user" in result
            assert "header variables" in result
            assert "image-builder-client-id" in result
            assert "image-builder-client-secret" in result
            assert "Error: Client ID is required to access the Image Builder API" in result
