"""
Pytest configuration and shared fixtures for image-builder-mcp tests.
"""

import pytest
from .test_utils import start_mcp_server_process, cleanup_server_process


@pytest.fixture
def default_response_size():
    """Default response size for pagination tests."""
    return 7


@pytest.fixture
def test_client_credentials():
    """Test client credentials."""
    return {
        'client_id': 'test-client-id',
        'client_secret': 'test-client-secret'
    }


@pytest.fixture
def mock_http_headers(client_creds):
    """Mock HTTP headers with test credentials."""
    return {
        'image-builder-client-id': client_creds['client_id'],
        'image-builder-client-secret': client_creds['client_secret']
    }


@pytest.fixture(scope="session")
def mcp_server_thread():  # pylint: disable=too-many-locals
    """Start MCP server in a separate thread using HTTP streaming."""
    server_url, server_process = start_mcp_server_process()

    try:
        yield server_url
    finally:
        cleanup_server_process(server_process)
