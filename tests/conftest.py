"""
Pytest configuration and shared fixtures for image-builder-mcp tests.
"""

import pytest
from unittest.mock import Mock

# Clean import - no sys.path.insert needed with proper package structure!
from image_builder_mcp import ImageBuilderMCP, ImageBuilderClient


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
def mock_http_headers(test_client_credentials):
    """Mock HTTP headers with test credentials."""
    return {
        'x-client-id': test_client_credentials['client_id'],
        'x-client-secret': test_client_credentials['client_secret']
    } 