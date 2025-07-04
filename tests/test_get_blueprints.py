import pytest
import json
import textwrap
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Clean import - no sys.path.insert needed with proper package structure!
from image_builder_mcp import ImageBuilderMCP, ImageBuilderClient
import image_builder_mcp.server as image_builder_mcp


class TestGetBlueprints:
    """Test suite for the get_blueprints() method."""

    @pytest.fixture
    def mock_api_response(self):
        """Mock API response based on the provided sample."""
        return {
            "data": [
                {
                    "description": "",
                    "id": "bd5bd5b7-2028-4371-9bf9-90b54565d549",
                    "last_modified_at": "2025-07-02T18:14:17Z",
                    "name": "test-rhel-9-x86_64-07022025-1708",
                    "version": 1
                },
                {
                    "description": "",
                    "id": "69ca10f4-f245-4226-ae99-83f3f02d7271",
                    "last_modified_at": "2025-07-02T21:12:12Z",
                    "name": "rhel-10-x86_64-07022025-1708",
                    "version": 1
                },
                {
                    "description": "",
                    "id": "32f14279-3db8-441b-8f91-9d84b0229787",
                    "last_modified_at": "2025-07-01T15:27:11Z",
                    "name": "test-rhel-10-x86_64-07012025-1726",
                    "version": 1
                },
                {
                    "description": "",
                    "id": "7fb574ff-4c8e-4f97-8d1f-0f646d4f4597",
                    "last_modified_at": "2025-06-30T11:12:15Z",
                    "name": "rhel-9-x86_64-06302025-1310",
                    "version": 1
                }
            ]
        }

    @pytest.fixture
    def mcp_server(self):
        """Create a mock MCP server instance."""
        server = ImageBuilderMCP(
            client_id='test-client-id',
            client_secret='test-client-secret',
            stage=False
        )
        return server

    @pytest.fixture
    def mock_client(self):
        """Create a mock ImageBuilderClient."""
        client = Mock(spec=ImageBuilderClient)
        client.client_id = 'test-client-id'
        client.domain = 'console.redhat.com'
        return client

    def test_get_blueprints_basic_functionality(self, mcp_server, mock_client, mock_api_response):
        """Test basic functionality of get_blueprints method."""
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.return_value = mock_api_response
            mcp_server.clients['test-client-id'] = mock_client

            result = mcp_server.get_blueprints(response_size=7)

            # Note that json reply below is sorted
            assert result == textwrap.dedent("""\
            [INSTRUCTION] Use the UI_URL to link to the blueprint
            [ANSWER]
            All 4 entries. There are no more.
            [{"reply_id": 1, "blueprint_uuid": "69ca10f4-f245-4226-ae99-83f3f02d7271", "UI_URL": "https://console.redhat.com/insights/image-builder/imagewizard/69ca10f4-f245-4226-ae99-83f3f02d7271", "name": "rhel-10-x86_64-07022025-1708"}, {"reply_id": 2, "blueprint_uuid": "bd5bd5b7-2028-4371-9bf9-90b54565d549", "UI_URL": "https://console.redhat.com/insights/image-builder/imagewizard/bd5bd5b7-2028-4371-9bf9-90b54565d549", "name": "test-rhel-9-x86_64-07022025-1708"}, {"reply_id": 3, "blueprint_uuid": "32f14279-3db8-441b-8f91-9d84b0229787", "UI_URL": "https://console.redhat.com/insights/image-builder/imagewizard/32f14279-3db8-441b-8f91-9d84b0229787", "name": "test-rhel-10-x86_64-07012025-1726"}, {"reply_id": 4, "blueprint_uuid": "7fb574ff-4c8e-4f97-8d1f-0f646d4f4597", "UI_URL": "https://console.redhat.com/insights/image-builder/imagewizard/7fb574ff-4c8e-4f97-8d1f-0f646d4f4597", "name": "rhel-9-x86_64-06302025-1310"}]""")

    def test_get_blueprints_with_response_size_limit(self, mcp_server, mock_client, mock_api_response):
        """Test get_blueprints with response size limitation."""
        # Setup mocks
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.return_value = mock_api_response
            mcp_server.clients['test-client-id'] = mock_client

            # Call with limited response size
            result = mcp_server.get_blueprints(response_size=2)

            # Extract JSON data from result
            json_start = result.find('[{"reply_id"')
            json_end = result.rfind('}]') + 2
            json_data = result[json_start:json_end]
            blueprints = json.loads(json_data)

            # Should return only 2 blueprints despite having 4 in response
            assert len(blueprints) == 2
            assert "Only 2 out of 4 returned" in result

    def test_get_blueprints_with_search_string(self, mcp_server, mock_client, mock_api_response):
        """Test get_blueprints with search string filtering."""
        # Setup mocks
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.return_value = mock_api_response
            mcp_server.clients['test-client-id'] = mock_client

            # Call with search string
            result = mcp_server.get_blueprints(response_size=10, search_string="rhel-10")

            # Extract JSON data from result
            json_start = result.find('[{"reply_id"')
            json_end = result.rfind('}]') + 2
            json_data = result[json_start:json_end]
            blueprints = json.loads(json_data)

            # Should return only blueprints containing "rhel-10"
            assert len(blueprints) == 2
            for blueprint in blueprints:
                assert "rhel-10" in blueprint['name'].lower()

    def test_get_blueprints_case_insensitive_search(self, mcp_server, mock_client, mock_api_response):
        """Test get_blueprints search is case insensitive."""
        # Setup mocks
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.return_value = mock_api_response
            mcp_server.clients['test-client-id'] = mock_client

            # Call with uppercase search string
            result = mcp_server.get_blueprints(response_size=10, search_string="TEST")

            # Extract JSON data from result
            json_start = result.find('[{"reply_id"')
            json_end = result.rfind('}]') + 2
            json_data = result[json_start:json_end]
            blueprints = json.loads(json_data)

            # Should find the blueprint with "TEST" in name (case insensitive search)
            assert len(blueprints) == 2
            blueprint_names = [bp['name'] for bp in blueprints]
            assert "test-rhel-9-x86_64-07022025-1708" in blueprint_names
            assert "test-rhel-10-x86_64-07012025-1726" in blueprint_names

    def test_get_blueprints_empty_response(self, mcp_server, mock_client):
        """Test get_blueprints with empty API response."""
        # Setup mocks
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.return_value = {"data": []}
            mcp_server.clients['test-client-id'] = mock_client

            # Call the method
            result = mcp_server.get_blueprints(response_size=7)

            # Should return empty list
            assert "[]" in result
            assert "All 0 entries" in result

    def test_get_blueprints_api_error(self, mcp_server, mock_client):
        """Test get_blueprints when API returns error."""
        # Setup mocks
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.side_effect = Exception("API Error")
            mcp_server.clients['test-client-id'] = mock_client

            # Call the method
            result = mcp_server.get_blueprints(response_size=7)

            # Should return error message
            assert result.startswith("Error: API Error")

    def test_get_blueprints_no_auth(self, mcp_server):
        """Test get_blueprints without authentication."""
        # Setup mocks - no credentials
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {}

            # Call the method
            result = mcp_server.get_blueprints(response_size=7)

            # Should return authentication error
            # The actual implementation makes API calls and gets 401 errors when no auth is provided
            assert result.startswith("Error:")

    def test_get_blueprints_internal_state_management(self, mcp_server, mock_client, mock_api_response):
        """Test that get_blueprints properly manages internal state."""
        # Setup mocks
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.return_value = mock_api_response
            mcp_server.clients['test-client-id'] = mock_client

            # Call the method
            mcp_server.get_blueprints(response_size=2)

            # Verify internal state was updated
            assert 'test-client-id' in mcp_server.blueprints
            assert len(mcp_server.blueprints['test-client-id']) == 4
            assert 'test-client-id' in mcp_server.blueprint_current_index
            assert mcp_server.blueprint_current_index['test-client-id'] == 3  # min(4, 2+1)

            # Verify blueprint data structure
            for i, blueprint in enumerate(mcp_server.blueprints['test-client-id']):
                assert blueprint['reply_id'] == i + 1
                assert 'blueprint_uuid' in blueprint
                assert 'UI_URL' in blueprint
                assert 'name' in blueprint

    def test_get_blueprints_null_search_string_handling(self, mcp_server, mock_client, mock_api_response):
        """Test handling of 'null' string as search parameter."""
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {
                'x-client-id': 'test-client-id',
                'x-client-secret': 'test-client-secret'
            }
            mock_client.make_request.return_value = mock_api_response
            mcp_server.clients['test-client-id'] = mock_client

            # Call with "null" string (workaround for LLama 3.3 70B Instruct)
            result = mcp_server.get_blueprints(response_size=10, search_string="null")

            # Should treat "null" string as None and return all blueprints
            json_start = result.find('[{"reply_id"')
            json_end = result.rfind('}]') + 2
            json_data = result[json_start:json_end]
            blueprints = json.loads(json_data)
            
            assert len(blueprints) == 4  # All blueprints should be returned

    def test_get_blueprints_no_auth_error_message(self):
        """Test that get_blueprints returns the no_auth_error() message when authentication is missing."""
        # Create MCP server without default credentials
        mcp_server = ImageBuilderMCP(
            client_id=None,
            client_secret=None,
            stage=False
        )
        
        # Test default transport mode
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {}  # No auth headers
            
            result = mcp_server.get_blueprints(response_size=7)
            
            # Check for relevant parts of the no_auth_error message for default transport
            assert "Tell the user" in result
            assert "IMAGE_BUILDER_CLIENT_ID" in result
            assert "IMAGE_BUILDER_CLIENT_SECRET" in result
            assert "mcp.json config" in result
            assert "Error: Client ID and secret are required to access the Image Builder API" in result

    def test_get_blueprints_no_auth_error_message_sse_transport(self):
        """Test that get_blueprints returns the no_auth_error() message for SSE transport when authentication is missing."""
        # Create MCP server with SSE transport
        mcp_server = ImageBuilderMCP(
            client_id=None,
            client_secret=None,
            stage=False,
            transport="sse"
        )
        
        with patch.object(image_builder_mcp, 'get_http_headers') as mock_headers:
            mock_headers.return_value = {}  # No auth headers
            
            result = mcp_server.get_blueprints(response_size=7)
            
            # Check for relevant parts of the no_auth_error message for SSE transport
            assert "Tell the user" in result
            assert "header variables" in result
            assert "x-client-ID" in result
            assert "x-client-secret" in result
            assert "Error: Client ID and secret are required to access the Image Builder API" in result 