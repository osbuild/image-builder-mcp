"""Image Builder MCP server for creating and managing Linux images."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools.tool import Tool
from mcp.types import ToolAnnotations
import uvicorn
import jwt

from .oauth import Middleware

from .client import ImageBuilderClient


class ImageBuilderMCP(FastMCP):  # pylint: disable=too-many-instance-attributes
    """MCP server for Red Hat Image Builder integration.

    This server provides tools for creating, managing, and building
    custom Linux images using the Red Hat Image Builder service.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            client_id: Optional[str],
            client_secret: Optional[str],
            default_response_size: int = 10,
            stage: Optional[bool] = False,
            proxy_url: Optional[str] = None,
            transport: Optional[str] = None,
            oauth_enabled: bool = False):
        self.stage = stage
        self.proxy_url = proxy_url
        self.transport = transport
        self.default_response_size = default_response_size
        # TBD: make this configurable
        # probably we want to destiguish a hosted MCP server from
        # a local one (deployed by a customer)
        self.image_builder_mcp_client_id = "mcp"
        self.oauth_enabled = oauth_enabled

        self.logger = logging.getLogger("ImageBuilderMCP")

        # could be used once we have e.g. "/distributions" available without authentication
        self.client_noauth = ImageBuilderClient(
            client_id=None,
            client_secret=None,
            stage=self.stage,
            proxy_url=self.proxy_url,
            image_builder_mcp_client_id=self.image_builder_mcp_client_id,
            oauth_enabled=self.oauth_enabled
        )

        # use dynamic attributes to get the distributions, architectures and image types
        # once the API is changed to un-authenticated access
        # self.distributions = self.client_noauth.make_request("distributions")
        self.distributions = [
            {'description': 'CentOS Stream 9', 'name': 'centos-9'},
            {'description': 'Fedora Linux 37', 'name': 'fedora-37'},
            {'description': 'Fedora Linux 38', 'name': 'fedora-38'},
            {'description': 'Fedora Linux 39', 'name': 'fedora-39'},
            {'description': 'Fedora Linux 40', 'name': 'fedora-40'},
            {'description': 'Fedora Linux 41', 'name': 'fedora-41'},
            {'description': 'Fedora Linux 42', 'name': 'fedora-42'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 10 Beta', 'name': 'rhel-10-beta'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 10', 'name': 'rhel-10.0'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 10', 'name': 'rhel-10'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-8.10'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-8'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-84'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-85'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-86'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-87'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-88'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 8', 'name': 'rhel-89'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9 beta', 'name': 'rhel-9-beta'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-9.6'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-9'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-90'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-91'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-92'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-93'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-94'},
            {'description': 'Red Hat Enterprise Linux (RHEL) 9', 'name': 'rhel-95'}
        ]

        try:
            # TBD: change openapi spec to have a proper schema-enum
            # for image types and architectures
            self.logger.info("Getting openapi")
            openapi = json.loads(self.get_openapi(1))

            self.image_types = list(openapi["components"]["schemas"]["ImageTypes"]["enum"])
            self.image_types.sort()

            self.architectures = list(openapi["components"]["schemas"]["ImageRequest"]
                                      ["properties"]["architecture"]["enum"])
            self.architectures.sort()

            self.logger.info("Supported image types: %s", self.image_types)
            self.logger.info("Supported architectures: %s", self.architectures)
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise ValueError("Error getting openapi for image types and architectures") from e

        general_intro = f"""You are a comprehensive Linux Image Builder assistant that creates custom
        Linux disk images, ISOs, and virtual machine images.

        You can build images for multiple Linux distributions including:
        - Red Hat Enterprise Linux (RHEL)
        - CentOS Stream
        - Fedora Linux

        You create various image formats suitable for:
        - Cloud deployments (AWS, Azure, GCP)
        - Virtual machines (VMware, guest images)
        - Edge computing devices
        - Container registries (OCI)
        - Bare metal installations (ISO installers)
        - WSL (Windows Subsystem for Linux)

        This service uses Red Hat's console.redhat.com image-builder osbuild.org infrastructure but serves
        general Linux image building needs across the entire ecosystem.

        🚨 CRITICAL BEHAVIORAL RULES:

        1. **NEVER CALL create_blueprint() IMMEDIATELY** when a user asks to create an image
        2. **ALWAYS GATHER COMPLETE INFORMATION FIRST** through a conversational approach
        3. **ASK SPECIFIC QUESTIONS** to collect all required details before making any API calls
        4. **BE HELPFUL AND CONSULTATIVE** - guide users through the image creation process

        WHEN A USER ASKS TO CREATE AN IMAGE OR ISO:
        - Start by asking about their specific needs and use case
        - Ask for blueprint name, distribution, architecture, image type, etc.
        - For RHEL images: Always ask about registration preferences
        - Ask about custom user accounts and any special configurations
        - Only call create_blueprint() after you have ALL required information

        AVAILABLE DISTRIBUTIONS: {', '.join([d['name'] for d in self.distributions])}
        AVAILABLE ARCHITECTURES: {', '.join(self.architectures)}
        AVAILABLE IMAGE TYPES: {', '.join(self.image_types)}

        Your goal is to be a knowledgeable consultant who helps users create the perfect
        custom Linux image, ISO, or virtual machine image for their specific deployment needs.

        IMPORTANT: When you decide a tool is needed, emit only the JSON tool_call block (no extra narrative).
        Use the tool_call field exactly as specified in the API schema

        CRITICAL TOOL CALLING INSTRUCTIONS:
        When you need to call a tool, you MUST use the tool_calls format, NOT plain text.

        """

        super().__init__(
            name="Image Builder MCP Server",
            instructions=general_intro
        )

        # cache the client for all users
        # TBD: purge cache after some time
        self.clients = {}
        self.client_id = None
        self.client_secret = None

        if client_id and client_secret:
            self.clients[client_id] = ImageBuilderClient(
                client_id,
                client_secret,
                stage=self.stage,
                proxy_url=self.proxy_url,
                image_builder_mcp_client_id=self.image_builder_mcp_client_id,
                oauth_enabled=self.oauth_enabled
            )
            self.client_id = client_id
            self.client_secret = client_secret

        self.register_tools()

    def register_tools(self):
        """Register all available tools with the MCP server."""
        # prepend generic keywords for use of many other tools
        # and register with "self.tool()"
        tool_functions = [self.get_openapi,
                          self.create_blueprint,
                          self.get_blueprints,
                          self.get_blueprint_details,
                          self.get_composes,
                          self.get_compose_details,
                          self.blueprint_compose
                          # self.compose
                          ]

        for f in tool_functions:
            tool = Tool.from_function(f)
            tool.annotations = ToolAnnotations(
                readOnlyHint=True,
                openWorldHint=True
            )
            description_str = f.__doc__.format(
                distributions=", ".join([d['name']
                                        for d in self.distributions]),
                architectures=", ".join(self.architectures),
                image_types=", ".join(self.image_types)
            )
            tool.description = description_str
            tool.title = description_str.split("\n", 1)[0]
            self.add_tool(tool)

    def get_client_id(self, headers: Dict[str, str]) -> str:
        """Get the client ID preferably from the headers."""
        client_id = self.client_id or ""
        if self.oauth_enabled:
            caller_headers_auth = headers.get("authorization")
            if caller_headers_auth and caller_headers_auth.startswith("Bearer "):
                # decode bearer token to get sid and use as client_id
                token = caller_headers_auth.split("Bearer ", 1)[-1]
                client_id = jwt.decode(
                    token, options={"verify_signature": False}).get("sid")
                self.logger.debug(
                    "Using sid from Bearer token as client_id: %s", client_id)
        else:
            client_id = headers.get("image-builder-client-id") or self.client_id or ""
            self.logger.debug("get_client_id request headers: %s", headers)

        # explicit check for mypy
        if not client_id:
            raise ValueError("Client ID is required to access the Image Builder API")
        return client_id

    def get_client_secret(self, headers: Dict[str, str]) -> str:
        """Get the client secret preferably from the headers."""
        client_secret = headers.get("image-builder-client-secret") or self.client_secret
        self.logger.debug("get_client_secret request headers: %s", headers)

        if not client_secret:
            raise ValueError("Client secret is required to access the Image Builder API")
        return client_secret

    def get_client(self, headers: Dict[str, str]) -> ImageBuilderClient:
        """Get the ImageBuilderClient instance for the current user."""
        client_id = self.get_client_id(headers)
        client = self.clients.get(client_id)
        if not client:
            client_secret = None
            if not self.oauth_enabled:
                client_secret = self.get_client_secret(headers)
            client = ImageBuilderClient(
                client_id,
                client_secret,
                stage=self.stage,
                proxy_url=self.proxy_url,
                image_builder_mcp_client_id=self.image_builder_mcp_client_id,
                oauth_enabled=self.oauth_enabled)
            self.clients[client_id] = client
        return client

    def no_auth_error(self, e: Exception) -> str:
        """Generate authentication error message based on transport type."""
        base_message = (
            "[INSTRUCTION] Tell the user that the MCP server setup is not valid! "
            "The user should go to [https://console.redhat.com](https://console.redhat.com) to "
            "'YOUR USER' ➡ My User Access ➡ Service Accounts create a service account and then set the "
        )

        if self.transport in ["sse", "http"]:
            return (
                f"{base_message}header variables `image-builder-client-id` and "
                "`image-builder-client-secret` in your request.\n"
                "Here is the direct link for the user's convenience: "
                "[https://console.redhat.com/iam/service-accounts](https://console.redhat.com/iam/service-accounts) "
                "Come up with a detailed description of this for the user. "
                "Only describe this, don't expose details about the tool function itself. "
                f"Don't proceed with the request before this is fixed. Error: {str(e)}."
            )

        return (
            f"{base_message}`IMAGE_BUILDER_CLIENT_ID` and `IMAGE_BUILDER_CLIENT_SECRET` "
            "in your mcp.json config.\n"
            "Here is the direct link for the user's convenience: "
            "[https://console.redhat.com/iam/service-accounts](https://console.redhat.com/iam/service-accounts) "
            "Come up with a detailed description of this for the user. "
            "Only describe this, don't expose details about the tool function itself. "
            f"Don't proceed with the request before this is fixed. Error: {str(e)}."
        )

    def compose(self,  # pylint: disable=too-many-arguments
                distribution: str,
                architecture: str = "x86_64",
                image_type: str = "guest-image",
                image_name: Optional[str] = None,
                image_description: Optional[str] = None) -> str:
        """Create a new, up to date, operating system image.
        Ensure that the data follows the ComposeRequest structure described in the OpenAPI spec.
        Gather all required details from the user before calling this function.

        Args:
            distribution: the distribution to use (available: {distributions})
            architecture: the architecture to use (available: {architectures})
            image_type: the type of image to create (available: {image_types})
            image_name: optional name for the image (ask user if they want to set this)
            image_description: optional description for the image (ask user if they want to set this)
        """
        try:
            client = self.get_client(get_http_headers())
        except ValueError as e:
            return self.no_auth_error(e)

        data = {
            "distribution": distribution,
            "client_id": self.image_builder_mcp_client_id,
            "image_requests": [
                {
                    "architecture": architecture,
                    "image_type": image_type,
                    "upload_request": {
                        "type": "aws.s3",
                        "options": {}
                    }
                }
            ]
            # "customizations": {…}
        }
        if image_name:
            data["image_name"] = image_name
        else:
            # Generate a default image name based on distribution and architecture
            name = f"{distribution}-{architecture}-{image_type}-"
            name += f"{datetime.now().strftime('%Y%m%d%H%M%S')}-mcp"
            data["image_name"] = name
        if image_description:
            data["image_description"] = image_description
        else:
            # Generate a default image description
            description = "Image created via image-builder-mcp on"
            description += f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            data["image_description"] = description
        try:
            # TBD: programmatically check against openapi
            response = client.make_request("compose", method="POST", data=data)
            return f"Compose created successfully: {json.dumps(response)}"
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {str(e)} for compose {json.dumps(data)}"

    def blueprint_compose(self, blueprint_uuid: str) -> str:
        """Compose an image from a blueprint UUID created with create_blueprint, get_blueprints.
        If the UUID is not clear, ask the user whether to create a new blueprint with create_blueprint
        or use an existing blueprint from get_blueprints.

        Args:
            blueprint_uuid: the UUID of the blueprint to compose

        Returns:
            The response from the image-builder API

        Raises:
        """
        try:
            client = self.get_client(get_http_headers())
        except ValueError as e:
            return self.no_auth_error(e)

        try:
            response = client.make_request(
                f"blueprints/{blueprint_uuid}/compose", method="POST")
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {str(e)} in blueprint_compose {blueprint_uuid}"

        response_str = "[INSTRUCTION] Use the tool get_compose_details to get the details of the compose\n"
        response_str += "like the current build status\n"
        response_str += "[ANSWER] Compose created successfully:"
        build_ids_str = []

        if isinstance(response, dict):
            return f"Error: the response of blueprint_compose is a dict. This is not expected. " \
                f"Response: {json.dumps(response)}"

        for build in response:
            if isinstance(build, dict) and 'id' in build:
                build_ids_str.append(f"UUID: {build['id']}")
            else:
                build_ids_str.append(f"Invalid build object: {build}")

        response_str += f"\n{json.dumps(build_ids_str)}"
        response_str += "\nWe could double check the details or start the build/compose"
        return response_str

    def get_openapi(self, response_size: int) -> str:
        """Get OpenAPI spec. Use this to get details e.g for a new blueprint

        Args:
            response_size: number of items returned (use 7 as default)

        Returns:
            List of blueprints

        Raises:
            Exception: If the image-builder connection fails.
        """
        # response_size is just a dummy parameter for langflow
        _ = response_size  # Unused parameter, required by interface
        try:
            response = self.client_noauth.make_request("openapi.json")
            return json.dumps(response)
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {str(e)}"

    def create_blueprint(self, data: dict) -> str:
        """Create a custom Linux image blueprint.

        ⚠️ CRITICAL: Only call this function after you have gathered ALL required information from the user.

        INFORMATION YOU MUST COLLECT FROM THE USER BEFORE CALLING:
        1. Blueprint name ("What would you like to name your blueprint? or should I generate a name?")
        2. Distribution ("Which distribution do you want? Available: {distributions}")
        3. Architecture ("Which architecture? Available: {architectures}")
        4. Image type ("What image type do you need? Available: {image_types} or take guest-image as default")
        5. Username ("Do you want to create a custom user account? If so, what username?")
        6. For RHEL images specifically: "Do you want to enable registration for Red Hat services?"
        7. Any customizations ("Do you need any specific packages, services, or configurations?")

        YOUR PROCESS AS THE AI ASSISTANT:
        1. If you haven't already, call get_openapi to understand the CreateBlueprintRequest structure
        2. Ask the user for ALL the required information listed above through conversation
        3. Only after collecting all information, call this function with properly formatted data

        Never make assumptions or fill in data yourself unless the user explicitly asks for it.
        Always ask the user for explicit input through conversation.

        Args:
            data: Complete blueprint data formatted according to CreateBlueprintRequest from OpenAPI spec

        Returns:
            The response from the image-builder API
        """
        try:
            client = self.get_client(get_http_headers())
        except ValueError as e:
            return self.no_auth_error(e)
        try:
            if os.environ.get("IMAGE_BUILDER_MCP_DISABLE_DESCRIPTION_WATERMARK", "").lower() != "true":
                desc_parts = [data.get("description", ""), "Blueprint created via image-builder-mcp"]
                data["description"] = "\n".join(filter(None, desc_parts))
            # TBD: programmatically check against openapi
            response = client.make_request(
                "blueprints", method="POST", data=data)
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {str(e)}"

        if isinstance(response, list):
            return "Error: the response of blueprint creation is a list. This is not expected. " \
                f"Response: {json.dumps(response)}"

        response_str = "[INSTRUCTION] Use the tool get_blueprint_details to get the details of the blueprint\n"
        response_str += "or ask the user to start the build/compose with blueprint_compose\n"
        response_str += f"Always show a link to the blueprint UI: {self.get_blueprint_url(client, response['id'])}\n"
        response_str += f"[ANSWER] Blueprint created successfully: {{'UUID': '{response['id']}'}}\n"
        response_str += "We could double check the details or start the build/compose"
        return response_str

    def get_blueprint_url(self, client: ImageBuilderClient, blueprint_id: str) -> str:
        """Get the URL for a blueprint."""
        return f"https://{client.domain}/insights/image-builder/imagewizard/{blueprint_id}"

    def get_blueprints(self, limit: int = 7, offset: int = 0, search_string: str | None = None) -> str:
        """EXECUTE: Show user's image blueprints (saved image templates/configurations for
        Linux distributions, packages, users).

        Args:
            limit: maximum number of items to return (default: 7)
            offset: number of items to skip (default: 0)
            search_string: substring to search for in the name (optional)

        Returns:
            List of blueprints with their UUIDs and details
        """

        try:
            client = self.get_client(get_http_headers())
        except ValueError as e:
            return self.no_auth_error(e)

        # workaround seen in LLama 3.3 70B Instruct
        if search_string == "null":
            search_string = None

        limit = limit or self.default_response_size
        if limit <= 0:
            limit = self.default_response_size
        try:
            # Make request with limit and offset parameters
            params = {"limit": limit, "offset": offset}
            response = client.make_request("blueprints", params=params)

            if isinstance(response, list):
                return "Error: the response of get_blueprints is a list. This is not expected. " \
                    f"Response: {json.dumps(response)}"

            # Sort data by created_at
            sorted_data = sorted(response["data"],
                                 key=lambda x: x.get("last_modified_at", ""),
                                 reverse=True)

            ret: list[dict] = []
            for i, blueprint in enumerate(sorted_data, 1):
                data = {"reply_id": i + offset,
                        "blueprint_uuid": blueprint["id"],
                        "UI_URL": self.get_blueprint_url(client, blueprint["id"]),
                        "name": blueprint["name"]}

                # Apply search filter if provided
                if search_string:
                    if search_string.lower() in data["name"].lower():
                        ret.append(data)
                else:
                    ret.append(data)

            intro = "[INSTRUCTION] Use the UI_URL to link to the blueprint\n"
            intro += "[ANSWER]\n"
            return f"{intro}\n{json.dumps(ret)}"
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {str(e)}"

    def get_blueprint_details(self, blueprint_identifier: str) -> str:
        """Get blueprint details.

        Args:
            blueprint_identifier: the UUID, name or reply_id to query

        Returns:
            Blueprint details

        Raises:
            Exception: If the image-builder connection fails.
        """
        if not blueprint_identifier:
            return "Error: a blueprint identifier is required"
        try:
            client = self.get_client(get_http_headers())
        except ValueError as e:
            return self.no_auth_error(e)

        try:
            # If the identifier looks like a UUID, use it directly
            if len(blueprint_identifier) == 36 and blueprint_identifier.count('-') == 4:
                response = client.make_request(f"blueprints/{blueprint_identifier}")
                if isinstance(response, dict):
                    return json.dumps([response])

                return json.dumps([{"error": "Unexpected list response", "data": response}])
            ret = f"[INSTRUCTION] Error: {blueprint_identifier} is not a valid blueprint identifier,"
            ret += "please use the UUID from get_blueprints\n"
            ret += "[INSTRUCTION] retry calling get_blueprints\n\n"
            ret += f"[ANSWER] {blueprint_identifier} is not a valid blueprint identifier"
            return ret
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {str(e)}"

    def _create_compose_data(self, compose: dict, reply_id: int, client: ImageBuilderClient) -> dict:
        """Create compose data dictionary with blueprint URL."""
        data = {
            "reply_id": reply_id,
            "compose_uuid": compose["id"],
            "blueprint_id": compose.get("blueprint_id", "N/A"),
            "image_name": compose.get("image_name", "")
        }

        if compose.get("blueprint_id"):
            data["blueprint_url"] = (f"https://{client.domain}/insights/image-builder/"
                                     f"imagewizard/{compose['blueprint_id']}")
        else:
            data["blueprint_url"] = "N/A"

        return data

    def _should_include_compose(self, data: dict, search_string: str | None) -> bool:
        """Determine if compose should be included based on search criteria."""
        if not search_string:
            return True
        return search_string.lower() in data["image_name"].lower()

    # NOTE: the _doc_ has escaped curly braces as __doc__.format() is called on the docstring
    def get_composes(self, limit: int = 7, offset: int = 0, search_string: str | None = None) -> str:
        """Get a list of all image builds (composes) with their UUIDs and basic status.

        **ALWAYS USE THIS FIRST** when checking image build status or finding builds.
        This returns the UUID needed for get_compose_details.

        Common uses:
        - Check status of recent builds → call this first
        - Find your latest build → call this first
        - Get any build information → call this first
        Ask the user if they want to get more composes and adapt "offset" accordingly.

        Args:
            limit: maximum number of items to return (default: 7)
            offset: number of items to skip (default: 0)
            search_string: substring to search for in the name (optional)

        Returns:
            List of composes with:
            - uuid: The unique identifier (REQUIRED for get_compose_details)
            - name: Blueprint name used
            - status: Current build status
            - created_at: When the build started

        Example response:
        [
            {{
                "uuid": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "name": "my-rhel-image",
                "status": "RUNNING",
                "created_at": "2025-01-18T10:30:00Z"
            }}
        ]
        """
        limit = limit or self.default_response_size
        if limit <= 0:
            limit = self.default_response_size
        try:
            client = self.get_client(get_http_headers())

            # Make request with limit and offset parameters
            params = {"limit": limit, "offset": offset}
            response = client.make_request("composes", params=params)

            if isinstance(response, list):
                return (f"Error: the response of get_composes is a list. This is not expected. "
                        f"Response: {json.dumps(response)}")

            # Sort data by created_at
            sorted_data = sorted(response["data"],
                                 key=lambda x: x.get("created_at", ""),
                                 reverse=True)

            ret: list[dict] = []
            for i, compose in enumerate(sorted_data, 1):
                data = self._create_compose_data(compose, i + offset, client)

                # Apply search filter if provided
                if self._should_include_compose(data, search_string):
                    ret.append(data)

            intro = ("[INSTRUCTION] Present a bulleted list and use the blueprint_url to link to the "
                     "blueprint which created this compose\n")
            intro += "[ANSWER]\n"
            return f"{intro}\n{json.dumps(ret)}"

        except ValueError as e:
            return self.no_auth_error(e)
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {str(e)}"

    def get_compose_details(self, compose_identifier: str) -> str:
        """Get detailed information about a specific image build.

        ⚠️ REQUIRES: You MUST have the compose UUID from get_composes() first.
        ⚠️ NEVER call this with generic terms like "latest", "recent", or "my build"

        Process:
        1. User asks about build status → call get_composes()
        2. Find the desired compose and copy its UUID
        3. Call this function with that exact UUID

        Args:
            compose_identifier: The exact UUID string from get_composes()
                            Example: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
                            NOT: "latest", "recent", "my-image", etc.

        Returns:
            Detailed compose information including:
            - Full status and progress
            - Error messages if failed
            - Download URLs if completed
            - Build logs
            - Artifact details
        """
        if not compose_identifier:
            return "Error: Compose UUID is required"
        try:
            client = self.get_client(get_http_headers())
        except ValueError as e:
            return self.no_auth_error(e)

        try:
            # If the identifier looks like a UUID, use it directly
            if len(compose_identifier) == 36 and compose_identifier.count('-') == 4:
                response = client.make_request(f"composes/{compose_identifier}")
                if isinstance(response, list):
                    self.logger.error(
                        "Error: the response of get_compose_details is a list. "
                        "This is not expected. Response for %s: %s",
                        compose_identifier, json.dumps(response))
                    return f"Error: Unexpected list response for {compose_identifier}"
                response["compose_uuid"] = compose_identifier
            else:
                ret = (f"[INSTRUCTION] Error: {compose_identifier} is not a valid compose identifier,"
                       "please use the UUID from get_composes\n")
                ret += "[INSTRUCTION] retry calling get_composes\n\n"
                ret += f"[ANSWER] {compose_identifier} is not a valid compose identifier"
                return ret

            intro = ""
            download_url = response.get("image_status", {}).get(
                "upload_status", {}).get("options", {}).get("url")
            upload_target = response.get("image_status", {}).get(
                "upload_status", {}).get("type")

            if download_url and upload_target == "oci.objectstorage":
                intro += """
[INSTRUCTION] Leave the URL as code block so the user can copy and paste it.

To run the image copy the link below and follow the steps below:

   * Go to "Compute" in Oracle Cloud and choose "Custom Images".
   * Click on "Import image", choose "Import from an object storage URL".
   * Choose "Import from an object storage URL" and paste the URL in the "Object Storage URL" field. The image type has to be set to QCOW2 and the launch mode should be paravirtualized.

```
{download_url}
```
"""
            elif download_url:
                intro += f"The image is available at [{download_url}]({download_url})\n"
                intro += "Always present this link to the user\n"
            # else depends on the status and the target if it can be downloaded

            return f"{intro}{json.dumps(response)}"
        # avoid crashing the server so we'll stick to the broad exception catch
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"Error: {e}"


def main():
    """Main entry point for the Image Builder MCP server."""
    parser = argparse.ArgumentParser(
        description="Run Image Builder MCP server.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--stage", action="store_true", help="Use stage API instead of production API")

    # Create subparsers for different transport modes
    subparsers = parser.add_subparsers(dest="transport", help="Transport mode")

    # stdio subcommand (default)
    subparsers.add_parser("stdio", help="Use stdio transport (default)")

    # sse subcommand
    sse_parser = subparsers.add_parser("sse", help="Use SSE transport")
    sse_parser.add_argument("--host", default="127.0.0.1", help="Host for SSE transport (default: 127.0.0.1)")
    sse_parser.add_argument("--port", type=int, default=9000, help="Port for SSE transport (default: 9000)")

    # http subcommand
    http_parser = subparsers.add_parser(
        "http", help="Use HTTP streaming transport")
    http_parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport (default: 127.0.0.1)")
    http_parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport (default: 8000)")

    args = parser.parse_args()

    # Default to stdio if no subcommand is provided
    if args.transport is None:
        args.transport = "stdio"

    # Get credentials from environment variables or user input
    client_id = os.getenv("IMAGE_BUILDER_CLIENT_ID")
    client_secret = os.getenv("IMAGE_BUILDER_CLIENT_SECRET")

    proxy_url = None
    if args.stage:
        proxy_url = os.getenv("IMAGE_BUILDER_STAGE_PROXY_URL")
        if not proxy_url:
            print("Please set IMAGE_BUILDER_STAGE_PROXY_URL to access the stage API")
            print("hint: IMAGE_BUILDER_STAGE_PROXY_URL=http://yoursquidproxy…:3128")
            sys.exit(1)

    if args.debug:
        logging.getLogger("ImageBuilderMCP").setLevel(logging.DEBUG)
        logging.getLogger("ImageBuilderClient").setLevel(logging.DEBUG)
        logging.getLogger("ImageBuilderOAuthMiddleware").setLevel(logging.DEBUG)
        logging.info("Debug mode enabled")

    oauth_enabled = os.getenv("OAUTH_ENABLED", "false").lower() == "true"

    # Create and run the MCP server
    mcp_server = ImageBuilderMCP(
        client_id,
        client_secret,
        stage=args.stage,
        proxy_url=proxy_url,
        transport=args.transport,
        oauth_enabled=oauth_enabled,
    )

    if args.transport == "sse":
        mcp_server.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "http":
        if oauth_enabled:
            app = mcp_server.http_app(transport="http")
            self_url = os.getenv(
                "SELF_URL",
                f"http://{args.host}:{args.port}",
            )
            oauth_url = os.getenv(
                "OAUTH_URL",
                "https://sso.redhat.com/auth/realms/redhat-external",
            )
            oauth_client = os.getenv("OAUTH_CLIENT")
            if not oauth_client:
                logging.fatal("OAUTH_CLIENT environment variable is required for OAuth-enabled HTTP transport")
                sys.exit(1)

            app.add_middleware(
                Middleware,
                self_url=self_url,
                oauth_url=oauth_url,
                oauth_client=oauth_client,
            )

            # Start the application
            uvicorn.run(app, host=args.host, port=args.port)
        else:
            mcp_server.run(transport="http", host=args.host, port=args.port)
    else:
        mcp_server.run()


if __name__ == "__main__":
    main()
