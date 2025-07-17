# Image Builder MCP

This repo is in a DRAFT and playground state!

An MCP server to interact with [hosted image builder](https://osbuild.org/docs/hosted/architecture/)

## Authentication

Go to https://console.redhat.com to `'YOUR USER' ➡ My User Access ➡ Service Accounts` create a service account
and then set the environment variables `IMAGE_BUILDER_CLIENT_ID` and `IMAGE_BUILDER_CLIENT_SECRET` accordingly.

## Run

### Using Python directly
Install the package in development mode:

```
pip install -e .
```

Then run using the CLI entry point:

```
image-builder-mcp sse
```

This will start `image-builder-mcp` server at http://localhost:9000/sse

For HTTP streaming transport:

```
`image-builder-mcp` http
```

This will start `image-builder-mcp` server with HTTP streaming transport at http://localhost:8000

### Using Podman/Docker

You can also copy the command from the [Makefile]
For SSE mode:
```
make run-sse
```

For HTTP streaming mode:
```
make run-http
```

You can also copy the command from the [Makefile]
For stdio mode:
```
make run-stdio
```

### Additional info

You can set the environment variable `IMAGE_BUILDER_MCP_DISABLE_DESCRIPTION_WATERMARK` to `True` to avoid
adding a hint to `image-builder-mcp` in newly created blueprints.

## Integrations

### VSCode
For the usage in your project, create a file called `.vscode/mcp.json` with
the following content.

```
{
    "inputs": [
        {
            "id": "image_builder_client_id",
            "type": "promptString",
            "description": "Enter the Image Builder Client ID",
            "default": ""
        },
        {
            "id": "image_builder_client_secret",
            "type": "promptString",
            "description": "Enter the Image Builder Client Secret",
            "default": ""
        }
    ],
    "servers": {
        "image-builder-mcp-stdio": {
            "type": "stdio",
            "command": "podman",
            "args": [
                "run",
                "--env",
                "IMAGE_BUILDER_CLIENT_ID",
                "--env",
                "IMAGE_BUILDER_CLIENT_SECRET",
                "--interactive",
                "--rm",
                "ghcr.io/osbuild/image-builder-mcp:latest"
            ],
            "env": {
                "IMAGE_BUILDER_CLIENT_ID": "${input:image_builder_client_id}",
                "IMAGE_BUILDER_CLIENT_SECRET": "${input:image_builder_client_secret}"
            }
        }
    }
}
```

### Cursor

Cursor doesn't seem to support `inputs` you need to add your credentials in the config file.
To start the integration create a file `~/.cursor/mcp.json` with
```
{
  "mcpServers": {
    "image-builder-mcp": {
        "type": "stdio",
        "command": "podman",
        "args": [
            "run",
            "--env",
            "IMAGE_BUILDER_CLIENT_ID",
            "--env",
            "IMAGE_BUILDER_CLIENT_SECRET",
            "--interactive",
            "--rm",
            "ghcr.io/osbuild/image-builder-mcp:latest"
        ],
        "env": {
            "removeprefix_IMAGE_BUILDER_CLIENT_ID": "YOUR_ID here, then remove 'removeprefix_'",
            "removeprefix_IMAGE_BUILDER_CLIENT_SECRET": "YOUR_SECRET here, then remove 'removeprefix_'"
        }
    }
  }
}
```

or use it via "Streamable HTTP"

start the server:

```
podman run --net host --rm ghcr.io/osbuild/image-builder-mcp:latest http
```

then integrate:

```
{
    "mcpServers": {
        "image-builder-mcp-http": {
            "type": "Streamable HTTP",
            "url": "http://localhost:8000/mcp/",
            "headers": {
                "removeprefix_image-builder-client-id": "YOUR_ID here, then remove 'removeprefix_'",
                "removeprefix_image-builder-client-secret": "YOUR_SECRET here, then remove 'removeprefix_'"
            }
        }
    }
}
```

### Generic STDIO

For generic integration into other tools via STDIO, you should set the environment variables
`IMAGE_BUILDER_CLIENT_ID` and `IMAGE_BUILDER_CLIENT_SECRET` and use this command for an
integration using podman:

```bash
podman run --env IMAGE_BUILDER_CLIENT_ID --env IMAGE_BUILDER_CLIENT_SECRET --interactive --rm ghcr.io/osbuild/image-builder-mcp:latest
```
