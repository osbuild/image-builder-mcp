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

Then run using either:

```
python -m image_builder_mcp sse
```

or using the CLI entry point:

```
image-builder-mcp sse
```

This will start image-builder-mcp server at http://localhost:9000/sse

### Using Podman/Docker

You can also copy the command from the [Makefile]
For SSE mode:
```
make run-sse
```

You can also copy the command from the [Makefile]
For stdio mode:
```
make run-stdio
```

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
                "ghcr.io/schuellerf/image-builder-mcp:latest"
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

Try the one-click installer

[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install-mcp?name=image-builder-mcp&config=JTdCJTIydHlwZSUyMiUzQSUyMnN0ZGlvJTIyJTJDJTIyY29tbWFuZCUyMiUzQSUyMnBvZG1hbiUyMHJ1biUyMC0tZW52JTIwSU1BR0VfQlVJTERFUl9DTElFTlRfSUQlMjAtLWVudiUyMElNQUdFX0JVSUxERVJfQ0xJRU5UX1NFQ1JFVCUyMC0taW50ZXJhY3RpdmUlMjAtLXJtJTIwZ2hjci5pbyUyRnNjaHVlbGxlcmYlMkZpbWFnZS1idWlsZGVyLW1jcCUzQWxhdGVzdCUyMiUyQyUyMmVudiUyMiUzQSU3QiUyMlJFTU9WRVBSRUZJWF9JTUFHRV9CVUlMREVSX0NMSUVOVF9JRCUyMiUzQSUyMllPVVJfSUQlMjBoZXJlJTJDJTIwdGhlbiUyMHJlbW92ZSUyMCdSRU1PVkVQUkVGSVhfJyUyMiUyQyUyMlJFTU9WRVBSRUZJWF9JTUFHRV9CVUlMREVSX0NMSUVOVF9TRUNSRVQlMjIlM0ElMjJZT1VSX1NFQ1JFVCUyMGhlcmUlMkMlMjB0aGVuJTIwcmVtb3ZlJTIwJ1JFTU9WRVBSRUZJWF8nJTIyJTdEJTdE)

or continue reading and install the config manually.

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
            "ghcr.io/schuellerf/image-builder-mcp:latest"
        ],
        "env": {
            "REMOVEPREFIX_IMAGE_BUILDER_CLIENT_ID": "YOUR_ID here, then remove 'REMOVEPREFIX_'",
            "REMOVEPREFIX_IMAGE_BUILDER_CLIENT_SECRET": "YOUR_SECRET here, then remove 'REMOVEPREFIX_'"
        }
    }
  }
}
```
