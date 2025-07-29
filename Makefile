build-prod: ## Build the container image but with the upstream tag
	podman build --tag ghcr.io/osbuild/image-builder-mcp:latest .

build: ## Build the container image
	podman build --tag image-builder-mcp .

# please set from outside
TAG ?= UNKNOWN

build-claude-extension: ## Build the Claude extension
	sed -i "s/---VERSION---/$(TAG)/g" claude_desktop/manifest.json
	zip -j image-builder-$(TAG).cdx claude_desktop/*
	sed -i "s/$(TAG)/---VERSION---/g" claude_desktop/manifest.json

lint: ## Run linting with pre-commit
	pre-commit run --all-files

test: ## Run tests with pytest (hides logging output)
	@echo "Running pytest tests..."
	env DEEPEVAL_TELEMETRY_OPT_OUT=YES pytest -v

test-verbose: ## Run tests with pytest with verbose output (shows logging output)
	@echo "Running pytest tests with verbose output..."
	env DEEPEVAL_TELEMETRY_OPT_OUT=YES pytest -vv -o log_cli=true

test-very-verbose: ## Run tests with pytest showing all intermediate agent steps (shows logging output)
	@echo "Running pytest tests with debug output..."
	env DEEPEVAL_TELEMETRY_OPT_OUT=YES pytest -vvv -o log_cli=true

test-coverage: ## Run tests with coverage reporting
	@echo "Running pytest tests with coverage..."
	env DEEPEVAL_TELEMETRY_OPT_OUT=YES pytest -v --cov=. --cov-report=html --cov-report=term-missing

install-test-deps: ## Install test dependencies
	pip install -e .[dev]

clean-test: ## Clean test artifacts and cache
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

help: ## Show this help message
	@echo "make [TARGETS...]"
	@echo
	@echo 'Targets:'
	@awk 'match($$0, /^([a-zA-Z_\/-]+):.*? ## (.*)$$/, m) {printf "  \033[36m%-30s\033[0m %s\n", m[1], m[2]}' $(MAKEFILE_LIST) | sort

.PHONY: build test test-coverage install-test-deps clean-test help run-sse run-http run-stdio

# `IMAGE_BUILDER_CLIENT_ID` and `IMAGE_BUILDER_CLIENT_SECRET` are optional
# if you hand those over via http headers from the client.
run-sse: build ## Run the MCP server with SSE transport
	# add firewall rules for fedora
	podman run --rm --network=host --env IMAGE_BUILDER_CLIENT_ID --env IMAGE_BUILDER_CLIENT_SECRET --name image-builder-mcp-sse localhost/image-builder-mcp:latest sse

run-http: build ## Run the MCP server with HTTP streaming transport
	# add firewall rules for fedora
	podman run --rm --network=host --env IMAGE_BUILDER_CLIENT_ID --env IMAGE_BUILDER_CLIENT_SECRET --name image-builder-mcp-http localhost/image-builder-mcp:latest http

# just an example command
# doesn't really make sense
# rather integrate this with an MCP client directly
run-stdio: build ## Run the MCP server with stdio transport
	podman run --interactive --tty --rm --env IMAGE_BUILDER_CLIENT_ID --env IMAGE_BUILDER_CLIENT_SECRET --name image-builder-mcp-stdio localhost/image-builder-mcp:latest
