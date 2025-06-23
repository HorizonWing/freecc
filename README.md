# Any LLM in Claude Code

[简体中文](README_zh.md)

> **Note**: This project is a fork of [CogAgent/claude-code-proxy](https://github.com/CogAgent/claude-code-proxy).

**Use any LiteLLM-supported model in Claude Code without Pro subscription!**

This is an Anthropic API proxy that translates API requests from Claude Code into the format of any backend service supported by LiteLLM.

![Anthropic API Proxy](pic.png)

## Quick Start

### Prerequisites

- An API key for the model you want to use.
- [uv](https://github.com/astral-sh/uv) installed.

### Installation and Configuration

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/chachako/freecc.git
    cd freecc
    ```

2.  **Install uv** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    *(uv will handle dependencies based on `pyproject.toml` when you run the server)*

3.  **Configure Environment Variables**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file to set up your model routing.

    **Model Mapping Configuration**

    > Claude Code calls a small model, `haiku`, for auxiliary tasks in the background, so you can choose a fast, low-cost model. For the main tasks, it calls a large model, `sonnet`.

    This is the primary configuration for routing `sonnet` and `haiku` model requests. It allows you to point these common model types to specific providers, each with its own API key and base URL.

    -   `BIG_MODEL_PROVIDER`: Provider for "big" models (e.g., `openai`, `vertex`, `xai`).
    -   `BIG_MODEL_NAME`: The actual model name to use (e.g., `gpt-4.1`).
    -   `BIG_MODEL_API_KEY`: The API key for this provider.
    -   `BIG_MODEL_API_BASE`: (Optional) A custom API endpoint for this provider.

    -   `SMALL_MODEL_PROVIDER`: Provider for "small" models.
    -   `SMALL_MODEL_NAME`: The actual model name to use.
    -   `SMALL_MODEL_API_KEY`: The API key for this provider.
    -   `SMALL_MODEL_API_BASE`: (Optional) A custom API endpoint.

    **Global Provider Configuration**

    This configuration is for direct, provider-prefixed requests (e.g., `openai/gpt-4o-mini`). It also acts as a fallback for mapped models if a specific key or base URL is not provided above.

    -   `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `ANTHROPIC_API_KEY`
    -   `OPENAI_API_BASE`, `GEMINI_API_BASE`, `XAI_API_BASE`, `ANTHROPIC_API_BASE`

    **Vertex AI Specifics:**
    Vertex AI requires a Project ID and Location. These are global settings:
    -   `VERTEX_PROJECT_ID`: Your Google Cloud Project ID.
    -   `VERTEX_LOCATION`: The region for your Vertex AI resources.
    -   Set up Application Default Credentials (ADC) via `gcloud` or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

    **Logging Configuration**

    You can control the logging behavior via the `.env` file:
    -   `FILE_LOG_LEVEL`: Sets the log level for `claude-proxy.log`. Defaults to `DEBUG`.
    -   `CONSOLE_LOG_LEVEL`: Sets the log level for the console output. Defaults to `INFO`.
        -   Supported levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
    -   `LOG_REQUEST_BODY`: Set to `true` to log the full body of incoming requests. This is highly recommended if you are interested in the prompt engineering behind Claude Code 🤑.
    -   `LOG_RESPONSE_BODY`: Set to `true` to log the full body of responses from the downstream model.

4.  **Run the server**:
    ```bash
    uv run uvicorn server:app --host 127.0.0.1 --port 8082 --reload
    ```
    *(`--reload` is optional for development)*

### Using with Claude Code

1.  **Install Claude Code** (if you haven't already):
    ```bash
    npm install -g @anthropic-ai/claude-code
    ```

2.  **Connect to your proxy**:
    ```bash
    export ANTHROPIC_BASE_URL=http://localhost:8082 && claude
    ```

3.  **Enjoy!** Embrace a Claude Code with a larger context window.

### Configuration Examples

**Example 1: Standard OpenAI API Compatible Configuration**
*Map both large and small models to different OpenAI models.*
```dotenv
# .env
BIG_MODEL_PROVIDER="openai"
BIG_MODEL_NAME="gpt-4.1"
BIG_MODEL_API_KEY="sk-..."

SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL_NAME="gpt-4o-mini"
SMALL_MODEL_API_KEY="sk-..."
SMALL_MODEL_API_BASE="https://xyz.llm.com/v1" # Can be a different BASE URL
```

**Example 2**
*Use a fast model for `haiku` and a powerful model for `sonnet`.*
```dotenv
# .env
# For 'sonnet', use Vertex AI's Gemini 1.5 Pro
BIG_MODEL_PROVIDER="vertex"
BIG_MODEL_NAME="gemini-1.5-pro-preview-0514"
VERTEX_PROJECT_ID="your-gcp-project-id" # Required for Vertex

# For 'haiku', use a local, OpenAI-compatible model running on port 8000
SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL_NAME="local-llama-3"
SMALL_MODEL_API_BASE="http://localhost:8000/v1"
SMALL_MODEL_API_KEY="lm-studio" # A key might not be needed for local models
```

## How It Works

When a request from Claude Code contains a model name like `sonnet` or `haiku` (e.g., `claude-3-sonnet-20240229`), this proxy uses your configured `BIG_MODEL_*` or `SMALL_MODEL_*` variables to route the request. This means you can replace the default Claude Sonnet 4 in CC with any model that is cheaper or has a larger context window.

As long as the provider and model name are valid for LiteLLM, the proxy request will work.

The process is as follows:

1.  **Receive Request**: Receives the request in Anthropic API format.
2.  **Translate Format**: Translates the request to the target provider's format via LiteLLM.
3.  **Inject Credentials**: Dynamically injects the API key and endpoint based on your configuration.
4.  **Send Request**: Sends the translated request to the selected provider.
5.  **Translate Response**: Converts the provider's response back to the Anthropic format.
6.  **Return Response**: Returns the formatted response to the client.

## Vibe

This project is maintained by Claude Code & Gemini 2.5 Pro 🤪.

## Contributing

Pull Requests are welcome.
