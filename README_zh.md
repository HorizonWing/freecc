# Any LLM in Claude Code

[English Version](README.md)

> **注意**: 本项目基于 [CogAgent/claude-code-proxy](https://github.com/CogAgent/claude-code-proxy) 修改而来。

**在 Claude Code 中使用任何 LiteLLM 支持的模型，而不需要订阅 Claude Pro!**

一个 Anthropic API 代理，用于将 Claude Code 中的 API 请求转换为由 LiteLLM 支持的任意后端服务格式。

![Anthropic API Proxy](pic.png)

## 快速开始

### 前提条件

- 准备好你要使用的模型的 API 密钥。
- 已安装 [uv](https://github.com/astral-sh/uv)。

### 安装与配置

1.  **克隆本仓库**:
    ```bash
    git clone https://github.com/chachako/freecc.git
    cd freecc
    ```

2.  **安装 uv** (若未安装):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    *(当您运行服务时，`uv` 会根据 `pyproject.toml` 文件自动处理依赖)*

3.  **配置环境变量**:
    复制环境文件示例:
    ```bash
    cp .env.example .env
    ```
    编辑 `.env` 文件以设置您的模型路由。

    **模型映射配置**

    > Claude Code 会在后台运行时调用一个小模型 `haiku` 来处理一些辅助性的工作，所以可以选择一个速度快、成本低的模型。正式处理工作时则会调用一个大型模型 `sonnet`。
    
    这是路由 `sonnet` 和 `haiku` 模型请求的主要配置。它允许您将这些常用的模型类型指向特定的提供商，并为其设置独立的 API 密钥和接口地址。

    -   `BIG_MODEL_PROVIDER`: 用于处理"大型"模型的提供商 (例如: `openai`, `vertex`, `xai`)。
    -   `BIG_MODEL_NAME`: 要使用的实际模型名称 (例如: `gpt-4.1`)。
    -   `BIG_MODEL_API_KEY`: 该提供商的 API 密钥。
    -   `BIG_MODEL_API_BASE`: (可选) 该提供商的自定义 API 地址。

    -   `SMALL_MODEL_PROVIDER`: 用于处理"小型"模型的提供商。
    -   `SMALL_MODEL_NAME`: 要使用的实际模型名称。
    -   `SMALL_MODEL_API_KEY`: 该提供商的 API 密钥。
    -   `SMALL_MODEL_API_BASE`: (可选) 该提供商的自定义 API 地址。

    **全局提供商配置**

    此配置用于处理带有提供商前缀的直接请求（例如 `openai/gpt-4o-mini`）。如果模型映射配置中未提供特定的密钥或 API 地址，此处的配置也可作为备用选项。

    -   `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `ANTHROPIC_API_KEY`
    -   `OPENAI_API_BASE`, `GEMINI_API_BASE`, `XAI_API_BASE`, `ANTHROPIC_API_BASE`

    **Vertex AI 特殊配置:**
    Vertex AI 需要项目 ID 和地理位置。这些是全局设置：
    -   `VERTEX_PROJECT_ID`: 您的 Google Cloud 项目 ID。
    -   `VERTEX_LOCATION`: 您的 Vertex AI 资源所在的区域。
    -   请通过 `gcloud` 设置应用默认凭证 (ADC)，或设置 `GOOGLE_APPLICATION_CREDENTIALS` 环境变量。

    **日志配置**

    您可以通过 `.env` 文件控制日志行为：
    -   `FILE_LOG_LEVEL`: 设置 `claude-proxy.log` 文件的日志级别。默认为 `DEBUG`。
    -   `CONSOLE_LOG_LEVEL`: 设置控制台输出的日志级别。默认为 `INFO`。
        -   支持的级别: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`。
    -   `LOG_REQUEST_BODY`: 设置为 `true` 以记录传入请求的完整内容。如果你对 Claude Code 背后的提示词工程感兴趣，则开启此选项 🤑。
    -   `LOG_RESPONSE_BODY`: 设置为 `true` 以记录下游模型的完整响应内容。

4.  **运行服务器**:
    ```bash
    uv run uvicorn server:app --host 127.0.0.1 --port 8082 --reload
    ```
    *(`--reload` 为可选参数，适用于开发环境)*

### 在 Claude Code 中使用

1.  **安装 Claude Code** (若未安装):
    ```bash
    npm install -g @anthropic-ai/claude-code
    ```

2.  **连接到您的代理**:
    ```bash
    export ANTHROPIC_BASE_URL=http://localhost:8082 && claude
    ```

3.  **Enjoy!** 拥抱更大上下文的 Claude Code。

### 配置示例

**示例 1: 标准 OpenAI API 兼容配置**
*将大型和小型模型都映射到不同的 OpenAI 模型。*
```dotenv
# .env
BIG_MODEL_PROVIDER="openai"
BIG_MODEL_NAME="gpt-4.1"
BIG_MODEL_API_KEY="sk-..."

SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL_NAME="gpt-4o-mini"
SMALL_MODEL_API_KEY="sk-..."
SMALL_MODEL_API_BASE="https://xyz.llm.com/v1" # 可以是不同的 BASE URL
```

**示例 2**
*为 `haiku` 使用一个快速模型，为 `sonnet` 使用一个强大模型。*
```dotenv
# .env
# 对于 'sonnet', 使用 Vertex AI 的 Gemini 1.5 Pro
BIG_MODEL_PROVIDER="vertex"
BIG_MODEL_NAME="gemini-1.5-pro-preview-0514"
VERTEX_PROJECT_ID="your-gcp-project-id" # 对于 Vertex 是必需的

# 对于 'haiku', 使用一个在本地 8000 端口运行的、兼容 OpenAI 的模型
SMALL_MODEL_PROVIDER="openai"
SMALL_MODEL_NAME="local-llama-3"
SMALL_MODEL_API_BASE="http://localhost:8000/v1"
SMALL_MODEL_API_KEY="lm-studio" # 本地模型可能不需要密钥
```

## 原理

当 Claude Code 发送的请求中，如果模型名称包含 `sonnet` 或 `haiku` (例如 `claude-3-sonnet-20240229`)，本代理会使用您配置的 `BIG_MODEL_*` 或 `SMALL_MODEL_*` 变量来路由请求。这就意味着我们可以用任何成本更低，或者上下文窗口更大的模型替换 CC 中默认的 Claude Sonnet 4。

只要提供商和模型名称对于 LiteLLM 是有效的，代理请求就能 Work。

流程如下:

1.  **接收请求**: 以 Anthropic 的 API 格式接收请求
2.  **转换格式**: 通过 LiteLLM 将请求转换为目标提供商的格式
3.  **动态注入凭证**: 根据您的配置，动态注入 API 密钥和接口地址
4.  **发送请求**: 将转换后的请求发送至选定的提供商
5.  **转换响应**: 将提供商的响应转换回 Anthropic 格式
6.  **返回响应**: 将格式化后的响应返回给客户端

## Vibe

本项目由 Claude Code & Gemini 2.5 Pro 维护 🤪。

## 贡献

欢迎 PR。 🎁
