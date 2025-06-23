from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import logging.handlers

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
# Get root logger, set level to DEBUG to catch all messages
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Remove any existing handlers to prevent conflicts or duplication
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create a file handler to log messages to a file
file_handler = logging.handlers.RotatingFileHandler(
    "claude-proxy.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB file, 5 backups
)
file_handler.setLevel(logging.DEBUG)  # Log everything to the file
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s [%(name)s] - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Create a console handler to show messages on the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Only show INFO and higher on console

# Add console handler to the root logger
root_logger.addHandler(console_handler)

# Get a logger for the current module
logger = logging.getLogger(__name__)

# --- Environment variable for logging ---
LOG_REQUEST_BODY = os.environ.get("LOG_REQUEST_BODY", "false").lower() == "true"
LOG_RESPONSE_BODY = os.environ.get("LOG_RESPONSE_BODY", "false").lower() == "true"

if LOG_REQUEST_BODY:
    logger.info("📄 Full request body logging is ENABLED.")
if LOG_RESPONSE_BODY:
    logger.info("📄 Full response body logging is ENABLED.")

# Configure uvicorn to be quieter
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]

        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Enable dropping unsupported parameters for LiteLLM
litellm.drop_params = True
logger.info("✅ Enabled LiteLLM drop_params=True to ignore unsupported API parameters.")

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"    # Vertex AI
    GREEN = "\033[92m"   # OpenAI
    YELLOW = "\033[93m"  # Gemini
    RED = "\033[91m"     # Anthropic (Direct)
    MAGENTA = "\033[95m" # Tools/Messages count
    CYAN = "\033[96m"    # Claude (Original) & xAI
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.INFO and isinstance(record.msg, str) and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys/credentials from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE") # Custom base URL for OpenAI-compatible APIs
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if OPENAI_API_BASE:
    logger.info(f"🟢 OpenAI custom base URL configured: {OPENAI_API_BASE}")

# For Vertex AI, LiteLLM typically uses Application Default Credentials (ADC)
# Ensure ADC is set up (e.g., `gcloud auth application-default login`)
# Or set GOOGLE_APPLICATION_CREDENTIALS environment variable
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID") # Required by LiteLLM for Vertex
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")  # Required by LiteLLM for Vertex

# Check if Vertex AI credentials are properly configured
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if VERTEX_PROJECT_ID:
    logger.info(f"🔵 Vertex AI Project ID configured: {VERTEX_PROJECT_ID}")
    logger.info(f"🔵 Vertex AI Location configured: {VERTEX_LOCATION}")
    if GOOGLE_APPLICATION_CREDENTIALS:
        logger.info(f"🔵 Vertex AI using service account credentials from: {GOOGLE_APPLICATION_CREDENTIALS}")
    else:
        logger.info("🔵 Vertex AI will use Application Default Credentials (ADC)")
else:
    logger.warning("⚠️ VERTEX_PROJECT_ID not set. Vertex AI models will not work correctly without it.")

# API Key for xAI
XAI_API_KEY = os.environ.get("XAI_API_KEY")
if XAI_API_KEY:
    logger.info("🟣 xAI API key configured")
else:
    logger.warning("⚠️ XAI_API_KEY not set. xAI models will not work without it.")

# GROQ_API_KEY removed

# Get preferred provider (default to openai)
# Possible values: 'openai', 'google', 'vertex', 'xai'
# PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()
# logger.info(f"🔧 Preferred provider set to: {PREFERRED_PROVIDER}")

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
# BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
# SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# --- New, more flexible model routing configuration ---
BIG_MODEL_PROVIDER = os.environ.get("BIG_MODEL_PROVIDER", "openai").lower()
BIG_MODEL_NAME = os.environ.get("BIG_MODEL_NAME", "gpt-4.1")
SMALL_MODEL_PROVIDER = os.environ.get("SMALL_MODEL_PROVIDER", "openai").lower()
SMALL_MODEL_NAME = os.environ.get("SMALL_MODEL_NAME", "gpt-4.1-mini")

logger.info(f"🔧 Big model mapping: 'sonnet' -> {BIG_MODEL_PROVIDER}/{BIG_MODEL_NAME}")
logger.info(f"🔧 Small model mapping: 'haiku' -> {SMALL_MODEL_PROVIDER}/{SMALL_MODEL_NAME}")

# Model lists (OPENAI_MODELS, GEMINI_MODELS, VERTEX_AI_MODELS, XAI_MODELS) have been removed
# to allow for flexible, user-defined model names.
# The logic now relies on provider prefixes (e.g., 'openai/') in the model name.

# Helper function to clean schema for Gemini/Vertex
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini/Vertex."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini/Vertex schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

# Updated ThinkingConfig to accept client format
class ThinkingConfigClient(BaseModel):
    budget_tokens: Optional[int] = None
    type: Optional[str] = None

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfigClient] = None # Use the updated client-facing model
    original_model: Optional[str] = None  # Will store the original model name
    is_mapped: bool = False # Internal flag to track if the model was mapped

    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value
        is_mapped_flag = False

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}'")

        # Remove provider prefixes for haiku/sonnet matching
        clean_v = v
        if '/' in clean_v:
            # e.g., "openai/claude-3-haiku-20240307" -> "claude-3-haiku-20240307"
            clean_v = clean_v.split('/', 1)[1]

        # --- Mapping Logic --- START ---
        mapped = False

        # Map Haiku (small) and Sonnet (big) based on independent provider configurations
        if 'haiku' in clean_v.lower():
            provider = SMALL_MODEL_PROVIDER
            if provider == "vertex": provider = "vertex_ai" # LiteLLM prefix for vertex
            new_model = f"{provider}/{SMALL_MODEL_NAME}"
            mapped = True
        elif 'sonnet' in clean_v.lower():
            provider = BIG_MODEL_PROVIDER
            if provider == "vertex": provider = "vertex_ai"
            new_model = f"{provider}/{BIG_MODEL_NAME}"
            mapped = True
        # --- Mapping Logic --- END ---

        if mapped:
            # Enhanced logging with provider-specific emojis
            if new_model.startswith("vertex_ai/"):
                logger.info(f"🔵 VERTEX MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            elif new_model.startswith("xai/"):
                logger.info(f"🟣 XAI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            elif new_model.startswith("gemini/"):
                logger.info(f"🟡 GEMINI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            elif new_model.startswith("openai/"):
                logger.info(f"🟢 OPENAI MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
            else:
                logger.info(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred, use the original model name.
             # If it's prefixed, the downstream logic will handle it.
             # If it's not prefixed, it will default to Anthropic or fail if no key is found.
             new_model = v
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'vertex_ai/', 'xai/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is. It will default to the Anthropic provider if not otherwise handled.")
             else:
                 logger.debug(f"ℹ️ Using already prefixed model: '{new_model}'")

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model
            values['is_mapped'] = is_mapped_flag

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfigClient] = None # Use the updated client-facing model
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    is_mapped: bool = False # Internal flag to track if the model was mapped

    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        original_model = v
        new_model = v # Default to original value
        is_mapped_flag = False

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}'")

        # Remove provider prefixes for haiku/sonnet matching
        clean_v = v
        if '/' in clean_v:
            clean_v = clean_v.split('/', 1)[1]

        # --- Mapping Logic --- START ---
        mapped = False

        if 'haiku' in clean_v.lower():
            provider = SMALL_MODEL_PROVIDER
            if provider == "vertex": provider = "vertex_ai"
            new_model = f"{provider}/{SMALL_MODEL_NAME}"
            mapped = True
        elif 'sonnet' in clean_v.lower():
            provider = BIG_MODEL_PROVIDER
            if provider == "vertex": provider = "vertex_ai"
            new_model = f"{provider}/{BIG_MODEL_NAME}"
            mapped = True
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             new_model = v # Use the original value
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'vertex_ai/', 'xai/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             # else: it's already prefixed, no special log needed here

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model
            values['is_mapped'] = is_mapped_flag

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path

    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")

    # Process the request and get the response
    response = await call_next(request)

    return response

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    messages = []

    # Add system message if present
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                text_content = ""
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            result_content = parse_tool_result_content(getattr(block, "content", None)) # Use helper
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            source = getattr(block, 'source', {})
                            if isinstance(source, dict) and source.get('type') == 'base64':
                                image_url = f"data:image/{source.get('media_type', 'jpeg')};base64,{source.get('data', '')}"
                                processed_content.append({"type": "image_url", "image_url": {"url": image_url}})
                            else:
                                logger.warning(f"Image block source format not explicitly handled for LiteLLM: {source}")
                                processed_content.append({"type": "image", "source": source})

                        elif block.type == "tool_use":
                             if msg.role == "assistant":
                                 tool_call_data = {
                                     "id": block.id,
                                     "type": "function",
                                     "function": {
                                         "name": block.name,
                                         "arguments": json.dumps(block.input)
                                     }
                                 }
                                 if not messages: messages.append({"role": msg.role, "content": None, "tool_calls": []})
                                 if messages[-1]["role"] != msg.role or "tool_calls" not in messages[-1]:
                                      messages.append({"role": msg.role, "content": None, "tool_calls": []})
                                 messages[-1]["tool_calls"].append(tool_call_data)
                             else:
                                 logger.error("Invalid Request: A 'tool_use' content block is only allowed in an 'assistant' message. This block will be ignored.")

                        elif block.type == "tool_result":
                             if msg.role == 'user':
                                 messages.append({
                                     "role": "tool",
                                     "tool_call_id": block.tool_use_id,
                                     "content": parse_tool_result_content(getattr(block, "content", None))
                                 })
                             else:
                                 logger.error("Invalid Request: A 'tool_result' content block is only allowed in a 'user' message. This block will be ignored.")
                        else:
                            logger.error(f"Unsupported content block type: '{getattr(block, 'type', 'unknown')}'. This block will be ignored.")
                if processed_content:
                     if messages and messages[-1]["role"] == "assistant" and messages[-1].get("content") is None and messages[-1].get("tool_calls"):
                         messages[-1]["content"] = processed_content
                     else:
                         messages.append({"role": msg.role, "content": processed_content})


    # LiteLLM request dict structure
    litellm_request = {
        "model": anthropic_request.model,
        "messages": messages,
        "max_tokens": anthropic_request.max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters
    if anthropic_request.stop_sequences: litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p: litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k: litellm_request["top_k"] = anthropic_request.top_k

    # Convert tools
    is_vertex_model = anthropic_request.model.startswith("vertex_ai/")
    is_gemini_model = anthropic_request.model.startswith("gemini/")

    if anthropic_request.tools:
        openai_tools = []
        for tool in anthropic_request.tools:
            if hasattr(tool, 'model_dump'): tool_dict = tool.model_dump(exclude_unset=True)
            elif hasattr(tool, 'dict'): tool_dict = tool.dict(exclude_unset=True)
            else:
                try: tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}"); continue
            input_schema = tool_dict.get("input_schema", {})
            if is_vertex_model or is_gemini_model:
                 target_provider = "Vertex AI" if is_vertex_model else "Gemini"
                 logger.debug(f"Cleaning schema for {target_provider} tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)
            openai_tools.append({
                "type": "function",
                "function": {"name": tool_dict["name"], "description": tool_dict.get("description", ""), "parameters": input_schema}
            })
        litellm_request["tools"] = openai_tools

    # Convert tool_choice
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'model_dump'): tool_choice_dict = anthropic_request.tool_choice.model_dump(exclude_unset=True)
        elif hasattr(anthropic_request.tool_choice, 'dict'): tool_choice_dict = anthropic_request.tool_choice.dict(exclude_unset=True)
        else: tool_choice_dict = anthropic_request.tool_choice

        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto": litellm_request["tool_choice"] = "auto"
        elif choice_type == "any": litellm_request["tool_choice"] = "required"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {"type": "function", "function": {"name": tool_choice_dict["name"]}}
        else: litellm_request["tool_choice"] = "auto"

    # Clean up messages for final request
    litellm_request["messages"] = [m for m in litellm_request["messages"] if m.get("content") or m.get("tool_calls")]

    # Handle thinking field conversion
    if anthropic_request.thinking and anthropic_request.thinking.type == "enabled":
        # Add the format expected by the downstream API
        litellm_request["thinking"] = {"enabled": True}
        logger.debug("💡 Added 'thinking: {\"enabled\": True}' to LiteLLM request.")
    # Else: if thinking is None or type != "enabled", don't add thinking field

    return litellm_request


def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any],
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    try:
        prompt_tokens, completion_tokens, response_id = 0, 0, f"msg_{uuid.uuid4()}"
        choices, message, finish_reason = [], {}, "stop"

        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            choices, usage_info = litellm_response.choices, litellm_response.usage
            response_id = getattr(litellm_response, 'id', response_id)
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        elif isinstance(litellm_response, dict):
             choices = litellm_response.get("choices", [{}])
             usage_info = litellm_response.get("usage", {})
             response_id = litellm_response.get("id", response_id)
             prompt_tokens = usage_info.get("prompt_tokens", 0)
             completion_tokens = usage_info.get("completion_tokens", 0)
        else: # Try conversion
            try:
                response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                choices = response_dict.get("choices", [{}])
                usage_info = response_dict.get("usage", {})
                response_id = response_dict.get("id", response_id)
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
            except Exception as conv_err:
                 logger.error(f"Could not convert litellm_response: {conv_err}")
                 raise conv_err # Re-raise after logging

        if choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                 message = first_choice.get("message", {})
                 finish_reason = first_choice.get("finish_reason", "stop")
            elif hasattr(first_choice, 'message'):
                 message = first_choice.message
                 finish_reason = getattr(first_choice, 'finish_reason', "stop")
                 if hasattr(message, 'model_dump'): message = message.model_dump()
                 elif hasattr(message, '__dict__'): message = message.__dict__

        content_text = message.get("content", "")
        tool_calls = message.get("tool_calls", None)
        content = []
        if content_text: content.append({"type": "text", "text": content_text})

        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")
            if not isinstance(tool_calls, list): tool_calls = [tool_calls]
            for tool_call in tool_calls:
                 tool_call_dict = {}
                 if isinstance(tool_call, dict): tool_call_dict = tool_call
                 elif hasattr(tool_call, 'model_dump'): tool_call_dict = tool_call.model_dump()
                 elif hasattr(tool_call, '__dict__'): tool_call_dict = tool_call.__dict__
                 function_call = tool_call_dict.get("function", {})
                 tool_id = tool_call_dict.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                 name = function_call.get("name", "")
                 arguments_str = function_call.get("arguments", "{}")
                 try: arguments = json.loads(arguments_str)
                 except json.JSONDecodeError: arguments = {"raw_arguments": arguments_str}
                 content.append({"type": "tool_use", "id": tool_id, "name": name, "input": arguments})

        stop_reason_map = {"length": "max_tokens", "tool_calls": "tool_use", "stop": "end_turn"}
        stop_reason = stop_reason_map.get(finish_reason, finish_reason or "end_turn")

        if not content: content.append({"type": "text", "text": ""})

        return MessagesResponse(
            id=response_id, model=original_request.original_model or original_request.model,
            role="assistant", content=content, stop_reason=stop_reason, stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
        )
    except Exception as e:
        import traceback
        logger.error(f"Error converting response: {str(e)}\n{traceback.format_exc()}")
        return MessagesResponse(
            id=f"msg_error_{uuid.uuid4()}", model=original_request.model, role="assistant",
            content=[{"type": "text", "text": f"Error converting backend response: {str(e)}"}],
            stop_reason="error", usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': original_request.original_model or original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

        text_buffer, tool_buffers = "", {}
        current_block_index, text_block_index = -1, -1
        tool_block_indices = {}
        final_usage = {"input_tokens": 0, "output_tokens": 0}
        final_stop_reason = "end_turn"

        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        async for chunk in response_generator:
            if hasattr(chunk, 'usage') and chunk.usage:
                final_usage["input_tokens"] = getattr(chunk.usage, 'prompt_tokens', final_usage["input_tokens"])
                final_usage["output_tokens"] = getattr(chunk.usage, 'completion_tokens', final_usage["output_tokens"])

            if not chunk.choices: continue
            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                stop_reason_map = {"length": "max_tokens", "tool_calls": "tool_use", "stop": "end_turn"}
                final_stop_reason = stop_reason_map.get(choice.finish_reason, choice.finish_reason or "end_turn")

            if delta.content:
                if text_block_index == -1:
                    current_block_index += 1; text_block_index = current_block_index
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': text_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                text_buffer += delta.content
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': text_block_index, 'delta': {'type': 'text_delta', 'text': delta.content}})}\n\n"

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    tool_index = tool_call_delta.index
                    if tool_index not in tool_block_indices:
                        current_block_index += 1; anthropic_block_index = current_block_index
                        tool_block_indices[tool_index] = anthropic_block_index
                        tool_id = tool_call_delta.id or f"toolu_{uuid.uuid4().hex[:24]}"
                        tool_name = tool_call_delta.function.name if tool_call_delta.function else ""
                        tool_buffers[anthropic_block_index] = {"id": tool_id, "name": tool_name, "input": ""}
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_block_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': {}}})}\n\n"

                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        anthropic_block_index = tool_block_indices[tool_index]
                        arg_chunk = tool_call_delta.function.arguments
                        tool_buffers[anthropic_block_index]["input"] += arg_chunk
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_block_index, 'delta': {'type': 'input_json_delta', 'partial_json': arg_chunk}})}\n\n"

        if text_block_index != -1: yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': text_block_index})}\n\n"
        for _, anthropic_idx in tool_block_indices.items(): yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': anthropic_idx})}\n\n"

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': final_usage['output_tokens']}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    except Exception as e:
        import traceback
        logger.error(f"Error during streaming conversion: {e}\n{traceback.format_exc()}")
        try:
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error'}, 'usage': {'output_tokens': 0}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        except: pass


@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        # Use the original model name from the raw request for display purposes.
        original_model_for_display = body_json.get("model", "unknown")

        display_model = original_model_for_display
        if "/" in display_model: display_model = display_model.split("/")[-1]

        logger.debug(f"📊 PROCESSING REQUEST: Original Model='{original_model_for_display}', Mapped Model='{request.model}', Stream={request.stream}, IsMapped={request.is_mapped}")

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)

        # --- Dynamic Credential/Endpoint Injection ---
        # This logic determines which API key and base URL to use based on the request type.
        if request.is_mapped:
            # Tier 1: Mapped requests ('sonnet'/'haiku') use their specific configurations.
            original_clean = original_model_for_display.split('/')[-1]
            if 'haiku' in original_clean.lower():
                if SMALL_MODEL_API_KEY: litellm_request["api_key"] = SMALL_MODEL_API_KEY
                if SMALL_MODEL_API_BASE: litellm_request["api_base"] = SMALL_MODEL_API_BASE
                logger.debug(f"Using SMALL model mapped credentials for {request.model}")
            elif 'sonnet' in original_clean.lower():
                if BIG_MODEL_API_KEY: litellm_request["api_key"] = BIG_MODEL_API_KEY
                if BIG_MODEL_API_BASE: litellm_request["api_base"] = BIG_MODEL_API_BASE
                logger.debug(f"Using BIG model mapped credentials for {request.model}")
        else:
            # Tier 2: Direct, prefixed requests use the global provider configurations.
            provider = request.model.split('/')[0] if '/' in request.model else None
            if provider == "openai":
                if OPENAI_API_KEY: litellm_request["api_key"] = OPENAI_API_KEY
                if OPENAI_API_BASE: litellm_request["api_base"] = OPENAI_API_BASE
                logger.debug(f"Using global OpenAI credentials for {request.model}")
            elif provider == "gemini":
                if GEMINI_API_KEY: litellm_request["api_key"] = GEMINI_API_KEY
                if GEMINI_API_BASE: litellm_request["api_base"] = GEMINI_API_BASE
                logger.debug(f"Using global Gemini credentials for {request.model}")
            elif provider == "xai":
                if XAI_API_KEY: litellm_request["api_key"] = XAI_API_KEY
                if XAI_API_BASE: litellm_request["api_base"] = XAI_API_BASE
                logger.debug(f"Using global xAI credentials for {request.model}")
            elif provider == "anthropic":
                if ANTHROPIC_API_KEY: litellm_request["api_key"] = ANTHROPIC_API_KEY
                if ANTHROPIC_API_BASE: litellm_request["api_base"] = ANTHROPIC_API_BASE
                logger.debug(f"Using global Anthropic credentials for {request.model}")
            elif provider == "vertex_ai":
                # Vertex AI uses project/location instead of a single base URL.
                if VERTEX_PROJECT_ID:
                    litellm_request["vertex_project"] = VERTEX_PROJECT_ID
                    litellm_request["vertex_location"] = VERTEX_LOCATION
                    logger.debug(f"Using Vertex AI credentials for {request.model}")
                else:
                    logger.warning(f"⚠️ VERTEX_PROJECT_ID not set for Vertex AI model {request.model}.")
            else:
                # Fallback for models without a known prefix or a dedicated API key.
                if ANTHROPIC_API_KEY: litellm_request["api_key"] = ANTHROPIC_API_KEY
                logger.debug(f"Using default Anthropic API key for model: {request.model}")

        if LOG_REQUEST_BODY:
            # Log the request for debugging, filtering out the API key.
            request_for_logging = {k: v for k, v in litellm_request.items() if k != 'api_key'}
            try:
                logger.debug(f"LiteLLM Request Body:\n{json.dumps(request_for_logging, indent=2, ensure_ascii=False)}")
            except Exception:
                logger.debug(f"LiteLLM Request Body (fallback): {request_for_logging}")

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", raw_request.url.path, display_model, request.model,
            len(litellm_request['messages']), num_tools, 200
        )

        if request.stream:
            response_generator = await litellm.acompletion(**litellm_request)
            return StreamingResponse(handle_streaming(response_generator, request), media_type="text/event-stream")
        else:
            start_time = time.time()
            logger.info(f"🚀 Sending request to {request.model}...")
            litellm_response = await litellm.acompletion(**litellm_request)
            elapsed_time = time.time() - start_time

            if LOG_RESPONSE_BODY:
                try:
                    logger.debug(f"LiteLLM Response Body:\n{json.dumps(litellm_response.model_dump(), indent=2, ensure_ascii=False)}")
                except Exception:
                    logger.debug(f"LiteLLM Response Body (fallback): {litellm_response}")

            # Enhanced provider-specific response logging
            if request.model.startswith("vertex_ai/"):
                logger.info(f"🔵 VERTEX RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            elif request.model.startswith("xai/"):
                logger.info(f"🟣 XAI RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            elif request.model.startswith("gemini/"):
                logger.info(f"🟡 GEMINI RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            elif request.model.startswith("openai/"):
                logger.info(f"🟢 OPENAI RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            else:
                logger.info(f"✅ RESPONSE RECEIVED: Model={request.model}, Time={elapsed_time:.2f}s")
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            return anthropic_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_details = {"error": str(e), "type": type(e).__name__, "traceback": error_traceback}
        # Safely add additional error attributes, converting non-serializable ones
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                value = getattr(e, attr)
                # Convert potentially non-serializable types to strings for JSON logging
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    try:
                        value = str(value) # Attempt string conversion
                    except:
                        value = f"<{type(value).__name__} object (unserializable)>" # Fallback
                error_details[attr] = value
        try:
            logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")
        except Exception as log_e:
            # Fallback logging if json.dumps still fails (shouldn't happen now, but safe)
            logger.error(f"Error processing request (fallback log): {error_details}")
            logger.error(f"Logging failed due to: {log_e}")

        status_code = getattr(e, 'status_code', 500)
        # Ensure detail_message is a string
        detail_message = getattr(e, 'message', None) # Get original message if possible
        if detail_message is None:
            detail_message = str(e) # Fallback to string representation of exception
        if isinstance(detail_message, bytes): detail_message = detail_message.decode('utf-8', errors='ignore')
        raise HTTPException(status_code=status_code, detail=str(detail_message))


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        original_model = request.original_model or request.model
        display_model = original_model
        if "/" in display_model: display_model = display_model.split("/")[-1]

        # Clean model name for logging/prefix check
        provider_prefix = ""
        clean_model_name = request.model
        if clean_model_name.startswith("anthropic/"): provider_prefix = "anthropic/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("openai/"): provider_prefix = "openai/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("gemini/"): provider_prefix = "gemini/"; clean_model_name = clean_model_name[len(provider_prefix):]
        elif clean_model_name.startswith("vertex_ai/"): provider_prefix = "vertex_ai/"; clean_model_name = clean_model_name[len(provider_prefix):]
        # elif clean_model_name.startswith("groq/"): provider_prefix = "groq/"; clean_model_name = clean_model_name[len(provider_prefix):] # Removed Groq check
        elif clean_model_name.startswith("xai/"): provider_prefix = "xai/"; clean_model_name = clean_model_name[len(provider_prefix):]

        temp_msg_request_data = request.model_dump()
        temp_msg_request_data['max_tokens'] = 1 # Dummy value
        temp_msg_request = MessagesRequest(**temp_msg_request_data)
        converted_request = convert_anthropic_to_litellm(temp_msg_request)

        from litellm import token_counter

        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", raw_request.url.path, display_model, request.model,
            len(converted_request['messages']), num_tools, 200
        )

        token_count = token_counter(
            model=converted_request["model"],
            messages=converted_request["messages"],
        )
        return TokenCountResponse(input_tokens=token_count)

    except ImportError:
        logger.error("Could not import token_counter from litellm")
        return TokenCountResponse(input_tokens=1000) # Fallback
    except Exception as e:
        import traceback
        logger.error(f"Error counting tokens: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"    # Claude (Original) & xAI
    BLUE = "\033[94m"    # Vertex AI
    GREEN = "\033[92m"   # OpenAI
    YELLOW = "\033[93m"  # Gemini
    RED = "\033[91m"     # Anthropic (Direct)
    MAGENTA = "\033[95m" # Tools/Messages count
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

def log_request_beautifully(method, path, original_model_display, mapped_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing model mapping."""
    original_display = f"{Colors.CYAN}{original_model_display}{Colors.RESET}" # Original uses Claude's color for now

    endpoint = path
    if "?" in endpoint: endpoint = endpoint.split("?")[0]

    target_provider = "unknown"
    target_model_name = mapped_model
    target_color = Colors.GREEN # Default

    if "/" in mapped_model:
        try:
            target_provider, target_model_name = mapped_model.split("/", 1)
            if target_provider == "openai": target_color = Colors.GREEN
            elif target_provider == "gemini": target_color = Colors.YELLOW
            elif target_provider == "vertex_ai": target_color = Colors.BLUE
            # elif target_provider == "groq": target_color = Colors.MAGENTA # Removed Groq color
            elif target_provider == "xai": target_color = Colors.CYAN # Use Cyan for xAI
            elif target_provider == "anthropic": target_color = Colors.RED
        except ValueError:
            logger.warning(f"Could not parse provider from mapped model: {mapped_model}")
            target_provider = "unknown"
            target_model_name = mapped_model

    target_display = f"{target_color}{target_model_name}{Colors.RESET}"
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{original_display} → {target_display} ({target_provider}) {tools_str} {messages_str}"

    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
