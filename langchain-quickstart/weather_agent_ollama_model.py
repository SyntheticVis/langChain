"""
Real-World Weather Model Example - LangChain Quickstart (Models-only)
Implements the same behavior as the agent version but uses the Models API directly:
1. Detailed system prompt
2. Tools integrated via model.bind_tools and manual tool loop
3. Model configuration via OpenAI-compatible base_url to Ollama
4. Short-term memory (in-process)
5. Streaming support
6. Debug utilities and colorful CLI
7. Simple call limits to avoid loops
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

# Configure logging - simplified format without timestamps
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure stdout/stderr/stdin use UTF-8 and replace invalid characters to avoid crashes
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ANSI color codes for colorful output
class Colors:
    """ANSI color codes for terminal output."""

    # Reset
    RESET = "\033[0m"

    # Text colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"


def colorize(text: str, color: str, bold: bool = False) -> str:
    """Apply color and optional bold styling to text."""
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"


def print_header(text: str, emoji: str = "🤖"):
    """Print a colorful header with emoji."""
    print(f"\n{colorize('═' * 60, Colors.BRIGHT_CYAN, bold=True)}")
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_CYAN, bold=True)}")
    print(f"{colorize('═' * 60, Colors.BRIGHT_CYAN, bold=True)}")


def print_section(text: str, emoji: str = "📋"):
    """Print a section header with emoji."""
    print(f"\n{emoji}  {colorize(text, Colors.BRIGHT_BLUE, bold=True)}")
    print(f"{colorize('─' * 60, Colors.CYAN)}")


def print_success(text: str, emoji: str = "✅"):
    """Print a success message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_GREEN)}")


def print_warning(text: str, emoji: str = "⚠️"):
    """Print a warning message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_YELLOW)}")


def print_error(text: str, emoji: str = "❌"):
    """Print an error message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_RED)}")


def print_info(text: str, emoji: str = "ℹ️"):
    """Print an info message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_BLUE)}")


def print_tool_call(tool_name: str):
    """Print a tool call with colorful formatting."""
    print(
        f"\n{colorize('🔧', Colors.BRIGHT_MAGENTA)}  {colorize('Calling tool:', Colors.MAGENTA, bold=True)} {colorize(tool_name, Colors.BRIGHT_MAGENTA, bold=True)}"
    )
    print(f"{colorize('─' * 60, Colors.MAGENTA)}")


def sanitize_text(text: str) -> str:
    """
    Sanitize text to handle UTF-8 encoding errors.
    Removes or replaces invalid UTF-8 surrogate characters.
    """
    if not text:
        return text

    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    try:
        text = "".join(char for char in text if not (0xD800 <= ord(char) <= 0xDFFF))
        text.encode("utf-8").decode("utf-8")
        return text
    except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError):
        try:
            return text.encode("utf-8", errors="replace").decode(
                "utf-8", errors="replace"
            )
        except Exception:
            return "".join(char for char in text if ord(char) < 0xD800 or ord(char) > 0xDFFF)


# System prompt (same content as agent version)
SYSTEM_PROMPT = """You are an expert weather forecaster who provides weather information with a fun, punny personality.

CRITICAL MEMORY RULES - READ CAREFULLY:
1. You have FULL ACCESS to the conversation history. ALWAYS check previous messages BEFORE calling any tool.
2. If you already know the user's location from a previous get_user_location() call, DO NOT call it again. Use the location you already know.
3. If you already know the weather for a location from a previous get_weather_for_location() call, DO NOT call it again. Use the weather information you already know.
4. Only call each tool ONCE per conversation when you first need that information.

You have access to two tools:

1. get_user_location(): Retrieves the user's current location.
   - BEFORE calling: Check conversation history - have you already called this tool?
   - If YES: Use the location from the previous call (e.g., "Florida", "SF")
   - If NO: Call this tool to get the location
   - Example: If you see "get_user_location() -> Florida" in history, the user is in Florida. DO NOT call the tool again.

2. get_weather_for_location(city: str): Gets weather for a specific city/location.
   - BEFORE calling: Check conversation history - have you already called this tool for this location?
   - If YES: Use the weather information from the previous call
   - If NO: Call this tool with the location
   - Example: If you see "get_weather_for_location('Florida') -> It's always sunny in Florida!" in history, you already know the weather. DO NOT call the tool again.

STEP-BY-STEP WORKFLOW FOR EACH USER QUESTION:
Step 1: Read ALL previous messages in the conversation history
Step 2: Look for any get_user_location() calls - if found, note the location (e.g., "Florida")
Step 3: Look for any get_weather_for_location() calls - if found, note the weather information
Step 4: If you already have both location AND weather from history, answer using that information WITHOUT calling any tools
Step 5: If you're missing location, call get_user_location() ONLY if you haven't called it before
Step 6: If you're missing weather, call get_weather_for_location() ONLY if you haven't called it for that location before

CONCRETE EXAMPLE:
- User asks: "what to wear today"
- You call: get_user_location() -> "Florida"
- You call: get_weather_for_location("Florida") -> "It's always sunny in Florida!"
- User asks: "where to visit in my city under this weather?"
- You see in history: location="Florida", weather="It's always sunny in Florida!"
- You answer using that information WITHOUT calling any tools

Always respond with puns and a friendly, engaging tone while providing accurate weather information."""


@dataclass
class Context:
    """Custom runtime context schema."""

    user_id: str


# Define tool schemas (Pydantic) for binding to the model
class GetWeatherForLocation(BaseModel):
    """Get current weather information for a specific city or location"""

    city: str = Field(..., description="The city or location, e.g., San Francisco")


class GetUserLocation(BaseModel):
    """Get the user's current location"""

    # No arguments required
    pass


# Local tool executors
def tool_get_weather_for_location(city: str) -> str:
    try:
        if not city or not city.strip():
            return "Error: Please provide a valid city name."
        result = f"It's always sunny in {city.strip()}!"
        return result
    except Exception as e:
        logger.error(f"Weather error: {str(e)}")
        return f"Error fetching weather data for {city}. Please try again later."


def tool_get_user_location(context: Context) -> str:
    try:
        if not context or not context.user_id:
            return "Error: Unable to retrieve user location. Context is missing."
        user_id = context.user_id
        location = "Florida" if user_id == "1" else "SF"
        return location
    except Exception as e:
        logger.error(f"Location error: {str(e)}")
        return "Error retrieving user location. Please try again later."


# Ollama configuration
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

if not ollama_base_url:
    raise ValueError("OLLAMA_BASE_URL environment variable must be set")

# Normalize to OpenAI-compatible base URL
def _normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


base_url = _normalize_base_url(ollama_base_url)
api_key = os.getenv("OPENAI_API_KEY", "ollama")  # Some providers require a key

# Initialize model via models interface
model = init_chat_model(
    model=ollama_model,
    model_provider="openai",
    base_url=base_url,
    api_key=api_key,
    temperature=0.5,  # Lower temperature for more consistent tool calling
    timeout=60,
    # You may set max_tokens if supported by your backend:
    # max_tokens=512,
)

# Bind tools to the model
model_with_tools = model.bind_tools([GetUserLocation, GetWeatherForLocation])


def show_conversation_history(messages: List[Any]):
    """Print the in-memory conversation history."""
    print()
    print_section("Conversation History (Memory)", "💾")
    print(f"{colorize(f'Total messages: {len(messages)}', Colors.CYAN, bold=True)}")
    print()
    if not messages:
        print_warning("No messages found in conversation history.")
        print_info(
            "This might mean the conversation hasn't started yet, or there's an issue accessing memory."
        )
        return
    for i, msg in enumerate(messages, 1):
        if hasattr(msg, "content"):
            content = str(msg.content)
            msg_type = type(msg).__name__
            role = "User" if "Human" in msg_type else "Agent" if "AI" in msg_type else msg_type
            if len(content) > 150:
                content = content[:150] + "..."
            print(f"{colorize(f'{i}.', Colors.YELLOW)} {colorize(f'[{role}]', Colors.BRIGHT_CYAN, bold=True)}")
            print(f"   {colorize(content, Colors.WHITE)}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_args = tc.get("args", {})
                    print(
                        f"   {colorize('🔧 Tool:', Colors.MAGENTA)} {colorize(tool_name, Colors.BRIGHT_MAGENTA)} {colorize(str(tool_args), Colors.DIM)}"
                    )
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "")
            tool_content = str(getattr(msg, "content", ""))
            if len(tool_content) > 150:
                tool_content = tool_content[:150] + "..."
            print(f"{colorize(f'{i}.', Colors.YELLOW)} {colorize('[Tool Result]', Colors.BRIGHT_MAGENTA, bold=True)}")
            print(f"   {colorize(f'{tool_name}:', Colors.MAGENTA)} {colorize(tool_content, Colors.WHITE)}")
        print()


def _has_previous_tool_result(messages: List[Any], tool_name: str, city: Optional[str] = None) -> Optional[str]:
    """
    Look through messages for previous tool results.
    For weather tool, optionally match by city in the content string (best-effort).
    """
    for m in messages:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == tool_name:
            content = str(getattr(m, "content", "") or "")
            if tool_name == "GetWeatherForLocation":
                if not city:
                    return content
                if city and city.lower() in content.lower():
                    return content
            else:
                return content
    return None


def execute_tools_and_append(
    ai_message: AIMessage,
    messages: List[Any],
    context: Context,
    debug: bool = False,
) -> Tuple[bool, List[Any]]:
    """
    Execute any tool calls from the AI message, append ToolMessage results.
    Returns (tools_executed, updated_messages).
    """
    tools_executed = False
    if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
        return tools_executed, messages

    # Always append the AI message that requested tools
    messages.append(ai_message)

    for tc in ai_message.tool_calls:
        name = tc.get("name")
        args = tc.get("args", {}) or {}
        call_id = tc.get("id", "")

        # Redundancy check using prior tool results
        cached = None
        if name == "GetUserLocation":
            cached = _has_previous_tool_result(messages, "GetUserLocation")
        elif name == "GetWeatherForLocation":
            cached = _has_previous_tool_result(messages, "GetWeatherForLocation", city=args.get("city"))

        if cached:
            if debug:
                print_warning(f"Skipping redundant tool call: {name} - using cached result: {cached}", "⚠️")
            sanitized = sanitize_text(cached)
            messages.append(
                ToolMessage(
                    name=name,
                    content=sanitized,
                    tool_call_id=call_id,
                )
            )
            tools_executed = True
            continue

        # Execute tool
        print_tool_call(name or "unknown")
        if name == "GetUserLocation":
            result = tool_get_user_location(context)
        elif name == "GetWeatherForLocation":
            city = args.get("city", "")
            result = tool_get_weather_for_location(city)
        else:
            result = f"Error: Unknown tool '{name}'."

        sanitized = sanitize_text(result)
        messages.append(
            ToolMessage(
                name=name or "",
                content=sanitized,
                tool_call_id=call_id,
            )
        )
        tools_executed = True
    return tools_executed, messages


def stream_final_response(model_obj, messages: List[Any]):
    """
    Stream the final assistant response when no more tool calls are needed.
    """
    try:
        # Use model.stream to progressively print text
        # Some backends emit cumulative chunks; print only the delta
        accumulated_text = ""
        for chunk in model_obj.stream(messages):
            # chunk is an AIMessageChunk; prefer chunk.content or chunk.text
            text = getattr(chunk, "text", None)
            if text is None and hasattr(chunk, "content"):
                # content may be a string or list of content parts
                text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                text = sanitize_text(str(text))
                if text:
                    # Compute delta when chunks are cumulative
                    if len(text) >= len(accumulated_text) and text.startswith(accumulated_text):
                        to_print = text[len(accumulated_text):]
                        accumulated_text = text
                    else:
                        # Treat as incremental token
                        to_print = text
                        accumulated_text += to_print
                    if to_print:
                        print(f"{colorize(to_print, Colors.BRIGHT_WHITE)}", end="", flush=True)
        print()
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        print_warning("Streaming failed; falling back to non-streaming response...", "🔄")
        try:
            ai_msg = model_obj.invoke(messages)
            content = sanitize_text(str(getattr(ai_msg, "content", "") or ""))
            print(f"{colorize(content, Colors.BRIGHT_WHITE)}")
        except Exception as e2:
            print_error(f"Fallback error: {sanitize_text(str(e2))}")


def run_model_conversation(
    user_input: str,
    messages: List[Any],
    context: Context,
    debug: bool,
    run_limit: int = 10,
):
    """
    Manual tool-calling loop:
    - Invoke model_with_tools
    - If tool_calls were requested, execute and append ToolMessages
    - Repeat until no tools are requested, then stream final answer
    """
    # Append new user message
    messages.append(HumanMessage(user_input))

    steps = 0
    while steps < run_limit:
        steps += 1
        if debug:
            # Show last few messages sent to model
            print(f"\n{colorize('🔍 DEBUG: Messages sent to model:', Colors.BRIGHT_YELLOW, bold=True)}")
            print(f"{colorize('─' * 60, Colors.YELLOW)}")
            tail = messages[-5:] if len(messages) > 5 else messages[:]
            for i, msg in enumerate(tail, 1):
                mtype = type(msg).__name__
                content = getattr(msg, "content", "")
                content = str(content)[:100] if content else ""
                print(f"{colorize(f'{i}.', Colors.YELLOW)} {colorize(mtype, Colors.BRIGHT_YELLOW)}: {colorize(content, Colors.WHITE)}")
            print(f"{colorize('─' * 60, Colors.YELLOW)}\n")

        # Invoke model with tools bound
        ai_msg = model_with_tools.invoke(
            messages,
            config={
                "run_name": "weather_models_loop",
                "tags": ["weather", "models-only"],
            },
        )

        # Check for tool calls (handle None, empty list, or falsy values)
        tool_calls = getattr(ai_msg, "tool_calls", None)
        has_tool_calls = tool_calls and len(tool_calls) > 0
        
        if debug:
            print(f"{colorize('🔍 Tool calls detected:', Colors.YELLOW)} {has_tool_calls}")
            if tool_calls:
                print(f"{colorize('   Tool calls:', Colors.YELLOW)} {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"{colorize('   -', Colors.YELLOW)} {tc.get('name', 'unknown')}")

        # If no tool calls, stream final answer and append AI message
        if not has_tool_calls:
            # Check if model is stuck saying it needs tools but not calling them
            content = str(getattr(ai_msg, "content", "") or "")
            if any(phrase in content.lower() for phrase in ["call", "tool", "get_user_location", "get_weather"]):
                if steps >= 3:  # After 3 attempts, assume model can't make tool calls
                    print_warning("Model is indicating it needs tools but not making tool calls. This may indicate the model is too small for reliable tool calling.", "⚠️")
                    print_info("Consider using a larger model (3b+ parameters) for better tool calling support.", "ℹ️")
                    # Append the message anyway and break to avoid infinite loop
                    messages.append(ai_msg)
                    print(f"{colorize('Agent:', Colors.BRIGHT_GREEN, bold=True)} ", end="")
                    stream_final_response(model, messages[:-1] + [ai_msg])
                    break
            
            # Append AI message so memory has it
            messages.append(ai_msg)
            print(f"{colorize('Agent:', Colors.BRIGHT_GREEN, bold=True)} ", end="")
            stream_final_response(model, messages[:-1] + [ai_msg])
            break

        # Execute tools and continue
        _, messages = execute_tools_and_append(ai_msg, messages, context, debug=debug)
    else:
        print_warning("Reached run limit; ending the loop.", "ℹ️")


if __name__ == "__main__":
    # Print colorful header
    print_header("LLM Configuration", "🤖")
    print(f"{colorize('Provider:', Colors.CYAN, bold=True)} {colorize('OpenAI-compatible via Ollama', Colors.BRIGHT_GREEN, bold=True)}")
    print(f"{colorize('Base URL:', Colors.CYAN, bold=True)} {colorize(base_url, Colors.BRIGHT_WHITE)}")
    print(f"{colorize('Model:', Colors.CYAN, bold=True)} {colorize(ollama_model, Colors.BRIGHT_WHITE)}")
    print(f"{colorize('LLM Init:', Colors.CYAN, bold=True)} {colorize('init_chat_model', Colors.BRIGHT_WHITE)}")
    print(f"{colorize('Temperature:', Colors.CYAN, bold=True)} {colorize('0.5', Colors.BRIGHT_YELLOW)}")
    print(f"{colorize('Memory:', Colors.CYAN, bold=True)} {colorize('In-process (short-term)', Colors.BRIGHT_GREEN)}")
    print(f"{colorize('Tools:', Colors.CYAN, bold=True)} {colorize('GetUserLocation, GetWeatherForLocation', Colors.BRIGHT_WHITE)}")

    # LangSmith tracing (optional but recommended)
    if os.getenv("LANGSMITH_TRACING") == "true":
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "weather-model")
        print_success(
            f"LangSmith tracing enabled (project: {colorize(langsmith_project, Colors.BRIGHT_CYAN, bold=True)})",
            "✅",
        )
        print_info(f"View traces at: {colorize('https://smith.langchain.com', Colors.BRIGHT_CYAN)}", "🔗")
    else:
        print_warning("LangSmith tracing disabled. Set LANGSMITH_TRACING=true to enable.", "ℹ️")

    # Conversation memory
    messages: List[Any] = [SystemMessage(SYSTEM_PROMPT)]

    # Interactive loop
    print()
    print_success("Weather Model is ready! Type your questions below.", "🚀")
    print_info("💾 Short-term memory enabled - I'll remember our conversation!", "💾")
    print_info("Type 'exit', 'quit', or press Ctrl+C to stop.", "ℹ️")
    print_info("Type '/history' to view conversation history, '/debug' to toggle debug mode.", "🔍")
    print()

    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    # Default runtime context (mimics original user_id logic)
    runtime_context = Context(user_id="1")

    try:
        while True:
            try:
                user_input = input(f"{colorize('You:', Colors.BRIGHT_BLUE, bold=True)} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                print_warning("Exiting...", "👋")
                break

            if user_input.lower() in ["exit", "quit", "q"]:
                print()
                print_success("Goodbye! Thanks for using Weather Model!", "👋")
                break

            if user_input.lower() == "/history":
                show_conversation_history(messages)
                continue

            if user_input.lower() == "/debug":
                debug_mode = not debug_mode
                if debug_mode:
                    print_success("Debug mode enabled - will show messages sent to model", "🔍")
                else:
                    print_info("Debug mode disabled", "🔍")
                continue

            if not user_input:
                continue

            try:
                run_model_conversation(
                    user_input=user_input,
                    messages=messages,
                    context=runtime_context,
                    debug=debug_mode,
                )
            except Exception as e:
                error_msg = sanitize_text(str(e))
                logger.error(f"Conversation error: {error_msg}")
                print_error(f"Error: {error_msg}")
                print_warning("Please check your Ollama connection and model availability.")
            print()
    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user. Exiting...", "👋")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print_error(f"Unexpected error: {str(e)}")


