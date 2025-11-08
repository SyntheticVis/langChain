"""
Real-World Weather Agent Example - LangChain Quickstart
Adapted for Ollama (local LLM) instead of OpenAI
Demonstrates:
1. Detailed system prompts
2. Tools that integrate with external data
3. Model configuration with Ollama
4. Structured output
5. Conversational memory
6. Full agent creation and execution
7. LangSmith tracing and observability
8. Error handling and graceful degradation
9. Streaming support for real-time feedback
10. Improved tool descriptions for better tool calling
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LangSmith tracing is automatically enabled when environment variables are set:
# - LANGSMITH_TRACING=true
# - LANGSMITH_API_KEY=<your-api-key>
# - LANGSMITH_PROJECT=<project-name> (optional but recommended)
# Get your API key from https://smith.langchain.com


# Define system prompt with improved clarity
SYSTEM_PROMPT = """You are an expert weather forecaster who provides weather information with a fun, punny personality.

You have access to two tools:

1. get_weather_for_location(city: str): Use this tool to get weather information for a specific city or location.
   - Call this when the user explicitly mentions a city name or location.
   - Example: "What's the weather in New York?" -> use get_weather_for_location("New York")

2. get_user_location(): Use this tool to retrieve the user's current location.
   - Call this when the user asks about weather at their current location or "where I am".
   - Example: "What's the weather outside?" -> first use get_user_location(), then get_weather_for_location()

IMPORTANT: Always determine the location before providing weather information. If the user's question implies their current location, use get_user_location() first to find out where they are, then use get_weather_for_location() with that location.

Always respond with puns and a friendly, engaging tone while providing accurate weather information."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Define tools with improved error handling and descriptions
@tool
def get_weather_for_location(city: str) -> str:
    """
    Get current weather information for a specific city or location.
    
    This tool retrieves weather data including temperature, conditions, and forecast
    for the specified location. Use this when the user asks about weather in a
    specific place.
    
    Args:
        city: The name of the city or location to get weather for (e.g., "New York", "London", "Tokyo")
    
    Returns:
        A string containing weather information for the specified location.
    
    Example:
        get_weather_for_location("San Francisco") -> "It's always sunny in San Francisco!"
    """
    try:
        if not city or not city.strip():
            logger.warning("Empty city name provided to get_weather_for_location")
            return "Error: Please provide a valid city name."
        
        # In a real implementation, this would call an actual weather API
        # For now, this is a mock implementation
        result = f"It's always sunny in {city.strip()}!"
        logger.info(f"Weather retrieved for location: {city}")
        return result
    except Exception as e:
        logger.error(f"Error getting weather for {city}: {str(e)}")
        return f"Error fetching weather data for {city}. Please try again later."


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """
    Retrieve the user's current location based on their user ID.
    
    This tool looks up the user's location from their profile or session data.
    Use this when the user asks about weather at their current location or
    "where I am" without specifying a city.
    
    Args:
        runtime: The tool runtime context containing user information
    
    Returns:
        A string containing the user's location (e.g., "Florida", "San Francisco")
    
    Example:
        get_user_location(runtime) -> "Florida"
    """
    try:
        if not runtime or not runtime.context:
            logger.warning("Missing context in get_user_location")
            return "Error: Unable to retrieve user location. Context is missing."
        
        user_id = runtime.context.user_id
        if not user_id:
            logger.warning("Missing user_id in context")
            return "Error: User ID not found in context."
        
        # In a real implementation, this would query a user database
        location = "Florida" if user_id == "1" else "SF"
        logger.info(f"Location retrieved for user {user_id}: {location}")
        return location
    except Exception as e:
        logger.error(f"Error getting user location: {str(e)}")
        return "Error retrieving user location. Please try again later."


# Configure model - using Ollama (local LLM)
# Ollama configuration from environment variables
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Validate configuration
if not ollama_base_url:
    raise ValueError("OLLAMA_BASE_URL environment variable must be set")

# Model configuration with improved defaults
# Note: num_ctx controls context window size, adjust based on model capabilities
model = ChatOllama(
    model=ollama_model,
    base_url=ollama_base_url,
    temperature=0.5,  # Lower temperature for more consistent tool calling
    timeout=60,  # Increase timeout for slower models
    num_ctx=4096,  # Context window size (adjust based on model)
    streaming=True,  # Enable streaming for incremental responses
    # Additional options for better tool calling:
    # num_predict=512,  # Max tokens to generate
    # top_p=0.9,  # Nucleus sampling
    # repeat_penalty=1.1,  # Reduce repetition
)

logger.info(f"Initialized ChatOllama model: {ollama_model} at {ollama_base_url}")


# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# Set up memory
checkpointer = InMemorySaver()

# Create agent
# Use ToolStrategy for structured output with Ollama (works with any tool-calling model)
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),  # Use ToolStrategy for Ollama compatibility
    checkpointer=checkpointer
)

logger.info("Agent created successfully with tools and structured output support")


def extract_response(response: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Extract and format the agent response, handling both structured and unstructured outputs.
    
    This helper function handles the complexity of extracting responses from the agent,
    including structured responses, tool calls, and fallback to message content.
    
    Args:
        response: The response dictionary from agent.invoke()
        verbose: Whether to print debug information
    
    Returns:
        A dictionary containing:
        - structured_response: The structured response if available
        - message_content: The message content as fallback
        - tool_calls: List of tool calls made
        - has_structured: Boolean indicating if structured response is available
    """
    result = {
        "structured_response": None,
        "message_content": None,
        "tool_calls": [],
        "has_structured": False
    }
    
    # Check for tool calls in messages
    if response.get("messages"):
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                result["tool_calls"] = msg.tool_calls
                if verbose:
                    logger.debug(f"Tool calls detected: {msg.tool_calls}")
                break
    
    # Handle structured response (preferred)
    if 'structured_response' in response and response['structured_response']:
        result["structured_response"] = response['structured_response']
        result["has_structured"] = True
        if verbose:
            logger.info("Structured response available")
    else:
        # Fallback to message content
        if response.get("messages"):
            last_msg = response["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                result["message_content"] = last_msg.content
                if verbose:
                    logger.warning("No structured response, using message content")
                    if not result["tool_calls"]:
                        logger.warning(
                            "No tool calls detected. The model may be too small for reliable tool calling. "
                            "Consider using a larger model (3b+ parameters) for better tool calling support."
                        )
            else:
                if verbose:
                    logger.warning("No content found in response messages")
        else:
            if verbose:
                logger.warning("No messages found in response")
    
    return result


def print_response(extracted: Dict[str, Any], show_tool_calls: bool = True):
    """
    Print the extracted response in a user-friendly format.
    
    Args:
        extracted: The result dictionary from extract_response()
        show_tool_calls: Whether to display tool call information
    """
    if show_tool_calls and extracted["tool_calls"]:
        print(f"🔧 Tool calls made: {extracted['tool_calls']}")
    
    if extracted["has_structured"]:
        print("✅ Structured Response:")
        print(extracted['structured_response'])
    else:
        print("⚠️  No structured response available (falling back to message content)")
        if not extracted["tool_calls"]:
            print("⚠️  Warning: No tool calls were made. The model may be too small for reliable tool calling.")
            print("   Consider using a larger model (3b+ parameters) for better tool calling support.")
        
        if extracted["message_content"]:
            print(f"Agent Response: {extracted['message_content']}")
        else:
            print("⚠️  No response content available")


# Run agent
if __name__ == "__main__":
    # Check if environment variables are set
    if not ollama_base_url:
        logger.error("OLLAMA_BASE_URL environment variable is not set")
        print("Error: OLLAMA_BASE_URL environment variable is not set.")
        print("Please set: export OLLAMA_BASE_URL=<your-ollama-url>")
        exit(1)
    
    print("=" * 50)
    print("🤖 LLM Configuration")
    print("=" * 50)
    print(f"Provider: Ollama")
    print(f"Base URL: {ollama_base_url}")
    print(f"Model: {ollama_model}")
    print(f"LLM Class: ChatOllama")
    print(f"Temperature: 0.5")
    print(f"Context Window: 4096")
    print("=" * 50)
    
    # LangSmith tracing (optional but recommended)
    if os.getenv("LANGSMITH_TRACING") == "true":
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "weather-agent")
        print(f"✅ LangSmith tracing enabled (project: {langsmith_project})")
        print(f"   View traces at: https://smith.langchain.com")
        logger.info(f"LangSmith tracing enabled for project: {langsmith_project}")
    else:
        print("ℹ️  LangSmith tracing disabled. Set LANGSMITH_TRACING=true to enable.")
        logger.info("LangSmith tracing disabled")
    
    # `thread_id` is a unique identifier for a given conversation.
    config = {"configurable": {"thread_id": "1"}}

    # Example 1: First question with tool usage
    print("\n=== First Question ===")
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
            config=config,
            context=Context(user_id="1")
        )
        
        extracted = extract_response(response, verbose=True)
        print_response(extracted, show_tool_calls=True)
        # Expected: ResponseFormat with punny_response and weather_conditions
        
    except Exception as e:
        logger.error(f"Error during agent invocation: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")
        print("Please check your Ollama connection and model availability.")

    # Example 2: Follow-up question (continuing conversation)
    print("\n=== Follow-up Question ===")
    # Note that we can continue the conversation using the same `thread_id`.
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "thank you! see you later!"}]},
            config=config,
            context=Context(user_id="1")
        )
        
        extracted = extract_response(response, verbose=True)
        print_response(extracted, show_tool_calls=True)
        # Expected: ResponseFormat with punny_response (weather_conditions may be None)
        
    except Exception as e:
        logger.error(f"Error during agent invocation: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")
        print("Please check your Ollama connection and model availability.")
    
    # Example 3: Streaming support with stream_mode="messages" for token-level streaming
    # Based on: https://docs.langchain.com/oss/python/langchain/streaming
    print("\n=== Streaming Example ===")
    print("Streaming response (real-time feedback):")
    try:
        for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": "what's the weather in New York?"}]},
            config=config,
            context=Context(user_id="1"),
            stream_mode="messages"  # Stream LLM tokens as they're generated
        ):
            # Extract node information from metadata
            node = metadata.get("langgraph_node", "unknown")
            
            # Process content_blocks from the token
            if hasattr(token, "content_blocks"):
                content_blocks = token.content_blocks
            elif isinstance(token, dict) and "content_blocks" in token:
                content_blocks = token["content_blocks"]
            else:
                content_blocks = []
            
            # Process each content block
            for block in content_blocks:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    
                    # Handle text tokens - stream incrementally
                    if block_type == "text" and "text" in block:
                        text = block["text"]
                        if text:
                            print(text, end="", flush=True)
                    
                    # Handle tool call chunks
                    elif block_type == "tool_call_chunk":
                        tool_name = block.get("name")
                        tool_args = block.get("args", "")
                        tool_id = block.get("id")
                        
                        # Show tool call info when we have the name
                        if tool_name and tool_id:
                            print(f"\n🔧 Calling tool: {tool_name}\n", flush=True)
                        # Stream tool args as they're generated
                        elif tool_args and isinstance(tool_args, str) and tool_args.strip():
                            # Only print if it's meaningful content (not just partial JSON)
                            pass
                
                # Handle content_blocks as objects (if they have attributes)
                elif hasattr(block, "type"):
                    if block.type == "text" and hasattr(block, "text"):
                        print(block.text, end="", flush=True)
        
        print()  # Final newline
        print("✅ Streaming complete")
        
    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")
        # Fallback to regular invoke if streaming fails
        print("\n⚠️  Falling back to non-streaming mode...")
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": "what's the weather in New York?"}]},
                config=config,
                context=Context(user_id="1")
            )
            extracted = extract_response(response, verbose=False)
            print_response(extracted, show_tool_calls=True)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}", exc_info=True)
            print(f"❌ Fallback error: {str(fallback_error)}")

