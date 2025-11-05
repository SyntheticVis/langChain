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
"""

import os
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# LangSmith tracing is automatically enabled when environment variables are set:
# - LANGSMITH_TRACING=true
# - LANGSMITH_API_KEY=<your-api-key>
# - LANGSMITH_PROJECT=<project-name> (optional but recommended)
# Get your API key from https://smith.langchain.com


# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools that you MUST use when needed:

1. get_user_location: Use this FIRST to get the user's location when they ask about "outside" or "where I am"
2. get_weather_for_location: Use this to get the weather for a specific location (requires a city name)

IMPORTANT: When a user asks about the weather, you MUST:
- First call get_user_location to find where they are
- Then call get_weather_for_location with the location name
- Then provide a punny response about the weather

Always use the tools - don't just talk about using them. Actually call them."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# Run agent
if __name__ == "__main__":
    # Ollama configuration
    # Default to localhost:11434, but can be overridden via environment variable
    # If Ollama is in another Docker container, use the container name or IP
    # Examples:
    #   - Same Docker network: "http://ollama:11434"
    #   - Host machine: "http://host.docker.internal:11434"
    #   - Localhost: "http://localhost:11434"
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "granite3.1-moe:3b")  # Default to llama3.2, can use llama2, mistral, etc.
    
    print(f"Connecting to Ollama at: {ollama_base_url}")
    print(f"Using model: {ollama_model}")
    
    # LangSmith tracing (optional but recommended)
    if os.getenv("LANGSMITH_TRACING") == "true":
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "weather-agent-ollama")
        print(f"✅ LangSmith tracing enabled (project: {langsmith_project})")
        print(f"   View traces at: https://smith.langchain.com")
    else:
        print("ℹ️  LangSmith tracing disabled. Set LANGSMITH_TRACING=true to enable.")
    
    # Configure model - using Ollama (local LLM)
    # Note: Smaller models (like 2b) may have limited tool calling capabilities
    # Consider using larger models (llama3.2:3b, llama3.2:7b, etc.) for better tool calling
    model = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0.1,  # Lower temperature for more deterministic tool calling
        timeout=60,  # Longer timeout for local models
        num_ctx=4096,  # Context window size
    )
    
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
    
    # `thread_id` is a unique identifier for a given conversation.
    config = {"configurable": {"thread_id": "1"}}

    print("=== First Question ===")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
        config=config,
        context=Context(user_id="1")
    )

    # Debug: Print all messages to see if tool calls were made
    if response.get("messages"):
        print("\n--- Message History (for debugging) ---")
        tool_calls_found = False
        for i, msg in enumerate(response["messages"]):
            msg_type = type(msg).__name__
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_found = True
                print(f"Message {i} ({msg_type}): ✅ Tool calls found: {msg.tool_calls}")
            elif hasattr(msg, "content"):
                content_preview = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else msg.content
                print(f"Message {i} ({msg_type}): {content_preview}")
        if not tool_calls_found:
            print("\n⚠️  WARNING: No tool calls were actually made!")
            print("   The model is generating text about calling tools, but not actually calling them.")
            print("   This is common with smaller models (2b parameters).")
            print("   Solution: Use a larger model with better tool calling support:")
            print("   - llama3.2:3b or llama3.2:7b")
            print("   - mistral (7b)")
            print("   - qwen2.5:7b")
        print("--- End Message History ---\n")

    # Handle structured response (may not always be present with Ollama)
    if 'structured_response' in response:
        print(f"Structured Response: {response['structured_response']}")
    else:
        # Fallback: print the last message if structured response not available
        if response.get("messages"):
            last_msg = response["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"Agent Response: {last_msg.content}")
            elif hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"Tool calls made: {last_msg.tool_calls}")
            else:
                print("Response:", response)
        else:
            print("Response:", response)

    print("\n=== Follow-up Question ===")
    # Note that we can continue the conversation using the same `thread_id`.
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]},
        config=config,
        context=Context(user_id="1")
    )

    # Handle structured response (may not always be present with Ollama)
    if 'structured_response' in response:
        print(response['structured_response'])
    else:
        # Fallback: print the last message if structured response not available
        if response.get("messages"):
            last_msg = response["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"Agent Response: {last_msg.content}")
            else:
                print("Response:", response)
        else:
            print("Response:", response)

