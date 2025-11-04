"""
Basic Agent Example - LangChain Quickstart
Adapted for OpenAI instead of Anthropic
"""

import os
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="gpt-4o-mini",  # Using OpenAI instead of Claude
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    # Print the final AI message
    if response.get("messages"):
        last_message = response["messages"][-1]
        if hasattr(last_message, "content"):
            print(f"Agent Response: {last_message.content}")
        else:
            print(f"Agent Response: {last_message}")

