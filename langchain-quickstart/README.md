# LangChain Quickstart Tutorial

This project demonstrates the LangChain quickstart tutorial, adapted to use **OpenAI** instead of Anthropic.

Based on: [LangChain Quickstart Documentation](https://docs.langchain.com/oss/python/langchain/quickstart)

## Setup

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Set Environment Variables

```bash
source env_exports.sh
```

Or export manually:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="<your-langsmith-api-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
export LANGSMITH_PROJECT=pr-husky-transfer-79
```

### 3. Run the Examples

**Basic Agent Example:**
```bash
python basic_agent.py
```

**Real-World Weather Agent Example:**
```bash
python weather_agent.py
```

## What's Included

### 1. Basic Agent (`basic_agent.py`)
A simple agent that can:
- Answer questions
- Call tools (weather function)
- Uses GPT-4o-mini as the language model

### 2. Real-World Weather Agent (`weather_agent.py`)
A production-ready agent that demonstrates:
- **Detailed system prompts** for better agent behavior
- **Tools** that integrate with external data and runtime context
- **Model configuration** for consistent responses
- **Structured output** for predictable results
- **Conversational memory** for chat-like interactions
- **Full agent creation** with all components

## Key Adaptations from Tutorial

The original tutorial uses Anthropic Claude. This version uses **OpenAI**:

1. **Model**: Changed from `claude-sonnet-4-5-20250929` to `gpt-4o-mini`
2. **Model initialization**: Uses `init_chat_model()` with OpenAI model name
3. **API Key**: Uses `OPENAI_API_KEY` instead of `ANTHROPIC_API_KEY`

## Features Demonstrated

### Basic Agent
- Simple tool definition
- Agent creation with `create_agent()`
- Basic invocation

### Weather Agent
- **System Prompt**: Detailed instructions for agent behavior
- **Tools**: 
  - `get_weather_for_location`: Simple function tool
  - `get_user_location`: Tool with runtime context access
- **Model Configuration**: Temperature, timeout, max_tokens
- **Structured Output**: Dataclass schema for consistent responses
- **Memory**: InMemorySaver for conversation state
- **Context**: Custom context schema (user_id)
- **Multi-turn Conversations**: Using thread_id to maintain conversation

## LangSmith Tracing

All agent runs are automatically traced to LangSmith when `LANGSMITH_TRACING=true` is set. You can view traces in the LangSmith dashboard at [smith.langchain.com](https://smith.langchain.com) under the project `pr-husky-transfer-79`.

## Next Steps

- Explore more tools and integrations
- Add persistent memory (database checkpointer)
- Implement more complex agent workflows
- Add evaluation and testing
- See the [LangChain documentation](https://docs.langchain.com) for more examples

