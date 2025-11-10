# Code Flow Summary - Quick Reference

## Key Code Sections for State Committing

### 1. Memory Setup (Line 495)
```python
checkpointer = InMemorySaver()  # Creates memory storage
```

### 2. Agent Creation with Memory (Lines 509-516)
```python
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer,  # ← Memory attached here
    middleware=middleware,
)
```

### 3. Config with Thread ID (Line 800)
```python
config = {"configurable": {"thread_id": "1"}}  # Unique conversation ID
```

### 4. User Input Processing (Lines 858-863)
```python
stream_agent_response(
    agent=agent,
    messages=[{"role": "user", "content": user_input}],  # ← User message
    config=config,  # ← Thread ID for memory
    context=Context(user_id="1")
)
```

### 5. Agent Stream Call (Lines 701-706)
```python
for token, metadata in agent.stream(
    {"messages": messages},  # ← Includes user message
    config=config,           # ← Thread ID loads previous state
    context=context,
    stream_mode="messages"
):
```

### 6. State Commit (AUTOMATIC - Inside LangGraph)
```python
# This happens automatically inside agent.stream():
# 1. Load previous state:
checkpoint = checkpointer.get(config)  # Loads state for thread_id="1"

# 2. Process with full history:
# - Previous messages + new user message
# - Agent processes
# - Tools execute (if needed)
# - Response generated

# 3. Save new state:
checkpointer.put(config, new_checkpoint)  # ← COMMIT HAPPENS HERE
```

---

## State Commit Flow (Simplified)

```
User Input
    ↓
agent.stream() called
    ↓
checkpointer.get(config)  ← Load previous state
    ↓
Append user message to history
    ↓
Agent processes (LLM + tools)
    ↓
checkpointer.put(config, state)  ← COMMIT (automatic)
    ↓
Stream response to user
```

---

## Where State is Saved

**Location**: Inside LangGraph's `agent.stream()` method (not visible in this code)

**When**: After each agent step completes

**What**: Complete message history including:
- User messages
- Agent messages
- Tool calls
- Tool results

**How**: 
- Key: `thread_id` from config
- Storage: InMemorySaver (Python memory)
- Format: Checkpoint object with channel_values

---

## Key Functions

| Function | Purpose | State Commit? |
|----------|---------|---------------|
| `stream_agent_response()` | Streams agent response | No (calls agent.stream) |
| `agent.stream()` | Executes agent workflow | **Yes (automatic)** |
| `checkpointer.get()` | Loads previous state | No (read only) |
| `checkpointer.put()` | Saves state | **Yes (automatic)** |
| `after_tool()` | Caches tool results | No (updates state) |

---

## Memory Access Points

1. **Before Processing**: `checkpointer.get(config)` - Loads previous state
2. **During Processing**: Middleware can access state via `state["messages"]`
3. **After Processing**: `checkpointer.put(config, state)` - Saves new state

---

## Important Notes

- ✅ **No explicit commit call** - LangGraph handles it automatically
- ✅ **State persists** - Same `thread_id` = same conversation
- ✅ **Incremental saves** - State saved after each step
- ✅ **Complete history** - All messages are preserved
- ⚠️ **InMemorySaver** - Lost when program exits (not persistent to disk)

