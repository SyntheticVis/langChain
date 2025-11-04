#!/bin/bash

# Convenience script to load environment and run examples
# Usage: ./run_example.sh [basic_agent.py|weather_agent.py]

cd "$(dirname "$0")"
source .venv/bin/activate
source env_exports.sh

if [ $# -eq 0 ]; then
    echo "Usage: ./run_example.sh [basic_agent.py|weather_agent.py]"
    echo ""
    echo "Available examples:"
    echo "  basic_agent.py    - Simple agent with weather tool"
    echo "  weather_agent.py  - Full weather agent with memory and structured output"
    exit 1
fi

EXAMPLE_FILE=$1

if [ ! -f "$EXAMPLE_FILE" ]; then
    echo "Error: $EXAMPLE_FILE not found"
    exit 1
fi

echo "Running $EXAMPLE_FILE with environment variables loaded..."
echo "LANGSMITH_PROJECT=$LANGSMITH_PROJECT"
echo ""
python "$EXAMPLE_FILE"

