#!/bin/bash

# Weather Agent Docker Run Script
# This script loads environment variables from .env and runs Docker commands

set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  Warning: .env file not found!"
    echo "Run './setup_env.sh' to create it, or 'cp .env.example .env' and edit manually"
    exit 1
fi

SCRIPT_NAME=""
AGENT_TYPE=""

# Parse arguments
case "$1" in
    ollama|weather_agent_ollama.py)
        SCRIPT_NAME="weather_agent_ollama.py"
        AGENT_TYPE="ollama"
        CONTAINER_NAME="weather-agent-ollama"
        ;;
    openai|weather_agent.py)
        SCRIPT_NAME="weather_agent.py"
        AGENT_TYPE="openai"
        CONTAINER_NAME="weather-agent-openai"
        ;;
    *)
        echo "Usage: $0 [ollama|openai]"
        echo ""
        echo "Examples:"
        echo "  $0 ollama     # Run weather_agent_ollama.py"
        echo "  $0 openai     # Run weather_agent.py"
        exit 1
        ;;
esac

# Build Docker image
echo "🔨 Building Docker image for $SCRIPT_NAME..."
docker build --build-arg SCRIPT="$SCRIPT_NAME" -t "weather-agent-$AGENT_TYPE:latest" .

# Run Docker container
echo ""
echo "🚀 Running $CONTAINER_NAME..."
echo ""

if [ "$AGENT_TYPE" == "ollama" ]; then
    docker run --rm -it \
        --env-file .env \
        -e OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}" \
        -e OLLAMA_MODEL="${OLLAMA_MODEL:-granite3.1-moe:3b}" \
        -e LANGSMITH_PROJECT="${LANGSMITH_PROJECT_OLLAMA:-${LANGSMITH_PROJECT:-weather-agent-ollama}}" \
        "weather-agent-$AGENT_TYPE:latest"
else
    docker run --rm -it \
        --env-file .env \
        -e LANGSMITH_PROJECT="${LANGSMITH_PROJECT_OPENAI:-${LANGSMITH_PROJECT:-weather-agent-openai}}" \
        "weather-agent-$AGENT_TYPE:latest"
fi

