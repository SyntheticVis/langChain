#!/bin/bash

# Weather Agent Environment Setup Script
# This script helps you set up your .env file from .env.example

set -e

ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

echo "=========================================="
echo "Weather Agent Environment Setup"
echo "=========================================="
echo ""

# Check if .env.example exists
if [ ! -f "$ENV_EXAMPLE" ]; then
    echo "❌ Error: $ENV_EXAMPLE not found!"
    exit 1
fi

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  Warning: $ENV_FILE already exists!"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Keeping existing $ENV_FILE"
        exit 0
    fi
fi

# Copy .env.example to .env
cp "$ENV_EXAMPLE" "$ENV_FILE"
echo "✅ Created $ENV_FILE from $ENV_EXAMPLE"
echo ""

# Prompt for values
echo "Please fill in your environment variables:"
echo ""

# LangSmith
read -p "Enable LangSmith tracing? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sed -i.bak 's/LANGSMITH_TRACING=false/LANGSMITH_TRACING=true/' "$ENV_FILE"
    read -p "Enter your LangSmith API key: " langsmith_key
    if [ ! -z "$langsmith_key" ]; then
        sed -i.bak "s|LANGSMITH_API_KEY=.*|LANGSMITH_API_KEY=$langsmith_key|" "$ENV_FILE"
    fi
    read -p "Enter LangSmith project name (default: weather-agent): " langsmith_project
    if [ ! -z "$langsmith_project" ]; then
        sed -i.bak "s|LANGSMITH_PROJECT=.*|LANGSMITH_PROJECT=$langsmith_project|" "$ENV_FILE"
    fi
else
    sed -i.bak 's/LANGSMITH_TRACING=true/LANGSMITH_TRACING=false/' "$ENV_FILE"
fi

# OpenAI
read -p "Enter your OpenAI API key (for weather_agent.py): " openai_key
if [ ! -z "$openai_key" ]; then
    sed -i.bak "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$openai_key|" "$ENV_FILE"
fi

# Ollama
read -p "Enter Ollama base URL (default: http://host.docker.internal:11434): " ollama_url
if [ ! -z "$ollama_url" ]; then
    sed -i.bak "s|OLLAMA_BASE_URL=.*|OLLAMA_BASE_URL=$ollama_url|" "$ENV_FILE"
fi

read -p "Enter Ollama model (default: granite3.1-moe:3b): " ollama_model
if [ ! -z "$ollama_model" ]; then
    sed -i.bak "s|OLLAMA_MODEL=.*|OLLAMA_MODEL=$ollama_model|" "$ENV_FILE"
fi

# Clean up backup file
rm -f "$ENV_FILE.bak"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "Your .env file has been created. You can now:"
echo "  - Run: docker-compose up weather-agent-ollama --build"
echo "  - Run: docker-compose up weather-agent-openai --build"
echo ""
echo "To edit your .env file manually:"
echo "  nano .env"
echo "  # or"
echo "  vim .env"
echo ""

