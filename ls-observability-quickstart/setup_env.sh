#!/bin/bash

# Setup script for LangSmith Observability Quickstart
# This script helps you set up environment variables

echo "LangSmith Observability Quickstart - Environment Setup"
echo "===================================================="
echo ""
echo "Please set the following environment variables:"
echo ""
echo "export LANGSMITH_TRACING=true"
echo "export LANGSMITH_API_KEY=\"<your-langsmith-api-key>\""
echo "export OPENAI_API_KEY=\"<your-openai-api-key>\""
echo "export LANGSMITH_PROJECT=pr-husky-transfer-79"
echo "# Optional: export LANGSMITH_WORKSPACE_ID=\"<your-workspace-id>\""
echo ""
echo "Or run this script with your actual keys:"
echo "./setup_env.sh <your-langsmith-api-key> <your-openai-api-key> [workspace-id]"
echo ""

if [ $# -ge 2 ]; then
    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY="$1"
    export OPENAI_API_KEY="$2"
    export LANGSMITH_PROJECT=pr-husky-transfer-79
    
    if [ $# -ge 3 ]; then
        export LANGSMITH_WORKSPACE_ID="$3"
    fi
    
    echo "Environment variables set!"
    echo ""
    echo "To make them persistent, add them to your ~/.zshrc or ~/.bashrc"
    echo ""
    echo "You can now run:"
    echo "  python app.py          # Base app (no tracing)"
    echo "  python app_step4.py    # With LLM tracing"
    echo "  python app_step5.py    # Full application tracing"
else
    echo "To use this script, provide your API keys:"
    echo "  ./setup_env.sh <langsmith-api-key> <openai-api-key> [workspace-id]"
fi
