#!/bin/bash

# Convenience script to load environment and run the application
# Usage: ./run_app.sh [app.py|app_step4.py|app_step5.py]

cd "$(dirname "$0")"
source .venv/bin/activate
source env_exports.sh

if [ $# -eq 0 ]; then
    echo "Usage: ./run_app.sh [app.py|app_step4.py|app_step5.py]"
    echo ""
    echo "Available apps:"
    echo "  app.py       - Base app (no tracing)"
    echo "  app_step4.py - With LLM tracing"
    echo "  app_step5.py - Full application tracing"
    exit 1
fi

APP_FILE=$1

if [ ! -f "$APP_FILE" ]; then
    echo "Error: $APP_FILE not found"
    exit 1
fi

echo "Running $APP_FILE with environment variables loaded..."
echo "LANGSMITH_PROJECT=$LANGSMITH_PROJECT"
echo ""
python "$APP_FILE"


