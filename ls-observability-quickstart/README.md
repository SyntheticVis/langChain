# LangSmith Observability Quickstart

This project demonstrates how to set up LangSmith tracing for a RAG application.

## Setup

### 1. Create Virtual Environment (from scratch)

To recreate the virtual environment from scratch:

```bash
python3 create_venv.py
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -U langsmith openai
```

### 2. Set Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Then edit `.env` with your actual API keys:
- `LANGSMITH_API_KEY`: Get this from [smith.langchain.com](https://smith.langchain.com)
- `OPENAI_API_KEY`: Get this from the OpenAI dashboard

Alternatively, you can export them directly:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="<your-langsmith-api-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
export LANGSMITH_PROJECT=pr-husky-transfer-79
```

Or use the provided script:
```bash
source env_exports.sh
```

### 3. Run the Application

**Option 1: Using the convenience script (recommended)**
```bash
./run_app.sh app.py          # Base app (no tracing)
./run_app.sh app_step4.py    # With LLM tracing
./run_app.sh app_step5.py    # Full application tracing
```

**Option 2: Manual setup**
The application has been set up in stages:

**Step 3: Base Application (no tracing)**
```bash
source .venv/bin/activate
source env_exports.sh
python app.py
```

**Step 4: With LLM Tracing**
```bash
source .venv/bin/activate
source env_exports.sh
python app_step4.py
```

**Step 5: Full Application Tracing**
```bash
source .venv/bin/activate
source env_exports.sh
python app_step5.py
```

**Note:** Make sure to source `env_exports.sh` in each new terminal session to load your environment variables.

## What's Included

- `app.py`: Base application without tracing
- `app_step4.py`: Application with LLM call tracing
- `app_step5.py`: Full application tracing with traceable decorator
- `create_venv.py`: Script to recreate virtual environment from scratch
- `setup_env.sh`: Interactive environment setup script
- `env_exports.sh`: Environment variables (edit with your API keys)
- `run_app.sh`: Convenience script to run apps with environment loaded
