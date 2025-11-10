FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (both scripts)
COPY weather_agent.py .
COPY weather_agent_ollama.py .
COPY weather_agent_ollama_model.py .

# Use ARG to specify which script to run (default: weather_agent_ollama.py)
ARG SCRIPT=weather_agent_ollama.py
ENV SCRIPT=${SCRIPT}

# Run the application
CMD ["sh", "-c", "python ${SCRIPT}"]
