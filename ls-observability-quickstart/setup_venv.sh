#!/bin/bash

# Script to recreate virtual environment from scratch

echo "Creating virtual environment from scratch..."
echo "==========================================="

# Remove existing venv if it exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv directory..."
    rm -rf .venv
fi

# Create new virtual environment
echo "Creating new virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -U langsmith openai

echo ""
echo "Virtual environment created successfully!"
echo ""
echo "To activate it, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  pip list"


