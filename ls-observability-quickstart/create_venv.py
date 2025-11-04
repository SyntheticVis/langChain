#!/usr/bin/env python3
"""
Script to recreate virtual environment from scratch
"""
import subprocess
import sys
import os
import shutil

def main():
    print("Creating virtual environment from scratch...")
    print("=" * 50)
    
    venv_path = ".venv"
    
    # Remove existing venv if it exists
    if os.path.exists(venv_path):
        print(f"Removing existing {venv_path} directory...")
        shutil.rmtree(venv_path)
    
    # Create new virtual environment
    print(f"Creating new virtual environment at {venv_path}...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    
    # Determine the correct pip path based on OS
    if sys.platform == "win32":
        pip_path = os.path.join(venv_path, "Scripts", "pip")
        python_path = os.path.join(venv_path, "Scripts", "python")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install dependencies
    print("Installing dependencies (langsmith, openai)...")
    subprocess.run([pip_path, "install", "-U", "langsmith", "openai"], check=True)
    
    print("\n" + "=" * 50)
    print("Virtual environment created successfully!")
    print("\nTo activate it:")
    if sys.platform == "win32":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")
    print("\nTo verify installation:")
    print("  pip list")

if __name__ == "__main__":
    main()


