#!/usr/bin/env python3
"""
API Key Setup Script

This script helps you set up your API keys for the medical red-teaming framework.
You can either set them directly in config.py or use environment variables.
"""

import os
import sys
from pathlib import Path

def setup_api_keys_interactive():
    """
    Interactive setup for API keys.
    """
    print("🔑 Medical Red-Teaming Framework - API Key Setup")
    print("=" * 50)
    print()
    print("This script will help you set up your API keys.")
    print("You can either:")
    print("1. Set them directly in config.py")
    print("2. Use environment variables")
    print()
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        setup_config_file()
    elif choice == "2":
        setup_environment_variables()
    else:
        print("Invalid choice. Please run the script again.")
        return

def setup_config_file():
    """
    Guide user to set API keys in config.py
    """
    print("\n📝 Setting up API keys in config.py")
    print("-" * 30)
    print()
    print("Please edit the config.py file and replace the placeholder values:")
    print()
    print("1. OPENAI_API_KEY = 'your-openai-api-key-here'")
    print("2. ANTHROPIC_API_KEY = 'your-anthropic-api-key-here'")
    print("3. GEMINI_API_KEY = 'your-gemini-api-key-here'")
    print("4. DEEPSEEK_API_KEY = 'your-deepseek-api-key-here'")
    print()
    print("Replace 'your-*-api-key-here' with your actual API keys.")
    print()
    print("Example:")
    print("OPENAI_API_KEY = 'sk-proj-abc123...'")
    print()
    
    # Check if config.py exists
    config_path = Path("config.py")
    if config_path.exists():
        print(f"✅ config.py found at: {config_path.absolute()}")
    else:
        print("❌ config.py not found. Please create it first.")

def setup_environment_variables():
    """
    Guide user to set environment variables
    """
    print("\n🌍 Setting up environment variables")
    print("-" * 30)
    print()
    print("You can set environment variables in several ways:")
    print()
    print("1. Export them in your shell:")
    print("   export OPENAI_API_KEY='your-openai-api-key'")
    print("   export ANTHROPIC_API_KEY='your-anthropic-api-key'")
    print("   export GEMINI_API_KEY='your-gemini-api-key'")
    print("   export DEEPSEEK_API_KEY='your-deepseek-api-key'")
    print()
    print("2. Create a .env file in the project root:")
    print("   OPENAI_API_KEY=your-openai-api-key")
    print("   ANTHROPIC_API_KEY=your-anthropic-api-key")
    print("   GEMINI_API_KEY=your-gemini-api-key")
    print("   DEEPSEEK_API_KEY=your-deepseek-api-key")
    print()
    print("3. Set them when running scripts:")
    print("   OPENAI_API_KEY=your-key python your_script.py")

def validate_current_setup():
    """
    Validate the current API key setup
    """
    print("\n🔍 Validating current API key setup")
    print("-" * 30)
    
    # Import config to check current setup
    try:
        from config import validate_api_keys
        if validate_api_keys():
            print("✅ All API keys are properly configured!")
        else:
            print("❌ Some API keys are missing or invalid.")
            print("Please run the setup script to configure them.")
    except ImportError:
        print("❌ config.py not found. Please create it first.")

def main():
    """
    Main function
    """
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_current_setup()
    else:
        setup_api_keys_interactive()

if __name__ == "__main__":
    main() 