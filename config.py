"""
Centralized API Key Configuration

This module provides a centralized way to manage all API keys used in the project.
Set your API keys here and import this module in other scripts.
"""

import os
from typing import Optional

# API Key Configuration
# Set your API keys here or use environment variables

# OpenAI API Key
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Anthropic API Key  
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")

# Google Gemini API Key
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")

# DeepSeek API Key
DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")

def setup_api_keys():
    """
    Set up all API keys as environment variables.
    Call this function at the start of your scripts.
    """
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-openai-api-key-here":
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your-anthropic-api-key-here":
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    
    if GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-api-key-here":
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your-deepseek-api-key-here":
        os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

def validate_api_keys():
    """
    Validate that all required API keys are set.
    Returns True if all keys are valid, False otherwise.
    """
    required_keys = [
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    ]
    
    missing_keys = []
    for key_name, key_value in required_keys:
        if not key_value or key_value.startswith("your-"):
            missing_keys.append(key_name)
    
    if missing_keys:
        print(f"⚠️  Warning: Missing or invalid API keys: {', '.join(missing_keys)}")
        print("Please set your API keys in config.py or as environment variables.")
        return False
    
    return True

# Auto-setup when module is imported
setup_api_keys() 