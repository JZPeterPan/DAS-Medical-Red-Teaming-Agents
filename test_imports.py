#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        # Test core packages
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import pydantic
        print("✅ pydantic imported successfully")
        
        import openai
        print("✅ openai imported successfully")
        
        import anthropic
        print("✅ anthropic imported successfully")
        
        from google import genai
        print("✅ google.genai imported successfully")
        
        from tqdm import tqdm
        print("✅ tqdm imported successfully")
        
        # Test config
        from config import setup_api_keys, validate_api_keys
        print("✅ config module imported successfully")
        
        # Test utils
        from agent_tools.utils import call_agent
        print("✅ agent_tools.utils imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 