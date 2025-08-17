from __future__ import annotations
import pandas as pd
import os
import sys

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized configuration
from config import setup_api_keys, validate_api_keys

# Setup API keys from centralized config
setup_api_keys()

# Validate API keys
if not validate_api_keys():
    print("⚠️  Some API keys are missing. Please check config.py or set environment variables.")
    print("Continuing with available keys...")

# Get API keys from environment (set by config.py)
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")

from google import genai
from google.genai import types
import anthropic
from openai import OpenAI
from agents import Agent, Runner

client_openai = OpenAI()
client_gemini = genai.Client(api_key=gemini_api_key)
client_anthropic = anthropic.Anthropic()
client_deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")


def generate_response(question, model_name, temperature: float = 0):
    instructions = (
        "Answer the question with only the final letter choice (A, B, C, D, E, or F ...) with no explanation. "
        "More than one choice can be correct, in which case return all of them separated by commas."
    )

    additional_instructions = (
        "Answer **ONLY** with the capital letter(s) of the correct choice, "
        "no spaces, no punctuation, no explanation. "
        "If you add anything else you will score 0.\n"
    )

    if model_name.startswith("claude-sonnet") or model_name.startswith("deepseek"): # these models are not good at following instructions
        question = question + "\n" + additional_instructions

    response = call_agent(model_name, user_msg=question, system_msg=instructions, temperature=temperature)
    # # Extract only the answer letter (A-Z)
    # import re
    # match = re.search(r'\b([A-Z])\b', response.upper())
    # final_answer = match.group(1) if match else response

    # if verbose:
        # print("Prompt:\n", prompt)
        # print("Raw Response:\n", response)
        # print("Final Answer:", final_answer)

    return response

# todo: only a temporary solution, need to be replaced
def call_openai_agent(agent_title: str, model: str, user_msg: str, system_msg: str = "Your are a helpful assistant.", image_url=None) -> str:
    # TODO: how to set temperature in agent?
    agent = Agent(
        name=agent_title,
        model=model,
        instructions=system_msg,
    )
    # response = await Runner.run(agent, input=user_msg)
    response = Runner.run_sync(agent, input=user_msg)
    return response.final_output

def call_agent(agent_name: str, user_msg: str, system_msg: str = "Your are a helpful assistant.", temperature=1.0, image_url=None) -> str:
    """
    Generic helper to call one of our "agents" (GPT-4o, etc.)
    based on the user’s environment. Adjust your client call here.
    """
    if agent_name in ["gpt-4o", "o1", "o3", "o4-mini", "o3-mini", "gpt-4.1", "gpt-4.1-mini", "o1-mini", "o1-pro"]:
        return call_gpt_client(agent_name, user_msg, system_msg, temperature, image_url)
    elif agent_name.startswith("gemini"):
        return call_gemini_client(agent_name, user_msg, system_msg, temperature, image_url)
    elif agent_name.startswith("deepseek"):
        return call_deepseek_client(agent_name, user_msg, system_msg, temperature, image_url)
    elif agent_name.startswith("claude"):
        return call_anthropic_client(agent_name, user_msg, system_msg, temperature, image_url)
    else:
        raise ValueError(f"Unsupported agent name: {agent_name}")
    
def call_deepseek_client(agent_name: str, user_msg: str, system_msg: str = "Your are a helpful assistant.", temperature=1, image_url=None) -> str:
    """
    Generic helper to call one of our "agents" (DeepSeek-V3, etc.)
    based on the user's environment. Adjust your client call here.
    """
    if image_url is None:
        input = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
                ]
    else:
        input = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content":
                    [
                    {"type": "input_text", "text": user_msg},
                    {"type": "input_image", "image_url": image_url},
                    ]
                }
            ]

    response = client_deepseek.chat.completions.create(
        model=agent_name,
        messages=input,
        temperature=temperature,
    )
    return response.choices[0].message.content


def call_gpt_client(agent_name: str, user_msg: str, system_msg: str = "Your are a helpful assistant.", temperature=1, image_url=None) -> str:
    """
    Generic helper to call one of our "agents" (GPT-4o, etc.)
    based on the user's environment. Adjust your client call here.
    """
    if image_url is None:
        input = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
                ]
    else:
        input = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content":
                        [
                        {"type": "input_text", "text": user_msg},
                        {"type": "input_image", "image_url": image_url},
                        ],
                    }
                ]

    if agent_name in ['o3-mini', "o4-mini", "o3"]:
        # print(f"[WARNING] 'temperature' is not supported with {agent_name}, using default value.")
        response = client_openai.responses.create(
            model=agent_name,
            input=input
        )
    elif agent_name in ["gpt-4o", "gpt-4.1-mini", "gpt-4.1"]:
        response = client_openai.responses.create(
            model=agent_name,
            input=input,
            temperature=temperature,
        )
    else:
        # todo: sort out the rest of the models
        raise ValueError(f"Unsupported agent name: {agent_name}")

    return response.output_text.strip()

def call_gemini_client(agent_name: str, user_msg: str, system_msg: str = "Your are a helpful assistant.", temperature = 1, image_url=None) -> str:
    """
    Generic helper to call one of our "agents" (Gemini, etc.)
    based on the user's environment. Adjust your client call here.
    """
    response = client_gemini.models.generate_content(
        model=agent_name,
        config=types.GenerateContentConfig(
            system_instruction=system_msg,
            temperature=temperature),
        contents=user_msg,
    )
    return response.text

def call_anthropic_client(agent_name: str, user_msg: str, system_msg: str = "Your are a helpful assistant.", temperature = 1, image_url=None) -> str:
    """
    Generic helper to call one of our "agents" (Gemini, etc.)
    based on the user's environment. Adjust your client call here.
    """
    response = client_anthropic.messages.create(
        model=agent_name,
        max_tokens=5000,
        temperature=temperature,
        system=system_msg,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_msg
                    }
                ]
            }
        ]
    )
    return response.content[0].text


