"""
Medical Fact Checking Pipeline with Structured Output Types (v5)

This module implements a sophisticated medical fact checking pipeline using specialized AI agents
to detect and classify various types of medical misinformation, hallucinations, and factual errors
in LLM responses. The system uses a multi-agent approach with specialized agents for different
aspects of medical fact checking.

Key Components:
- BaseAgentOutput: Base Pydantic model for structured agent outputs
- Specialist Agents: Individual agents for specific fact checking tasks
- Orchestrator: Coordinates specialist agents and synthesizes results

The pipeline supports:
- Structured output validation using Pydantic models
- Multiple specialist agents with specific expertise
- Web search integration for fact verification
- Configurable processing parameters
- Processing specific ranges of rows using start and end indices
- JSON-only input/output processing
"""

from __future__ import annotations

import os
import argparse
import logging
import json
from pathlib import Path
from typing import (
    List, Union, Literal, Optional, Dict, Any, Tuple,
    TypeVar, Generic, Sequence, Callable
)
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
from pydantic import BaseModel, Field, validator
from agents import Agent, Runner, WebSearchTool, ModelSettings
from agent_prompts_o1pro_v1 import (
    medfact_prompt, medfact_description,
    citation_prompt, citation_description,
    reasoning_prompt, reasoning_description,
    context_prompt, context_description,
    safety_prompt, safety_description,
    instruction_prompt, instruction_description,
    hallucination_prompt, hallucination_description,
    orchestrator_prompt
)
from agent_outputs_o1pro import (
    BaseAgentOutput, MedFactOutput, CitationOutput, ReasoningOutput,
    ContextOutput, SafetyOutput, InstructionOutput, HallucinationOutput,
    OrchestratorOutput, ClassificationCode,
    SPECIALIST_REASONING_MAX_WORDS, ORCHESTRATOR_RATIONALE_MAX_WORDS,
    AGENT_DECISION_REASONING_MAX_WORDS
)
from collections import Counter

# Type aliases for better readability
AgentOutput = TypeVar('AgentOutput', bound='BaseAgentOutput')
AgentResponse = Tuple[str, str]  # (classification, reasoning)

# Import centralized configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import setup_api_keys, validate_api_keys

# Setup API keys from centralized config
setup_api_keys()

# Validate API keys
if not validate_api_keys():
    print("⚠️  Some API keys are missing. Please check config.py or set environment variables.")
    print("Continuing with available keys...")

# Constants
DEFAULT_MODEL = "gpt-4o"
SUPPORTED_MODELS = ["gpt-4o", "o3", "o4-mini"]
SEARCH_AGENT_MODEL = "o3"  # Model to use for agents that need web search

def parse_response(output: Union[BaseAgentOutput, OrchestratorOutput]) -> AgentResponse:
    """
    Extract classification and reasoning from structured output.
    
    Handles both specialist agent outputs and orchestrator outputs,
    converting them to a standardized (classification, reasoning) tuple.
    
    Args:
        output: Structured output from any agent
        
    Returns:
        AgentResponse: Tuple of (classification, reasoning)
        
    Raises:
        ValueError: If output format is invalid
    """
    if isinstance(output, OrchestratorOutput):
        return str(output.merged_codes), output.rationale
    if isinstance(output, BaseAgentOutput):
        return str(output.classification), output.reasoning
    raise ValueError(f"Invalid output type: {type(output)}")


def get_model_settings(model: str) -> ModelSettings:
    """
    Get appropriate model settings based on the model type.
    
    Args:
        model: The model name to get settings for
        
    Returns:
        ModelSettings: Configured model settings
    """
    if model == "gpt-4o":
        return ModelSettings(tool_choice="auto", temperature=0.0)
    elif model == "o4-mini":
        return ModelSettings(tool_choice="auto")
    return ModelSettings(tool_choice="auto")


def create_orchestrator(
    orchestrator_model: str = DEFAULT_MODEL,
    sub_agent_model: str = DEFAULT_MODEL
) -> Agent:
    """
    Create and configure the orchestrator agent.
    
    Args:
        orchestrator_model: AI model to use for the orchestrator
        sub_agent_model: AI model to use for all sub-agents
        
    Returns:
        Agent: Configured orchestrator agent
        
    Raises:
        ValueError: If model is not supported
    """
    if orchestrator_model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported orchestrator model: {orchestrator_model}")
    if sub_agent_model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported sub-agent model: {sub_agent_model}")
    
    # Create specialist agents
    agents = [
        make_agent(
            "MedFactChecker",
            medfact_prompt,
            medfact_description,
            MedFactOutput,
            uses_search=True,
            model=sub_agent_model
        ),
        make_agent(
            "CitationVerifier",
            citation_prompt,
            citation_description,
            CitationOutput,
            uses_search=True,
            model=sub_agent_model
        ),
        make_agent(
            "ReasoningAuditor",
            reasoning_prompt,
            reasoning_description,
            ReasoningOutput,
            model=sub_agent_model
        ),
        make_agent(
            "ContextKeeper",
            context_prompt,
            context_description,
            ContextOutput,
            model=sub_agent_model
        ),
        make_agent(
            "SafetyGuardian",
            safety_prompt,
            safety_description,
            SafetyOutput,
            model=sub_agent_model
        ),
        make_agent(
            "InstructionWatcher",
            instruction_prompt,
            instruction_description,
            InstructionOutput,
            model=sub_agent_model
        ),
        make_agent(
            "HallucinationScout",
            hallucination_prompt,
            hallucination_description,
            HallucinationOutput,
            model=sub_agent_model
        )
    ]
    
    return Agent(
        name="MedFact Orchestrator",
        instructions=orchestrator_prompt.strip(),
        tools=agents,
        model=orchestrator_model,
        model_settings=get_model_settings(orchestrator_model),
        output_type=OrchestratorOutput
    )


def setup_logging(log_path: Path) -> None:
    """
    Configure logging for the pipeline.
    
    Sets up both file and console logging with appropriate
    formatting and log levels.
    
    Args:
        log_path: Path to the log file
    """
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("MedFactCheck pipeline start")


def make_agent(
    name: str,
    prompt: str,
    tool_description: str,
    output_type: type[BaseAgentOutput],
    uses_search: bool = False,
    model: str = DEFAULT_MODEL
) -> Agent:
    """
    Create a specialist agent with the given configuration.
    
    Args:
        name: Name of the agent
        prompt: System prompt for the agent
        tool_description: Description of the agent's capabilities
        output_type: Pydantic model class for structured output
        uses_search: Whether the agent should have access to web search
        model: AI model to use (default: DEFAULT_MODEL)
        
    Returns:
        Agent: Configured specialist agent
        
    Raises:
        ValueError: If model is not supported
    """
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")
        
    # Use SEARCH_AGENT_MODEL for agents that need web search
    agent_model = SEARCH_AGENT_MODEL if uses_search else model
    
    # Skip web search for o3 model
    search_tool = None
    if uses_search and agent_model != "o3":
        search_tool = WebSearchTool(search_context_size="medium")
    
    # Create model settings with appropriate parameters for each model
    if agent_model == "o3":
        model_settings = ModelSettings(
            tool_choice="required" if search_tool else "auto"
        )
    else:
        model_settings = ModelSettings(
            tool_choice="required" if search_tool else "auto",
            temperature=0.0
        )
    
    return Agent(
        name=name,
        instructions=prompt.strip(),
        model=agent_model,
        tools=[search_tool] if search_tool else [],
        model_settings=model_settings,
        output_type=output_type,
    ).as_tool(
        tool_name=name,
        tool_description=tool_description.strip(),
    )


def process_row_json(row: dict, orchestrator: Agent) -> dict:
    """
    Process a single input dict through the orchestrator, return output dict (including input fields).
    Skips the row if parse_model_response or Runner.run_sync raises an error.
    """
    prompt = row.get("Prompt", row.get("prompt", ""))
    response = row.get("Response", row.get("response", ""))
    row_idx = row.get("row_idx", "unknown")
    
    try:
        payload = f"<user>{prompt.strip()}</user>\n<llm>{response}</llm>"
        result = Runner.run_sync(orchestrator, payload)
        output = result.final_output
        pred_cls, reasoning = parse_response(output)
        
        # Print detailed output including agent decisions
        print(f"\nProcessing row {row_idx}:")
        print(str(output))
        print("-" * 80)
        
        # Compose output dict: include all input fields, plus orchestrator output fields
        out = dict(row)
        out.update({
            "merged_codes": getattr(output, "merged_codes", None),
            "rationale": getattr(output, "rationale", None),
            "agent_decisions": [
                {
                    "code": d.code,
                    "called": d.called,
                    "reasoning": d.reasoning,
                    **({"classification": d.classification, "cls_reasoning": d.cls_reasoning} if d.called else {})
                } for d in getattr(output, "agent_decisions", [])
            ]
        })
        return out
    except Exception as e:
        # Log and skip this row
        raise RuntimeError(f"Failed to process input: {e}")


def main() -> None:
    """Main entry point for the medical fact checking pipeline (JSON-only mode)."""
    parser = argparse.ArgumentParser(
        description="Medical Fact Checking Pipeline (JSON-only mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to the input JSON file (with prompt/response pairs)"
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to the output JSON file (detailed_outputs.json)"
    )
    parser.add_argument(
        "--orchestrator-model",
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help="OpenAI model to use for the orchestrator agent"
    )
    parser.add_argument(
        "--sub-agent-model",
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help="OpenAI model to use for all sub-agents"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        required=True,
        help="Starting row index to process (inclusive)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        required=True,
        help="Ending row index to process (inclusive)"
    )
    parser.add_argument(
        "--ignore-exist",
        action="store_true",
        help="Ignore existing results and overwrite them with new results"
    )
    parser.add_argument(
        "--log",
        default="pipeline.log",
        help="Path to the log file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(Path(args.log))

    # Load input JSON
    with open(args.input_json, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    if isinstance(json_data, dict) and "results" in json_data:
        input_rows = json_data["results"]
    elif isinstance(json_data, list):
        input_rows = json_data
    else:
        raise ValueError("Unrecognized JSON format for input.")

    # Load existing output JSON if present
    output_path = Path(args.output_json)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing_outputs = json.load(f)
        # Build set of unique keys for fast lookup
        existing_keys = {row.get("row_idx"): row for row in existing_outputs}
    else:
        existing_outputs = []
        existing_keys = {}

    # Only process the specified range
    start_idx = args.start_idx
    end_idx = args.end_idx
    input_rows = input_rows[start_idx:end_idx+1]

    # Create orchestrator
    orchestrator = create_orchestrator(
        orchestrator_model=args.orchestrator_model,
        sub_agent_model=args.sub_agent_model
    )

    # Process all input rows
    new_outputs = []
    for i, row in enumerate(input_rows, start=start_idx):
        key = row.get("row_idx", i)
        if not args.ignore_exist and key in existing_keys:
            logging.info(f"Skipping already processed input (row_idx: {key})")
            continue
        try:
            out = process_row_json(row, orchestrator)
            new_outputs.append(out)
            
            # Log detailed processing output
            logging.info(f"Processing row {i}:")
            logging.info(f"  Classification: {out.get('merged_codes', 'N/A')}")
            logging.info(f"  Rationale: {out.get('rationale', 'N/A')}")
            logging.info(f"  Agent Decisions:")
            for decision in out.get('agent_decisions', []):
                code = decision.get('code', 'N/A')
                called = decision.get('called', False)
                reasoning = decision.get('reasoning', 'N/A')
                logging.info(f"    Code {code}: {'Called' if called else 'Not called'} - {reasoning}")
                if called:
                    classification = decision.get('classification', 'N/A')
                    cls_reasoning = decision.get('cls_reasoning', 'N/A')
                    logging.info(f"      Output: {classification} - {cls_reasoning}")
            
            logging.info(f"Processed input (row_idx: {key})")
        except Exception as e:
            logging.error(f"Error processing input (row_idx: {key}): {e}")

    # Merge new outputs with existing (overwrite if ignore_exist)
    if args.ignore_exist:
        # Remove any existing outputs with the same key as new outputs
        new_keys = {row.get("row_idx") for row in new_outputs}
        merged_outputs = [row for row in existing_outputs if row.get("row_idx") not in new_keys] + new_outputs
    else:
        merged_outputs = existing_outputs + new_outputs

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_outputs, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved {len(merged_outputs)} results to {output_path}")

    # Print response quality stats
    print_response_quality_stats(output_path)


def print_response_quality_stats(output_json_path: Path):
    """Print statistics about the response quality from the JSON output file."""
    with open(output_json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    total = len(results)
    if total == 0:
        print("No results to analyze.")
        return
    
    # Extract ground truth and predictions
    ground_truth = []
    predictions = []
    
    for r in results:
        # Get ground truth from Hallucination/Accuracy field
        gt = r.get("Hallucination/Accuracy", None)
        if isinstance(gt, str):
            gt = float(gt) if gt.replace('.', '').isdigit() else 0
        ground_truth.append(gt)
        
        # Get prediction from merged_codes
        merged_codes = r.get("merged_codes", "0")
        if isinstance(merged_codes, list):
            merged_codes = merged_codes[0] if merged_codes else "0"
        # Convert to binary: 0 = no problem, anything else = problem
        pred = 0 if merged_codes == "0" else 1
        predictions.append(pred)
    
    # Calculate confusion matrix
    true_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
    false_positives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
    true_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0)
    false_negatives = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n=== Classification Statistics ===")
    print(f"Total responses: {total}")
    print(f"Ground truth positives: {sum(ground_truth)}")
    print(f"Ground truth negatives: {total - sum(ground_truth)}")
    print(f"Predicted positives: {sum(predictions)}")
    print(f"Predicted negatives: {total - sum(predictions)}")
    
    print(f"\nConfusion Matrix:")
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives:  {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    
    print(f"\nPerformance Metrics:")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1_score:.2%}")
    
    # Count responses with any problem (merged_codes != '0')
    def is_problem(merged_codes):
        if isinstance(merged_codes, list):
            return any(code != "0" for code in merged_codes)
        return merged_codes != "0"
    responses_with_problem = sum(is_problem(r.get("merged_codes", "0")) for r in results)
    print(f"\n=== Response Quality Metrics ===")
    print(f"Responses detected with any problem: {responses_with_problem} ({responses_with_problem/total*100:.2f}%)")
    
    # Per-code stats
    code_counter = Counter()
    for r in results:
        codes = r.get("merged_codes", [])
        if isinstance(codes, str):
            codes = codes.split(",") if codes else []
        for code in set(codes):
            if code in {str(i) for i in range(1,8)}:
                code_counter[code] += 1
    print("\nSub-Code Problem Detection Percentage:")
    for code in range(1,8):
        count = code_counter[str(code)]
        percent = count / total * 100
        print(f"  Code-{code}: {count} responses ({percent:.2f}%)")
    
    # Agent call distribution
    agent_calls = Counter()
    for r in results:
        for d in r.get("agent_decisions", []):
            if d.get("called"):
                agent_calls[str(d.get("code"))] += 1
    print("\nAgent Call Distribution:")
    for code in range(1,8):
        calls = agent_calls[str(code)]
        call_rate = calls / total * 100
        print(f"  Agent {code}: {calls} calls ({call_rate:.1f}%)")


if __name__ == "__main__":
    main() 