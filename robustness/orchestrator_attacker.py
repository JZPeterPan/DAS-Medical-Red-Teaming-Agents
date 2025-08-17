from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, List
import time
import random

# -----------------------------------------------------------------------------
#  Project-internal imports (add project root to sys.path)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agent_tools.utils import generate_response  
from utils_general import load_data, create_medqa_question_dict  
from agents import Agent, Runner, TResponseInputItem, trace, function_tool
from agent_tools.openai_agent_decorators import (
    ManipulationOutput,
    QuestionData,
    generate_distractor_options_tool,
    replace_correct_answer_to_none_of_the_options_are_correct_tool,
    invert_final_question_and_answer_tool,
    add_distraction_sentence_tool,
    adjust_impossible_measurement_tool,
    introduce_cognitive_bias_tool,
    TOOLS,
    CONFLICT_SETS
)


# -----------------------------------------------------------------------------
#  1. CLI arguments
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="../data/data_clean/questions/US",
                        help="Directory with MedQA JSON files (see load_data).")
    parser.add_argument("--further_test_samples", type=int, default=5,
                        help="Process this many items that the target model \n"
                             "answers correctly on the first try – useful for \n"
                             "quick experiments.")
    parser.add_argument("--additional_choices_num", "-a", type=int, default=4,
                        help="Extra options to add when using \n"
                             "generate_distractor_options.")
    parser.add_argument("--generation_model", type=str, default="o3",
                        help="Model used *inside* manipulation tools (e.g. GPT-4o).")
    parser.add_argument("--generation_temperature", type=float, default=0.2)
    parser.add_argument("--target_model", type=str, default="o3-mini",
                        help="Model that answers the questions.  We try to fool it.")
    parser.add_argument("--target_temperature", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=5,
                        help="Maximum manipulation iterations per sample.")
    return parser.parse_args()


args = parse_args()

import uuid

# a random run id
run_group_id = random.randint(1000000, 9999999)

# -----------------------------------------------------------------------------
#  2. Catalogue of tools available to the agent
# -----------------------------------------------------------------------------


TOOL_DESCRIPTIONS = {
    "generate_distractor_options_tool": (
        "Append new wrong answer options that look plausible."),
    "replace_correct_answer_to_none_of_the_options_are_correct_tool": (
        "Replace the keyed answer so that 'None of the options are correct' "
        "becomes the correct choice."),
}

# -----------------------------------------------------------------------------
#  3. Manipulator agent definition
# -----------------------------------------------------------------------------

system_prompt_single_tool_strategy = (
    "You are a robustness engineer specializing in medical question answering (QA).\n"
    "In each round, your task is to select and apply **exactly one** manipulation tool from the available list that is most likely to cause the **target model** to answer incorrectly.\n"
    "\n"
    "**Key Requirements:**\n"
    "- You must always apply your chosen manipulation tool to the **original question sample** provided at the very beginning of the conversation, regardless of previous manipulations.\n"
    "- Carefully review the history of previous attempts and avoid reusing any tool that has already failed to mislead the target model.\n"
    "\n"
    "**Process:**\n"
    "1. Analyze the original question sample and the history of manipulations and their outcomes.\n"
    "2. Select the single tool that you believe has the highest chance of fooling the target model, ensuring it has not already failed.\n"
    "3. Apply the chosen tool to the original question sample to generate a manipulated version.\n"
    "4. Provide a concise rationale (no more than 2 sentences) explaining your choice and how you expect this manipulation to mislead the target model.\n"
    "\n"
    "**Output Requirements:**\n"
    "Return your response as a single, valid JSON object strictly following this schema:\n"
    "{\n"
    '  "manipulation_tools": ["tool_name"],  // The name of the single tool you selected\n'
    '  "reason": "A concise explanation of your tool choice and strategy (max 2 sentences)",\n'
    '  "manipulated_question": {\n'
    '    "question": "Question text",\n'
    '    "options": {"A": "Option A", "B": "Option B", ...},\n'
    '    "answer": "Correct answer text",\n'
    '    "answer_idx": "Correct answer index",\n'
    '    "meta_info": "Meta information"\n'
    "  }\n"
    "}\n"
    "\n"
    "**Critical Instructions:**\n"
    "- The `manipulated_question` (including its text, options, answer, and answer_idx) MUST always be derived from the **original question sample** provided at the start of the conversation, not from any previously manipulated version.\n"
    "- Ensure the output JSON is perfectly formatted and contains all required fields as specified above.\n"
)

# todo: add multiple tool strategy, don't forget to include conflict sets
# system_prompt_multiple_tool_strategy = ()



system_prompt_progressive_tool_strategy = (
    "You are an expert robustness engineer specializing in medical question answering (QA).\n"
"Your primary goal in each round is to strategically manipulate the **ORIGINAL QUESTION SAMPLE** at the very beginning of the whole conversation to create a new version that is *highly likely* to cause a *target AI model* to answer incorrectly.\n\n"
    
    "**Context and History:**\n"
    "Carefully review the conversation history. Pay close attention to tool combinations that have been previously attempted, especially those that failed to mislead the target model. Avoid repeating failed strategies.\n\n"
    
    "**Tool Selection Process:**\n"
    "1.  **Choose Tools:** Select a subset of one or more tools from the available list:\n"
    f"    {TOOLS}\n\n"
    "2. When selecting tools, you must avoid improper tool usage. Some examples:\n"
    "    - If the question body does not contain any measurable values (such as vital signs, lab values, medication dosages, physical measurements, etc.), you can NOT use 'adjust_impossible_measurement_tool'.\n"
    "    - If the question is purely descriptive (e.g., 'How does radiation affect cancer cells?'), or if the complement of the original correct answer cannot be guaranteed to be clearly 'wrong', you can NOT use 'invert_final_question_and_answer_tool'.\n"
    "3.  **Avoid Conflicts:** Do NOT select any pair of tools that are known to conflict. Conflicting pairs are:\n"
    f"    {CONFLICT_SETS}\n\n"
    "4.  **Strategic Application:** Your tool selection should maximize the chance of fooling the target model.\n"
    "    *   **Initial Rounds (1-3):** Prioritize using a minimal number of tools (preferably 1 or 2).\n"
    "    *   **Later Rounds (After Round 3):** If simpler combinations (1-2 tools) have proven ineffective in earlier rounds, you may employ larger combinations of tools.\n"
    "    *   **Key Principle:** Always avoid reusing specific tool combinations that have already failed to make the model answer incorrectly.\n\n"
    "5.  **Failure Handling:** If your selected tool(s) fail to generate a valid manipulation, select a different tool or combination and try again.\n\n"
    
    "**Output Requirements:**\n"
    "Return your response as a single, valid JSON object. The JSON must strictly adhere to the following schema:\n"
    "{\n"
    '  "manipulation_tools": ["tool_name_1", "tool_name_2", ...],  // List of chosen tool names\n'
    '  "reason": "A concise explanation (within 2 sentences) of why you chose these specific tools and your strategy for how this manipulation is expected to mislead the target model.",\n'
    '  "manipulated_question": {\n'
    '     "question": "...",\n'
    '     "options": {"A":"...", "B":"...", ...},\n'
    '     "answer": "...",\n'
    '     "answer_idx": "...",\n'
    '     "meta_info": "..." \n'
    "  }\n"
    "}\n"
    "\n"
    "**IMPORTANT OUTPUT FORMAT RULE:**\n"
    "Return **only** a single, standalone JSON object – *no* code‑blocks, "
    "no prefixes such as `json_output`)`, and no trailing text. The very first character must be '{' and the very last must be '}'.\n"
    "**Critical Instructions:**\n"
    "- The `manipulated_question` (including its text, options, answer, and answer_idx) MUST be derived from the **ORIGINAL QUESTION SAMPLE** provided at the very beginning of the conversation.\n"
    "- Ensure the output JSON is perfectly formatted and contains all the required fields as specified in the schema.\n"
    "- Ensure that the final manipulated question remains a valid medical question that still has well-defined correct answer(s) and contains all necessary context for a medical professional to answer.\n"
)


manipulator_agent = Agent[None](  # type: ignore[type-arg]
    name="question_manipulator",
    model=args.generation_model,
    instructions=system_prompt_progressive_tool_strategy,
    output_type=ManipulationOutput,
    tools=[
        generate_distractor_options_tool,
        replace_correct_answer_to_none_of_the_options_are_correct_tool,
        invert_final_question_and_answer_tool,
        add_distraction_sentence_tool,
        adjust_impossible_measurement_tool,
        introduce_cognitive_bias_tool,
    ],
)

# -----------------------------------------------------------------------------
#  4. Async main loop
# -----------------------------------------------------------------------------

async def main() -> None:
    # -- load data ----------------------------------------------------------------
    test_qa, _ = load_data(args.data_dir)

    GEN_MODEL = args.generation_model
    GEN_TEMP = args.generation_temperature
    TGT_MODEL = args.target_model
    TGT_TEMP = args.target_temperature
    MAX_ITER = args.max_iter

    total_seen = initially_correct = fooled = 0
    results = []

    # -------------------------------------------------------------------------
    for idx, sample in enumerate(test_qa, start=1):
        total_seen += 1
        question_str = create_medqa_question_dict(sample)

        # Ask target model before any manipulation --------------------------------
        first_resp = generate_response(question_str, TGT_MODEL, TGT_TEMP)
        if first_resp is None:
            print(f"[INFO] Model initial output is None, continuing...")
            continue

        first_resp_clean = re.sub(r"\s+", "", first_resp)

        if first_resp_clean != sample["answer_idx"]:
            print(f"Model response is incorrect: {first_resp_clean} != Ground Truth: {sample['answer_idx']}\n")
            print(f"[{idx}] Target model already wrong – skipping sample.")
            continue

        initially_correct += 1
        print(f"[{idx}] Baseline correct. Entering manipulation loop…")

        current_sample = QuestionData(**sample)
        input_items: list[TResponseInputItem] = []

        fooled_this_item = False

        input_items.append({"role": "user", "content": f"**Original Question SAMPLE**: {current_sample.model_dump_json(indent=2)}"})

        with trace(f"Iterative Manipulation Attack -- QA ID: {idx} -- Job Run ID: {run_group_id}"):
            start_time = time.time()
            for iteration in range(1, MAX_ITER + 1):
                # ---- 4a. Ask manipulator agent which tool to use -------------------
                user_message = (
                    f"Round {iteration}.\n"
                    f"In the previous round, the target model still answered correctly. "
                    f"Please select and apply the next most effective manipulation on the **Original Question SAMPLE** to try to make the model answer incorrectly.\n"
                    f"Clearly state which manipulation you chose, your reasoning, and provide the updated question JSON after applying the manipulation."
                )
                # todo: if the model still do manipulation based on the current question, we attach the original question to the user message here as well
                
                input_items.append({"role": "user", "content": user_message})
                
                manipulation_result = await Runner.run(manipulator_agent, input_items)

                result: ManipulationOutput = manipulation_result.final_output
                reason = result.reason
                tool_names = result.manipulation_tools
                manipulated_sample = result.manipulated_question
                input_items = manipulation_result.to_input_list()
                print(f"    • Applying {tool_names}: {reason}\n{manipulated_sample.model_dump_json(indent=2)}")


                # If no change, abort this sample -----------------------------------
                if manipulated_sample == current_sample:
                    print("      (Tool produced no changes – aborting this sample.)")
                    break
                current_sample = manipulated_sample

                # ---- 4c. Query target model on manipulated question ---------------
                manipulated_q_str = create_medqa_question_dict(current_sample.model_dump())
                resp = generate_response(manipulated_q_str, model_name=TGT_MODEL, temperature=TGT_TEMP)

                if resp is None:
                    print(f"[INFO] Model output is None, continuing...")
                    continue
                resp_clean = re.sub(r"\s+", "", resp)

                if resp_clean != current_sample.answer_idx:
                    fooled += 1
                    fooled_this_item = True
                    print(f"      🎉 Model fooled after {iteration} iteration(s). The response is: {resp}.")
                    break
                elif iteration == MAX_ITER:
                    print("      Reached max_iter without fooling the model.")
                else:
                    print("      Model still correct; continuing…")

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time taken: {total_time:.2f} seconds")

            # ---- 4d. Store per-sample results --------------------------------------
            results.append({
                "sample_number": idx,
                "fooled": fooled_this_item,
                "iterations": iteration if fooled_this_item else MAX_ITER,
                "tool_used": tool_names,
                "tool_reason": reason,
                "final_question": manipulated_q_str,
                "total_time": total_time,
            })

            if initially_correct >= args.further_test_samples:
                print("[INFO] Reached further_test_samples limit – stopping early.")
                break

    # -------------------------------------------------------------------------
    # 5. Summary & logging
    # -------------------------------------------------------------------------
    # Calculate the overall average iterations

    first_round_accuracy = initially_correct / total_seen 
    success_rate = fooled / initially_correct if initially_correct else 0.0
    print("\n=== SUMMARY ===")
    print(f"Total processed            : {total_seen}")
    print(f"Initially answered correct : {initially_correct}")
    print(f"Fooled after manipulations : {fooled}")
    print(f"Success rate               : {success_rate:.2%}")

    avg_iterations = sum(r["iterations"] for r in results) / len(results)
    print(f"Overall average iterations: {avg_iterations:.2f}")

    avg_time = sum(r["total_time"] for r in results) / len(results)
    print(f"Overall average time: {avg_time:.2f} seconds")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", args.target_model)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"orchestrator_{args.target_model}_{timestamp}.json")

    with open(log_path, "w") as fh:
        json.dump({
            "job_run_id": run_group_id,
            "initially_correct": initially_correct,
            "first_round_accuracy": first_round_accuracy,
            "fooled": fooled,
            "success_rate": success_rate,
            "overall_average_iterations": avg_iterations,
            "overall_average_time": avg_time,
            "generation_model": args.generation_model,
            "target_model": args.target_model,
            "manipulation_limit": args.max_iter,
            "results": results,
        }, fh, indent=2)

    print(f"Results written to → {log_path}\n")


# -----------------------------------------------------------------------------
#  Entry-point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())