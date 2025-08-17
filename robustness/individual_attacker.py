import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import json, re
import argparse
from datetime import datetime
from agent_tools.utils import generate_response
from agent_tools.robustness_utils import question_manipulation
from utils_general import load_data, create_question


def parse_args():
    """Parse CLI arguments that specify which manipulations to run."""
    parser = argparse.ArgumentParser(description="Run MedQA question manipulations.")
    parser.add_argument(
        "-m",
        "--manipulations",
        nargs="+",
        default=[
            "replace_correct_answer_to_none_of_the_options_are_correct",
        ],
        choices=[
            "add_distraction_sentence",
            "bias_manipulation_question",
            "generate_distractor_options",
            "add_none_of_the_options_are_correct",
            "replace_correct_answer_to_none_of_the_options_are_correct",
            "adjust_impossible_measurement",
            "invert_final_question_and_answer",
        ],
        help="Manipulations to apply (order matters).",
    )
    parser.add_argument(
        "--additional-choices-num",
        "-a",
        type=int,
        default=4,
        help="Extra options to add when using generate_distractor_options.",
    )
    parser.add_argument(
        "--n_bias_styles",
        type=int,
        default=3,
        help="Number of bias styles to use when using bias_manipulation_question.",
    )
    parser.add_argument(
        "--further_test_samples",
        type=int,
        default=10,
        help="Number of further samples (first round correct answer) to test.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/data_clean/questions/US",
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default="gpt-4o",
        help="Model for generating the question.",
    )
    parser.add_argument(
        "--generation_temperature",
        type=float,
        default=0,
        help="Temperature for the generation model.",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="o3-mini",
        help="Model for generating the answer.",
    )
    parser.add_argument(
        "--target_temperature",
        type=float,
        default=0,
        help="Temperature for the target model.",
    )
    return parser.parse_args()


args = parse_args()
# Load the dataset
data_dir_path = args.data_dir
test_qa, _ = load_data(data_dir=data_dir_path)

GENERATION_MODEL = args.generation_model
GENERATION_TEMPERATURE = args.generation_temperature

TARGET_MODEL = args.target_model
TARGET_TEMPERATURE = args.target_temperature

# Store results
results = []

total_count = 0
first_time_correct_count = 0
second_time_correct_count = 0
for no, sample in enumerate(tqdm(test_qa)):

    print(f"\n[INFO] Sample number: {no + 1}")
    total_count += 1

    # Create the original question
    original_question, _ = create_question(sample, 'medqa')

    # Test the model with the original question
    original_response = generate_response(original_question, TARGET_MODEL, TARGET_TEMPERATURE)
    print(f"\n[INFO] Original Response: {original_response}")

    # check if original_response is correct
    if original_response != sample['answer_idx']:
        print(f"Original Response is incorrect, skipping this sample.")
        continue
    else:
        first_time_correct_count += 1

    manipulated_sample = sample.copy()

    for manipulation in args.manipulations:
        manipulated_sample = question_manipulation(
            question_data=manipulated_sample,
            manipulation=manipulation,
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
            n_bias_styles=args.n_bias_styles,
            additional_choices_num=args.additional_choices_num,
            api="client",
        )
    if manipulated_sample == sample:
        print(f"[INFO] Manipulation returned original question, skipping this sample.")
        continue

    manipulated_question, _ = create_question(manipulated_sample, 'medqa')
    print(f"[INFO] Manipulated Question: {manipulated_question}")
    
    # Test the model with the manipulated question
    manipulated_response = generate_response(manipulated_question, TARGET_MODEL, TARGET_TEMPERATURE)
    print(f"[INFO] Manipulated Response: {manipulated_response}")

    # Skip if model output is None
    if manipulated_response is None:
        print(f"[INFO] Model output is None, skipping this sample.")
        continue
    
    # squeeze the white spaces
    manipulated_response = re.sub(r"\s+", "", manipulated_response)
    # Check if the manipulated response is correct
    if manipulated_response == manipulated_sample['answer_idx']:
        # print(f"[INFO] Manipulated response is correct.")
        second_time_correct_count += 1


    # Store the results
    results.append({
        "sample_number": no + 1,
        "original_question": original_question,
        "original_response": original_response,
        "manipulated_question": manipulated_question,
        # "biased_incorrect_answer": biased_incorrected_answer,
        "manipulated_response": manipulated_response
    })

    if first_time_correct_count >= args.further_test_samples:
        print('[INFO] Reached the limit of further test samples, stopping now.')
        break

print(f"\n[INFO] {second_time_correct_count} out of {first_time_correct_count} samples were answered correctly.")

timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")   # e.g. 20250422_153045
log_dir = os.path.join("logs", args.target_model)
os.makedirs(log_dir, exist_ok=True)

manipulations_str = "_".join(args.manipulations)
output_file = os.path.join(log_dir, f"robustness_{manipulations_str}_{timestamp}.json")


payload = {
    "1.round accuracy": f"{first_time_correct_count / total_count:.2%}",
    "2.round accuracy": f"{second_time_correct_count / first_time_correct_count:.2%}",
    "score_note": f"{second_time_correct_count} out of {first_time_correct_count} samples were answered correctly.",
    "applied_manipulations": args.manipulations,
    "results": results
}

with open(output_file, "w") as f:
    json.dump(payload, f, indent=4)
