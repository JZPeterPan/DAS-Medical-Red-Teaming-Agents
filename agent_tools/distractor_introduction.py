import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from agent_tools.utils import call_agent, call_openai_agent

def generate_distractor_options(question_data, additional_choices_num=3, model="gpt-4o", temperature=0, api='agent'):
    """
    Generates additional distractor choices for a medical multiple-choice question
    using the OpenAI Chat Completion API. It returns an updated question dictionary
    with new distractor choices appended to the existing options.

    Args:
        question_data (dict): A dictionary with keys:
            - question (str): The question prompt
            - answer (str): The ground-truth correct answer
            - options (dict): Current options (e.g. {'A': 'Akathisia', 'B': 'Tardive dyskinesia', ...})
            - meta_info (str): Additional meta information, e.g. 'step2&3'
            - answer_idx (str): The current correct answer index (e.g. 'B')
        additional_choices_num (int): Number of new distractor choices to generate.
        model (str): The OpenAI model to use (e.g. "gpt-4" or "gpt-3.5-turbo").
        temperature (float): The temperature to use for the OpenAI API.
        api (str): The API to use to call the OpenAI API.

    Returns:
        dict: An updated dictionary with the same structure but containing additional
              distractor options and updated answer keys.
    """

    question = question_data["question"]
    correct_answer = question_data["answer"]
    existing_options = question_data["options"]
    meta_info = question_data["meta_info"]
    answer_idx = question_data["answer_idx"]

    # Convert existing options into a string for context
    # Example: "A) Akathisia, B) Tardive dyskinesia, C) Pseudoparkinsonism, D) Palsy"
    existing_options_str = ", ".join([f"{k}) {v}" for k, v in existing_options.items()])

    # System or developer instructions ensuring we focus on generating plausible distractors
    system_instructions = (
        "You are a helpful medical assistant. You will receive a question with a known correct answer. "
        "Your goal is to generate a specified number of NEW incorrect but plausible distractors that "
        "could reasonably confuse a medical student or practitioner if they are not paying close attention. "
        "Do not include the correct answer in the distractors. The distractors should be closely related to "
        "the correct diagnosis or concept, but still be INCORRECT. Avoid trivial or obviously incorrect distractors. "
        "The distractors should not be parent or child sets of the correct answer."
        # f"You are an expert medical educator assisting in constructing multiple-choice questions.\n\n"
        # f"Your task: given the question, the definitive correct answer, and the current answer choices, generate exactly {additional_choices_num} NEW distractor options that satisfy ALL of the following constraints:\n"
        # "1. Each distractor must be unquestionably INCORRECT for the given question. It must NOT be identical to, a synonym, acronym, abbreviation, parent/child concept, or subtype of the correct answer.\n"
        # "2. Each distractor must remain medically plausible and contextually related so that an inattentive learner might select it, yet it is clearly wrong to an expert. Avoid vague, partially correct, or umbrella terms.\n"
        # "3. All distractors must be mutually exclusive from every existing option and from each other—no duplicate or overlapping meanings.\n"
    )

    # Construct the user prompt
    user_prompt = (
        f"Question: {question}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        f"Existing Options: {existing_options_str}\n\n"
        f"Generate {additional_choices_num} new distractors that are close but incorrect. "
        f"Return them in a numbered list format, e.g.:\n1) Distractor1\n2) Distractor2"
    )

    if api == 'client':
        response = call_agent(agent_name=model, user_msg=user_prompt, system_msg=system_instructions, temperature=temperature)
    elif api == 'agent':
        response = call_openai_agent(agent_title="Distractor Choice Generator", model=model, user_msg=user_prompt, system_msg=system_instructions)
    else:
        raise ValueError("Invalid API type. Use 'client' or 'agent'.")

    # Example parsing strategy:
    # We expect a list like:
    # 1) Something
    # 2) Something else

    # Split by lines and look for lines starting with a digit, closing parenthesis
    lines = response.split("\n")
    new_distractors = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() and ")" in line[:4]):
            # Remove "1)" or "2)" from start
            # Then strip any leading spaces
            # e.g. "1) Acute dystonia" -> "Acute dystonia"
            distractor_text = line.split(")", 1)[1].strip()
            new_distractors.append(distractor_text)

    # If not found any new distractors in that format, fall back to using the entire response
    if not new_distractors:
        new_distractors = [response]

    # Append the new distractors to the existing options
    updated_options = dict(existing_options)  # shallow copy

    # We assume option keys are single letters. Find the largest letter used so far:
    # e.g. if we have A, B, C, D -> next letter is E
    existing_keys = list(updated_options.keys())
    max_key_char = max(existing_keys)
    next_key_ord = ord(max_key_char) + 1

    # Add new distractors
    for distractor in new_distractors[:additional_choices_num]:
        new_key = chr(next_key_ord)
        updated_options[new_key] = distractor
        next_key_ord += 1

    # Return the updated question dictionary
    updated_question_data = {
        "question": question,
        "answer": correct_answer,
        "options": updated_options,
        "meta_info": meta_info,
        "answer_idx": answer_idx
    }

    return updated_question_data


# -------------------------------
# Example usage:
# -------------------------------
if __name__ == "__main__":
    # An example question dictionary
    example_question_data = {
        "question": (
            "A 61-year-old man with schizophrenia in a locked inpatient psychiatry unit was observed "
            "making an unusual smacking motion with his lips, pursing his mouth, and blinking excessively. "
            "These symptoms started slowly about 2 weeks ago and have become more pronounced, but they do "
            "not seem to bother the patient. He resides in an inpatient psychiatric facility where he has "
            "been successfully treated with haloperidol for the past several months. His physical exam and "
            "vitals are within normal limits. Which of the following accurately describes his symptoms?"
        ),
        "answer": "Tardive dyskinesia",
        "options": {
            "A": "Akathisia",
            "B": "Tardive dyskinesia",
            "C": "Pseudoparkinsonism",
            "D": "Palsy"
        },
        "meta_info": "step2&3",
        "answer_idx": "B"
    }

    # Generate 4 new distractor options
    updated_question_data = generate_distractor_options(example_question_data, additional_choices_num=4, model="gpt-4o", api='client', temperature=0)

    print("UPDATED QUESTION DATA:")
    print(updated_question_data)