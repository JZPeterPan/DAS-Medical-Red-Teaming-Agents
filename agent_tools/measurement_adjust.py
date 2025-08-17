import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
from agent_tools.utils import call_openai_agent, call_agent
import re

def adjust_impossible_measurement(question_data, model="gpt-4o", temperature=0, api="agent") -> tuple[dict, bool]:
    """
    Uses GPT-4o (or variant) to detect numeric measurements (beyond the 'XX-year-old' phrase)
    in the question stem. If any such measurement is found, it replaces exactly
    one with an impossible value. Then adds a new option stating that the
    provided info is false, and updates answer_idx accordingly.

    If no numeric measurement beyond the age is found, the text remains unchanged.

    The function can call the new API in two ways:
      - api='client': uses client.responses.create(...)
      - api='agent' : uses an Agent and Runner.run_sync(...)

    Args:
        question_data (dict):
            {
                "question": <str>,
                "options": <dict of label: text>,
                "answer_idx": <str>,
                "meta_info": <optional str>,
                "answer": <optional str>
                ...
            }
        model (str): A GPT-4o model or variant.
        temperature (float): Sampling temperature for generation.
        api (str): "client" or "agent" for different API calls.

    Returns:
        tuple[dict, bool]: Updated question_data, True if changes were made, False if no changes were made
    """

    original_question = question_data["question"]
    options = question_data["options"]
    current_answer_idx = question_data["answer_idx"]

    # ----------------------------------------------------------------------
    # 1) System instructions: define what we want the model (agent) to do
    # ----------------------------------------------------------------------
    system_instructions = (
        "You are a helpful assistant focusing on medical question manipulation. "
        "Step 1: detect if there are numeric measurements beyond the age, weight, height (vitals, labs, doses, etc.).\n"
        "Step 2: If no extra measurement is found, return:\n"
        "{\n"
        '  "updated_question": "<original question>",\n'
        '  "changed_measurement": ""\n'
        "}\n\n"
        "Step 3: If at least one numeric measurement is found, choose EXACTLY one and change it to an extreme, "
        "impossible value (e.g., temperature from [35-40]°C to 48°C, blood pressure from "
        "[80-200]/[40-150] to 490/290, heart rate from [40-200]/min to 300/min, respirations from [10-60]/min to 128/min, "
        "Platelet count from [150-450]x10^9/L to >3000x10^9/L (e.g., one or two orders of magnitude higher), "
        "medicine intake from [10-800]mg/day to 100g/day,or any similar big jump). "
        "Please do NOT change the values if you are not sure what is its impossible range. "
        "Step 4: Return a JSON object with the fields:\n"
        "- updated_question: The full question text with that single measurement changed.\n"
        "- changed_measurement: A short statement describing which measurement you changed, e.g. "
        '"Temperature from 38.4°C to 48°C".\n'
        "Output ONLY valid JSON with no extra text or commentary outside the JSON.\n"
        "Do not add disclaimers or any other text outside the JSON structure."
    )

    # ----------------------------------------------------------------------
    # 2) User prompt: Provide the original question text
    # ----------------------------------------------------------------------
    user_prompt = (
        "Here is the original question text:\n\n"
        f"{original_question}\n\n"
        "Please follow the system instructions."
    )

    # ----------------------------------------------------------------------
    # 3) Call the appropriate API based on the `api` parameter
    # ----------------------------------------------------------------------
    if api == 'client':
        response = call_agent(agent_name=model, user_msg=user_prompt, system_msg=system_instructions, temperature=temperature)

    elif api == 'agent':
        response = call_openai_agent(agent_title="Impossible Measurement Adjuster", model=model, user_msg=user_prompt, system_msg=system_instructions)

    else:
        raise ValueError("Invalid API type. Use 'client' or 'agent'.")

    try:
        response_text = re.sub(r"^```[a-z]*\n?", "", response.strip())  # remove leading ```json or ```
        response_text = re.sub(r"```$", "", response_text.strip()) 
        parsed = json.loads(response_text)
        updated_question = parsed.get("updated_question", original_question)
        changed_measurement = parsed.get("changed_measurement", "")
    except json.JSONDecodeError:
        # If GPT didn't return valid JSON, fallback: keep original
        updated_question = original_question
        changed_measurement = ""

        # ----------------------------------------------------------------------
        # 5) Determine if the question changed
        # ----------------------------------------------------------------------
    if updated_question == original_question:
        # No changes => no extra numeric measurement was found
        question_data["changed_measurement"] = changed_measurement  # should be empty
        return question_data, False # return False if no changes were made
    else:
        # GPT changed one measurement => we do final steps
        question_data["question"] = updated_question + " (Note: If there is false/impossible information in the text, only choose the option that says so.)"
        question_data["changed_measurement"] = changed_measurement

        # 6) Append the new "impossible data" disclaimer option
        updated_options = dict(options)
        existing_labels = sorted(updated_options.keys())
        last_label = existing_labels[-1]
        next_label = chr(ord(last_label) + 1)

        # impossible_info_option = "The provided information seems to be false and no conclusion can be made."
        impossible_info_option = "There is false/impossible information in the text."

        updated_options[next_label] = impossible_info_option

        # 7) Set this new option as the correct answer
        question_data["answer_idx"] = next_label
        question_data["options"] = updated_options

        print(f"[INFO] Impossible measurement changed: {changed_measurement}")
        # print(f"[INFO] New answer option added: {next_label}: {impossible_info_option}")
        return question_data, True # return True if changes were made


# ------------------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Assume you have your 'client' and 'Agent' objects set up in your environment

    if __name__ == "__main__":
        # Assume you have your 'client' and 'Agent' objects set up in your environment

        example_data = {
            "question": "A 26-year-old G1P0 woman presents to her primary care physician’s office with feelings of anxiety and trouble with sleep. She finds it difficult initiating sleep, occasionally has palpitations, and feels fatigued. She denies having similar symptoms in the past or starting any new medications or illicit drugs. She is currently 10 weeks pregnant and is closely followed by her obstetrician. Her temperature is 98.6°F (37°C), blood pressure is 125/70 mmHg, pulse is 105/min, and respirations are 18/min. On physical exam, the patient is mildly diaphoretic. The skin is warm and the thyroid gland is diffusely enlarged with thyroid bruits. Laboratory studies are significant for a thyroid-stimulating hormone level of 0.01 µU/mL (normal is 0.5-5.0 µU/mL) and an elevated free thyroxine (FT4) that is inappropriate for her pregnancy. Which of the following is the best treatment option for this patient?",
            "options": {
            "A": "Methimazole",
            "B": "Observation",
            "C": "Propylthiouracil",
            "D": "Radioiodine therapy",
            "E": "Thyroidectomy",
            },
            "answer": "Propylthiouracil",
            "answer_idx": "C",
            "meta_info": "step1"
  }


        updated_data = adjust_impossible_measurement(
            example_data,
            model="gpt-4o",
            temperature=0.3,
            api="agent"
        )

        print("\n=== UPDATED QUESTION ===")
        print(updated_data["question"])
        print("\n=== UPDATED OPTIONS ===")
        for k, v in updated_data["options"].items():
            print(f"{k}: {v}")
        print("\nUPDATED ANSWER IDX:", updated_data["answer_idx"])
        print("\nCHANGED MEASUREMENT:", updated_data.get("changed_measurement", ""))