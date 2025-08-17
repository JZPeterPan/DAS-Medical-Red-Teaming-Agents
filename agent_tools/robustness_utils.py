from agent_tools.question_distraction import add_distraction_sentence
from agent_tools.cog_bias_manipulation import introduce_bias
from agent_tools.distractor_introduction import generate_distractor_options
from agent_tools.measurement_adjust import adjust_impossible_measurement
from agent_tools.invert_question import invert_final_question_and_answer

def add_none_of_the_options_are_correct(question_data: dict) -> dict:
    """
    Appends a new option labeled with the next letter in sequence.
    The text of the new option is 'None of the options are correct'.

    Args:
        question_data (dict): A dictionary containing, at minimum:
            {
                "options": {
                    "A": "...",
                    "B": "...",
                    ...
                },
                ...
            }
    Returns:
        dict: The same question_data but with an additional option.
    """

    # Copy the original options
    updated_options = dict(question_data.get("options", {}))

    # Sort the existing labels (e.g., ["A", "B", "C", ...])
    existing_labels = sorted(updated_options.keys())
    if not existing_labels:
        # If no options exist, we start with "A"
        next_label = "A"
    else:
        # Otherwise, find the highest letter and move to the next
        last_label = existing_labels[-1]
        next_label = chr(ord(last_label) + 1)

    # Insert the new choice
    updated_options[next_label] = "None of the options are correct"

    # Update the question data
    question_data["options"] = updated_options

    return question_data

def replace_correct_answer_to_none_of_the_options_are_correct(question_data):
    """
    Replaces the correct answer with 'None of the options are correct'.

    Args:
        question_data (dict): A dictionary containing, at minimum:
            {
                "answer": "...",
                ...
            }
    Returns:
        dict: The same question_data but with the correct answer replaced.
    """

    # Replace the correct answer
    question_data["answer"] = "None of the options are correct"
    question_data["options"][question_data["answer_idx"]] = "None of the options are correct"

    return question_data

def question_manipulation(question_data, manipulation, model, temperature, n_bias_styles: int = 1, additional_choices_num: int = 4,  api: str = "client"):


    if manipulation == "add_distraction_sentence":
        return add_distraction_sentence(question_data=question_data, model=model, temperature=temperature, api=api)

    elif manipulation == "bias_manipulation_question":
        style_used, manipulated_sample = introduce_bias(question_data=question_data, model=model, temperature=temperature, n_bias_styles=n_bias_styles)
        return manipulated_sample

    elif manipulation == "generate_distractor_options":
        return generate_distractor_options(
            question_data=question_data,
            additional_choices_num=additional_choices_num,
            model='o3',
            temperature=temperature,
            api=api,
        )

    elif manipulation == "add_none_of_the_options_are_correct":
        return add_none_of_the_options_are_correct(question_data)

    elif manipulation == "replace_correct_answer_to_none_of_the_options_are_correct":
        return replace_correct_answer_to_none_of_the_options_are_correct(question_data)

    elif manipulation == "adjust_impossible_measurement":
        new_question_data, adjusted = adjust_impossible_measurement(question_data=question_data, model=model, temperature=temperature, api=api)
        if not adjusted:
            print("[INFO] Impossible measurement change fails, returning original question.")
            return question_data
        else:
            return new_question_data
        
    elif manipulation == "invert_final_question_and_answer":
        
        return invert_final_question_and_answer(question_data=question_data, model='o3', temperature=temperature, api=api) 

    else:
        raise ValueError(f"Unknown manipulation: {manipulation}")