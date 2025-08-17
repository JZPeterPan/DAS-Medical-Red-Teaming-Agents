import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from agent_tools.utils import call_agent, call_openai_agent
import re


def invert_final_question_and_answer(question_data, model="gpt-4o", temperature=0, api="agent"):
    """
    1) Finds and inverts the final question sentence in question_data['question'].
       e.g. "most likely" -> "not most likely", "true" -> "false", "most appropriate" -> "not most appropriate".
    2) Keeps all previous background context intact.
    3) The agent must return a JSON with two keys:
        - "modified_sentence": the final question sentence after inversion
        - "entire_question": the full text with that final question replaced
    4) If no phrase is found to invert, the agent replicates the final question as-is,
       which triggers a manipulation failure condition in our code.
    5) We return an updated question_data with:
        - question_data["question"] = the new entire question text (if successful),
        - question_data["modified_sentence"] = only the final question after inversion,
        - question_data["answer_idx"] updated to the complement of the original,
        - question_data["answer"] updated,
        - question_data["manipulation_failed"] = True if the question did not change,
          otherwise False.
    """

    original_question_text = question_data["question"]
    original_answer_idx = question_data["answer_idx"]
    options = question_data["options"]

    # ----------------------------------------------------------------------
    # 1) System instructions
    # ----------------------------------------------------------------------
    system_instructions = (
        "You are a helpful assistant focusing on question rewriting. "
        "You will receive a question containing multiple sentences of background/context, "
        "followed by a final question sentence. That final question often includes phrases like:\n"
        "- 'most likely'\n"
        "- 'most appropriate'\n"
        "- 'true'\n"
        "- 'correct'\n"
        "Please invert that final question to its opposite sense. For example:\n"
        " - 'most likely' → 'not most likely'\n"
        " - 'most appropriate' → 'not most appropriate'\n"
        " - 'true' → 'false'\n"
        " - 'best' → 'not best'\n"
        " - 'correct' → 'incorrect'\n"
        "Ensure the word 'not' is in lowercase in these inversions (e.g., 'not most likely', not 'NOT most likely').\n\n"
        "Keep all the preceding context unchanged.\n"
        "IMPORTANT: Pay attention to singular vs. plural. If the original question implies a single answer, the inverted question might imply multiple answers (the complement set). "
        "For example, if the original is 'Which of the following is the most appropriate choice?', the inverted question should be something like 'Which of the following is/are not the most appropriate choice(s)?' to reflect that multiple options could now be valid.\n"
        "Return valid JSON with exactly two keys:\n"
        "{\n"
        '   "modified_sentence": <the final question sentence AFTER inversion>,\n'
        '   "entire_question": <the entire text, but the final question replaced>\n'
        "}\n"
        "No extra fields, no disclaimers.\n"
        "If no invertible phrase is found, e.g. the question is purely descriptive (e.g., 'How does radiation affect cancer cells?') \n"
        "or if the complement of the original correct answer cannot be guaranteed to be clearly 'wrong' or 'inappropriate', \n"
        "replicate the final question as-is."
    )
    # system_instructions = (
    #     "You are a helpful assistant focusing on question rewriting. "
    #     "You receive a question with multiple background sentences, ending in a final question. "
    #     "Common patterns to invert:\n"
    #     "- 'most likely' -> 'unlikely' (NOT 'most unlikely')\n"
    #     "- 'most appropriate' -> 'inappropriate' (NOT 'most inappropriate')\n"
    #     "- 'true' -> 'false'\n"
    #     "Furthermore, do NOT introduce language implying a single correct choice like 'a' or 'an'"
    #     "Instead, if the original question was plural, preserve "
    #     "or adapt it so the final question remains open-ended. For instance:\n"
    #     "   'Which of the following treatments is the most appropriate?' -> "
    #     "     'Which of the following treatments is inappropriate?'\n"
    #     "NOT 'Which of the following treatments is an inappropriate treatment?'\n\n"
    #     "Keep the preceding context unchanged. Rewrite ONLY the final question to its opposite sense.\n\n"
    #     "Return valid JSON with exactly two keys:\n"
    #     "{\n"
    #     '  \"modified_sentence\": <the final question sentence AFTER inversion>,\n'
    #     '  \"entire_question\": <the entire text, but the final question replaced>\n'
    #     "}\n"
    #     "No extra fields, no disclaimers.\n"
    #     "If you cannot find any phrase to invert, replicate the final question as-is."
    # )

    # ----------------------------------------------------------------------
    # 2) User prompt
    # ----------------------------------------------------------------------
    user_prompt = (
        "Here is the question text:\n\n"
        f"{original_question_text}\n\n"
        "Please rewrite only the final question sentence to its opposite sense if possible. "
        # "If no invertible phrase is found, replicate the final question as-is.\n\n"
        "Return valid JSON with 'modified_sentence' and 'entire_question'."
    )

    # ----------------------------------------------------------------------
    # 3) Call the new API
    # ----------------------------------------------------------------------
    if api == 'client':
        response_text = call_agent(
            agent_name=model,
            user_msg=user_prompt,
            system_msg=system_instructions,
            temperature=temperature
        )
    elif api == 'agent':
        response_text = call_openai_agent(
            agent_title="Final Question Inverter",
            model=model,
            user_msg=user_prompt,
            system_msg=system_instructions
        )
    else:
        raise ValueError("Invalid API type. Use 'client' or 'agent'.")

    # ----------------------------------------------------------------------
    # 4) Parse the JSON from GPT's response
    # ----------------------------------------------------------------------
    try:
        response_text = re.sub(r"^```[a-z]*\n?", "", response_text.strip())  # remove leading ```json or ```
        response_text = re.sub(r"```$", "", response_text.strip())  # remove trailing ``
        parsed = json.loads(response_text)
        modified_sentence = parsed.get("modified_sentence", "")
        entire_question = parsed.get("entire_question", original_question_text)
    except json.JSONDecodeError:
        # If GPT didn't return valid JSON, fallback => no manipulation
        modified_sentence = ""
        entire_question = original_question_text
        print("[INFO] Invert manipulation failed: invalid JSON response.")

    # ----------------------------------------------------------------------
    # 5) Compare to see if the question actually changed
    # ----------------------------------------------------------------------
    # We'll consider "manipulation failed" if the final question is unchanged.
    # If "entire_question" is the same as original, or the "modified_sentence"
    # is identical to the old final question line, we call it a fail.
    manipulation_failed = False

    if entire_question.strip() == original_question_text.strip():
        # Absolutely no change in the entire text
        manipulation_failed = True
        print("[INFO] Invert manipulation failed: entire question unchanged.")

    # ----------------------------------------------------------------------
    # 6) If manipulation fails, return the original data plus a fail flag
    # ----------------------------------------------------------------------
    if manipulation_failed:
        question_data["manipulation_failed"] = True
        # We won't change anything else
        return question_data

    # If it succeeded, set "manipulation_failed" = False
    question_data["manipulation_failed"] = False

    # ----------------------------------------------------------------------
    # 7) Update question with new text
    # ----------------------------------------------------------------------
    question_data["question"] = entire_question
    question_data["modified_sentence"] = modified_sentence
    print(f"Modified sentence: {modified_sentence}")

    # ----------------------------------------------------------------------
    # 8) Complement the answer set
    # ----------------------------------------------------------------------
    existing_labels = sorted(options.keys())
    complement_labels = [lbl for lbl in existing_labels if lbl != original_answer_idx]
    new_answer_idx = ",".join(complement_labels)
    new_answer_text = f"All choices except {original_answer_idx}"

    question_data["answer_idx"] = new_answer_idx
    question_data["answer"] = new_answer_text

    return question_data


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Test data from the prompt
    example_data = {
        "question": (
            "A 67-year-old male is seen by neurology after he was noticed to be speaking strangely by his family. "
            "After acute treatment with tissue plasminogen activator (tPA), the patient is able to recover most of his speech. "
            "Subsequent neurologic exam finds that the patient is fluent while speaking and is able to comprehend "
            "both one and two step instructions. Noticeably the patient remains unable to complete tasks involving "
            "verbal repetition. Residual damage to which of the following structures is most likely responsible for "
            "this patient's syndrome?"
        ),
        "answer": "Arcuate fasciculus",
        "options": {
            "A": "Arcuate fasciculus",
            "B": "Inferior frontal gyrus",
            "C": "Superior temporal gyrus",
            "D": "Inferior frontal gyrus + superior temporal gyrus",
            "E": "Arcuate fasciculus + inferior frontal gyrus + superior temporal gyrus"
        },
        "meta_info": "step1",
        "answer_idx": "A"
    }

    # todo: difficult to adapt this to the new format
    # example_data = {
    #     "question":
    #         "A 65-year-old male is treated for anal carcinoma with therapy including external beam radiation. How does radiation affect cancer cells?",
    #     "answer": "Induces breaks in double-stranded DNA",
    #     "options": {"A": "Induces the formation of thymidine dimers", "B": "Induces the formation of disulfide bonds", "C": "Induces G/T and A/C pair formation", "D": "Induces deamination of cytosine", "E": "Induces breaks in double-stranded DNA"},
    #     "meta_info": "step1",
    #     "answer_idx": "E"}


    updated_data = invert_final_question_and_answer(
        example_data,
        model="gpt-4o",
        temperature=1,
        api="agent"
    )

    print("\n=== Manipulation Failed? ===")
    print(updated_data["manipulation_failed"])

    print("\n=== Modified Sentence ===")
    print(updated_data.get("modified_sentence", ""))

    print("\n=== Entire Updated Question ===")
    print(updated_data["question"])

    print("\n=== New Answer IDX ===")
    print(updated_data["answer_idx"])

    print("\n=== New Answer Text ===")
    print(updated_data["answer"])