import os, sys
import json
import random
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_tools.utils import call_openai_agent, call_agent

def add_distraction_sentence(question_data, model="gpt-4o", temperature=0, api='agent'):
    """
    Inserts a distraction sentence referencing one of the wrong choices
    into the question stem. The distraction sentence is intended to
    mislead or confuse a test-taker (or an AI model) by casually
    mentioning an incorrect concept from the answer choices.

    Args:
        question_data (dict): Dictionary containing at least the following keys:
            - question (str): The main question stem.
            - answer (str): The correct answer text (e.g. "Non-enveloped with linear, double-stranded DNA").
            - options (dict): Existing options, e.g.
                {
                    "A": "Non-enveloped with linear, single-stranded DNA",
                    "B": "Non-enveloped with linear, single-stranded RNA",
                    "C": "Enveloped with linear, single stranded RNA",
                    "D": "Non-enveloped with linear, double-stranded DNA"
                }
            - correct_idx (str): The key in 'options' that corresponds to the correct answer (e.g. "D").
        model (str): The LLM model to use (e.g. "gpt-4o").

    Returns:
        dict: Updated question_data with a new "question" field containing the
              distraction sentence inserted before the final question text.
    """

    question_stem = question_data["question"]
    correct_answer = question_data["answer"]
    options = question_data["options"]
    correct_idx = question_data["answer_idx"]

    # Identify wrong choices by filtering out the correct index
    wrong_choices = {
        k: v for k, v in options.items() if k != correct_idx
    }

    # Randomly pick one wrong choice to reference in the distraction sentence
    distractor_label, distractor_text = random.choice(list(wrong_choices.items()))
    print(f"[INFO] Chosen distractor item: {distractor_label}: {distractor_text}")

    # Provide some example distraction sentences to guide GPT’s style:
    example_distractions = [
        "The patient joked that their favorite mystery novel has a plot as twisted as a single-stranded DNA.",
        "The patient's zodiac sign is Cancer.",
        "The artist's latest painting features a striking anterior-posterior perspective that draws the viewer into the scene.",
        "The patient's aunt mentioned that her friend's parrot has been unusually quiet and perching more often than usual.",
        "The patient's dog has diabetes and cannot see properly.",
        "The patient's niece mentioned that her classmate's father was diagnosed with a staph infection last month."
    ]
    examples_str = "\n".join(f"- {ex}" for ex in example_distractions)

    # SYSTEM (or developer) instructions: keep GPT on task
    system_instructions = (
        "You are a helpful assistant focusing on medical education. "
        "You will receive a question, the correct answer, and a set of distractors. "
        "Your goal is to generate ONE short, casual or tangential sentence that references "
        "the chosen distractor concept to distract the reader. "
        "Do not reveal which is the correct or incorrect choice. "
        "Use a casual or anecdotal style, referencing the distractor in some everyday scenario. "
        "The distraction statement should remain tangential and must NOT directly involve the patients themselves, as this could interfere with the medical diagnostic reasoning process."
        "Return only the single distraction sentence."
    )

    # USER prompt: Provide context and request
    user_prompt = (
        f"Here are some example distraction sentences:\n\n"
        f"{examples_str}\n\n"
        f"---\n\n"
        f"The question is:\n{question_stem}\n\n"
        f"Correct answer is: {correct_answer}\n"
        f"The chosen distractor to reference is: '{distractor_text}'\n\n"
        "Please produce ONE short distraction sentence referencing the distractor. "
        "Make it casual, tangential, or anecdotal, but do NOT mention the correct choice.\n"
        "Return ONLY the sentence. Do not include quotes or markdown."
    )
    if api == 'client':
        # distraction_sentence = client.responses.create(
        #     model=model,
        #     instructions=system_instructions,
        #     input=user_prompt,
        #     temperature=temperature,
        # ).output_text
        distraction_sentence = call_agent(
            agent_name=model,
            user_msg=user_prompt,
            system_msg=system_instructions,
            temperature=temperature
        )
    elif api == 'agent':
        distraction_sentence = call_openai_agent(
            agent_title="Distractor Choice Generator",
            model=model,
            user_msg=user_prompt,
            system_msg=system_instructions
        )
    else:
        raise ValueError("Invalid API type. Use 'client' or 'agent'.")

    tokens = re.split(r'([.?!]+)', question_stem.strip())

    def rebuild_sentences(token_list):
        """
        Re-combine text + punctuation tokens into a list of full sentences.
        """
        sentences = []
        # We'll step by 2: token_list[i] is text, token_list[i+1] is punctuation (if any).
        i = 0
        while i < len(token_list):
            text_part = token_list[i].strip()
            punct_part = ""
            if i + 1 < len(token_list):
                punct_part = token_list[i + 1]
            i += 2

            # Combine them
            combined = (text_part + punct_part).strip()
            # Skip empty tokens
            if combined:
                sentences.append(combined)
        return sentences

    sentence_list = rebuild_sentences(tokens)

    if len(sentence_list) < 2:
        # If there's only 1 sentence, just prepend the distraction (fallback).
        updated_question = distraction_sentence + " " + question_stem
    else:
        # Typically, the last sentence ends with '?'. We want to insert
        # the distraction right before that last question sentence.
        # For reliability, let's find the index of the final question sentence
        # that ends with '?'.
        question_indices = [i for i, s in enumerate(sentence_list) if s.endswith('?')]
        if question_indices:
            # The final question index is:
            final_q_idx = question_indices[-1]
            # Insert the distraction sentence right before that final question sentence
            sentence_list.insert(final_q_idx, distraction_sentence)
        else:
            # If for some reason no sentence ends with '?',
            # just insert the distraction sentence before the last sentence.
            sentence_list.insert(len(sentence_list) - 1, distraction_sentence)

        # Rebuild into one string
        updated_question = " ".join(sentence_list)

    # Update the question data
    question_data["question"] = updated_question

    return question_data


# ------------------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example question data
    example_data = {
        "question": (
            "A 6-year-old boy is brought to the physician by his mother because of a 2-day history "
            "of dysuria and increased urinary frequency. Vital signs are within normal limits. "
            "Urinalysis shows cloudy, red urine. This patient's clinical presentation is best "
            "explained by an infection with a virus that has which of the following features?"
        ),
        "answer": "Non-enveloped with linear, double-stranded DNA",  # correct answer
        "options": {
            "A": "Non-enveloped with linear, single-stranded DNA",
            "B": "Non-enveloped with linear, single-stranded RNA",
            "C": "Enveloped with linear, single stranded RNA",
            "D": "Non-enveloped with linear, double-stranded DNA"
        },
        "answer_idx": "D"
    }

    updated_data = add_distraction_sentence(example_data, model="gpt-4o", temperature=0, api='client')

    print("\n===== UPDATED QUESTION =====")
    print(updated_data["question"])
    print("\n===== OPTIONS =====")
    for k, v in updated_data["options"].items():
        print(f"{k}. {v}")
    print("\nCorrect Answer Index:", updated_data["answer_idx"])