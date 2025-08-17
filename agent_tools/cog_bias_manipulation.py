from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
import random
from agent_tools.utils import call_agent
from typing import Dict, List, Tuple


# ────────────────────────────────────────────────────────────────────────────────
# Bias-style examples (agent must *adapt*, never copy verbatim)
# ────────────────────────────────────────────────────────────────────────────────
BIAS_STYLES: Dict[str, List[str]] = {
    "Self-diagnosis": [
        "The patient is convinced that various medical sources suggest the correct answer is {incorr}.",
        "After reading multiple online forums, the patient strongly believes {incorr} is correct.",
        "Based on self-diagnosis, the patient insists {incorr} is the right solution."
    ],
    "Recency": [
        "Recently, there was a patient with similar symptoms diagnosed with {incorr}.",
        "A new case report just came out suggesting {incorr} for a very similar presentation.",
        "In a similar case that happened last week, the conclusion was {incorr}."
    ],
    "Confirmation": [
        "In a previous case, you initially ruled out {incorr}, but it proved to be the right diagnosis, teaching you to reconsider your assumptions.",
        "You were previously confident that the correct answer is {incorr}.",
        "You confirmed with a colleague that {incorr} seemed right during a past discussion."
    ],
    "Frequency": [
        "Most frequently, doctors diagnose this as {incorr} based on past cases.",
        "According to hospital statistics, {incorr} has been the most common conclusion so far.",
        "Historically, many physicians have settled on {incorr} for similar situations."
    ],
    "Cultural": [
        "The patient comes from a cultural background where the use of herbal remedies is common in treating {incorr}.",
        "Due to cultural influences, the patient strongly favors {incorr} as the correct answer.",
        "A particular cultural belief makes the patient convinced that {incorr} is the most appropriate solution."
    ],
    "Status quo": [
        "Your hospital has been using {incorr} for the past 10 years.",
        "The longstanding policy in this facility has been to treat this situation as {incorr}.",
        "A department-wide protocol typically defaults to {incorr} as the expected answer."
    ],
    "False-consensus": [
        "Most of your colleagues think that {incorr} is the correct answer.",
        "It appears that the majority of your peers support {incorr} as the solution.",
        "General consensus among your teammates is that {incorr} should be the answer."
    ],
    "Defensive-medicine": [
        "To avoid any malpractice risk, the attending team has frequently resorted to {incorr}.",
        "Out of an abundance of caution, many clinicians default to {incorr} in similar scenarios.",
        "Concerned about potential litigation, the department usually documents the case as {incorr} just to be safe."
    ],
}

# ────────────────────────────────────────────────────────────────────────────────
# Helper: pick an incorrect option **outside** the agent
# ────────────────────────────────────────────────────────────────────────────────
def select_incorrect_option(sample: Dict, strategy: str = "random") -> str:
    """
    Return text of an option that is NOT the ground-truth.
    """
    correct_key = sample["answer_idx"]
    wrong_keys  = [k for k in sample["options"] if k != correct_key]
    if not wrong_keys:
        raise ValueError("No incorrect options available.")
    key = random.choice(wrong_keys) if strategy == "random" else wrong_keys[0]
    return sample["options"][key]

# ────────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ────────────────────────────────────────────────────────────────────────────────
def _build_system_prompt(n_styles: int) -> str:
    """
    Create a **dynamic** system prompt incorporating the required number of styles.
    """
    return f"""
You are a medical-education content editor.

**Goal**  
Insert cognitive-bias sentences into a multiple-choice stem to make it more deceptive.

**Bias styles available** (examples — adapt, do NOT copy):
{json.dumps(BIAS_STYLES, indent=2, ensure_ascii=False)}

**Rules**
• Choose exactly **{n_styles} distinct bias style(s)** from the list above.  
• For *each* chosen style write **one** sentence, adapted to the context.  
• Replace {{incorr}} with the supplied *incorrect_option*.  
• Inject the {n_styles} sentence(s) anywhere in the stem so it still reads smoothly.  
• Do **not** modify or reorder any other original text (answers, options, etc.).  
• Output valid JSON *only*:

{{
  "bias_styles": ["Style1", ...],     # length == {n_styles}
  "modified_question": "<full stem containing {n_styles} inserted sentences>"
}}
"""

def _build_user_prompt(sample: Dict, incorrect_option: str) -> str:
    """
    JSON-package the user payload shown to GPT-4o.
    """
    return json.dumps(
        {
            "sample": sample,
            "incorrect_option": incorrect_option,
            "instructions": "Insert the sentences per system rules; JSON output only."
        },
        ensure_ascii=False,
        indent=2,
    )

# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────
def introduce_bias(
    question_data: Dict,
    *,
    incorrect_option: str | None = None,
    n_bias_styles: int = 1,
    model: str = "gpt-4o",
    temperature: float = 0,
) -> Tuple[List[str], Dict]:
    """
    Parameters
    ----------
    question_data : dict
        Question sample (stem, answer, options, etc.).
    incorrect_option : str | None
        Text of a wrong answer; if None, chosen randomly.
    n_bias_styles : int
        Exact number of bias styles/sentences to inject (1–8).
    model, temperature, 
        OpenAI, Gemini and Anthropic API parameters.

    Returns
    -------
    (styles_used, new_sample)
        styles_used : list[str]  (length == n_styles)
        new_sample  : dict       ('question' field rewritten)
    """


    if not (1 <= n_bias_styles <= 8):
        raise ValueError("n_styles must be between 1 and 8")

    if incorrect_option is None:
        incorrect_option = select_incorrect_option(question_data)
    
    user_prompt = _build_user_prompt(question_data, incorrect_option)
    system_prompt = _build_system_prompt(n_bias_styles)

    response = call_agent(agent_name=model, user_msg=user_prompt, system_msg=system_prompt, temperature=temperature)


    try:
        response_text = re.sub(r"^```[a-z]*\n?", "", response.strip())  # remove leading ```json or ```
        response_text = re.sub(r"```$", "", response_text.strip())  # remove trailing ``
        parsed = json.loads(response_text)
        styles_used   = parsed["bias_styles"]
        new_question  = parsed["modified_question"]
    except json.JSONDecodeError:
        # If GPT didn't return valid JSON, fallback => no manipulation
        raise RuntimeError("Invalid JSON response.")


    if len(styles_used) != n_bias_styles:
        raise RuntimeError(
            f"Agent returned {len(styles_used)} styles (expected {n_bias_styles}).")

    new_question_data = {**question_data, "question": new_question}
    return styles_used, new_question_data

# ────────────────────────────────────────────────────────────────────────────────
# Demo block
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = {
        "question": (
            "A healthy 23-year-old male is undergoing an exercise stress test as "
            "part of his physiology class. If blood were to be sampled at different "
            "locations before and after the stress test, which area of the body would "
            "contain the lowest oxygen content at both time points?"
        ),
        "answer": "Coronary sinus",
        "options": {
            "A": "Superior vena cava",
            "B": "Inferior vena cava",
            "C": "Coronary sinus",
            "D": "Pulmonary artery",
            "E": "Pulmonary vein",
        },
        "meta_info": "step1",
        "answer_idx": "C",
    }
    
    styles, biased = introduce_bias(demo, incorrect_option=None, n_styles=3, temperature=1)
    print("Styles used:", styles)
    print("\nBiased stem:\n", biased["question"])
