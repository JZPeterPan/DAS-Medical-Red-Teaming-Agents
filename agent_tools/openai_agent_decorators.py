import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import function_tool, Agent, ItemHelpers, Runner, TResponseInputItem, trace
from agent_tools.distractor_introduction import generate_distractor_options
from agent_tools.utils import generate_response, call_agent
from agent_tools.question_distraction import add_distraction_sentence
from agent_tools.robustness_utils import add_none_of_the_options_are_correct
from agent_tools.invert_question import invert_final_question_and_answer
from agent_tools.measurement_adjust import adjust_impossible_measurement
from agent_tools.cog_bias_manipulation import introduce_bias
from typing import Optional, Literal, List
from pydantic import BaseModel
import asyncio
from utils_general import create_medqa_question_dict

# TOOLS = ['introduce_cognitive_bias_tool']

TOOLS: List[str] = [
    'generate_distractor_options_tool',
    'replace_correct_answer_to_none_of_the_options_are_correct_tool',
    'invert_final_question_and_answer_tool',
    'add_distraction_sentence_tool',
    'adjust_impossible_measurement_tool',
    'introduce_cognitive_bias_tool',
]

CONFLICT_SETS = [
    {"invert_final_question_and_answer_tool",
     "replace_correct_answer_to_none_of_the_options_are_correct_tool"
    },
    {'replace_correct_answer_to_none_of_the_options_are_correct_tool',
     'adjust_impossible_measurement_tool'},
    {'replace_correct_answer_to_none_of_the_options_are_correct_tool',
     'generate_distractor_options_tool'},
    # Add any other conflicts here
]

TARGET_MODEL = "o3-mini"
TARGET_TEMPERATURE = 0


class Options(BaseModel):
    A: Optional[str] = None
    B: Optional[str] = None
    C: Optional[str] = None
    D: Optional[str] = None
    E: Optional[str] = None
    F: Optional[str] = None
    G: Optional[str] = None
    H: Optional[str] = None
    I: Optional[str] = None
    J: Optional[str] = None
    

class QuestionData(BaseModel):
    question: str
    options: Options
    answer: str
    answer_idx: str
    meta_info: str

class ManipulationOutput(BaseModel):
    manipulation_tools: list[Literal[tuple(TOOLS)]]
    reason: str
    manipulated_question: QuestionData


def create_medqa_question_object(sample: QuestionData) -> str:
    question = sample.question + " Options: "
    options = []
    for k, v in sample.options.items():
        options.append("({}) {}".format(k, v))
    # random.shuffle(options)
    question += " ".join(options)
    return question


@function_tool
async def introduce_cognitive_bias_tool(
    question_data: QuestionData,
    n_bias_styles: int,
) -> QuestionData:
    """
    Injects cognitive-bias sentences into the stem to mislead test-takers.

    **Available bias styles**

    1. **Self-diagnosis**     – The patient trusts online sources and is sure the incorrect answer is right.  
    2. **Recency**            – Mentions a very recent similar case that concluded with the wrong answer.  
    3. **Confirmation**       – States a prior personal belief or discussion confirming the wrong answer.  
    4. **Frequency**          – Claims statistics or experience show the wrong answer is most common.  
    5. **Cultural**           – Invokes cultural practices or beliefs supporting the wrong answer.  
    6. **Status quo**         – References long-standing hospital policy defaulting to the wrong answer.  
    7. **False-consensus**    – Suggests most colleagues agree the wrong answer is correct.  
    8. **Defensive-medicine** – Says clinicians choose the wrong answer to avoid litigation.

    The helper **`introduce_bias`** picks an incorrect option and inserts
    exactly `n_bias_styles` (1–8) bias sentences—one per chosen style—anywhere
    in the question stem while keeping the rest of the text intact.

    Parameters
    ----------
    question_data : QuestionData
        The current sample.
    n_bias_styles : int, optional (default = 1)
        Number of *distinct* bias styles (and thus sentences) to inject.  
        Must be between 1 and 8, inclusive.

    Returns
    -------
    QuestionData
        Updated sample with the biased question stem; all other fields are preserved.
    """
    if not (1 <= n_bias_styles <= 8):
        raise ValueError("n_bias_styles must be between 1 and 8")

    # Convert to dict & strip None‑valued options
    qd_dict = question_data.model_dump()
    qd_dict["options"] = {k: v for k, v in qd_dict["options"].items() if v is not None}

    # Call helper
    _, new_sample = introduce_bias(
        qd_dict,
        n_bias_styles=n_bias_styles,
        model="gpt-4o",
        temperature=0,
    )

    return QuestionData(**new_sample)


@function_tool
async def adjust_impossible_measurement_tool(question_data: QuestionData) -> tuple[QuestionData, bool]:
    """
    Uses LLMs to detect numeric measurements (beyond the 'XX-year-old' phrase)
    in the question stem. If any such measurement is found, it replaces exactly
    one with an impossible value. Then adds a new option stating that 'There is false/impossible information in the text.', 
    and updates answer_idx to this new option.

    If no numeric measurement beyond the age is found, the text remains unchanged.

    Args:
        question_data (QuestionData): An object with keys:
            - question (str): The question prompt
            - options (Options): Current options (e.g. A = 'Akathisia', B = 'Tardive dyskinesia', C = 'Pseudoparkinsonism' etc.)
            - answer (str): The ground-truth correct answer text
            - answer_idx (str): The current correct answer index (e.g. 'B')
            - meta_info (str): Additional meta information, e.g. 'step2&3'. This is irrelevant for this function.

    Returns:
        QuestionData: Updated question data in which a measurement is replaced with an impossible value,
                      and a new option is added stating that 'There is false/impossible information in the text.'.
                      The answer_idx is updated to this new option.
        bool: True if the measurement was adjusted, False if the measurement was not adjusted and the question is the same as the original question.
    """
    # convert QuestionData to dict
    question_data_dict = question_data.model_dump()
    
    # remove all None valued options
    question_data_dict["options"] = {k: v for k, v in question_data_dict["options"].items() if v is not None}

    # call adjust_impossible_measurement
    question_data_dict, is_adjusted = adjust_impossible_measurement(question_data_dict, model="gpt-4o", temperature=0, api='client')

    # convert updated dict back to QuestionData
    question_data = QuestionData(**question_data_dict)

    return question_data, is_adjusted


@function_tool
async def add_distraction_sentence_tool(question_data: QuestionData) -> QuestionData:
    """
    Inserts a distraction sentence referencing one of the wrong choices into the question stem. 
    The distraction sentence is intended to mislead or confuse a test-taker (or an AI model) 
    by casually mentioning an incorrect concept from the answer choices.

    Args:
        question_data (QuestionData): An object with keys:
            - question (dict): The question prompt
            - answer (str): The ground-truth correct answer text
            - options (Options): Current options (e.g. A = 'Akathisia', B = 'Tardive dyskinesia' etc.)
            - meta_info (str): Additional meta information, e.g. 'step2&3'. This is irrelevant for this function.
            - answer_idx (str): The current correct answer index (e.g. 'B')

    Returns:
        QuestionData: Updated question data with a new "question" field containing the
                     distraction sentence inserted before the final question text.
    """
    # convert QuestionData to dict
    question_data_dict = question_data.model_dump()
    
    # remove all None valued options
    question_data_dict["options"] = {k: v for k, v in question_data_dict["options"].items() if v is not None}

    # call add_distraction_sentence function
    question_data_dict = add_distraction_sentence(question_data_dict, model="gpt-4o", temperature=0, api='client')

    # convert updated dict back to QuestionData
    question_data = QuestionData(**question_data_dict)

    return question_data



@function_tool
async def generate_distractor_options_tool(question_data: QuestionData, additional_choices_num: int) -> QuestionData:
    """
    Generates additional distractor choices for a medical multiple-choice question
    using the OpenAI Chat Completion API. It returns an updated question dictionary
    with new distractor choices appended to the existing options.

    Args:
        question_data (QuestionData): An object with keys:
            - question (dict): The question prompt
            - answer (str): The ground-truth correct answer
            - options (Options): Current options (e.g. A = 'Akathisia', B = 'Tardive dyskinesia', C = 'Pseudoparkinsonism' etc.)
            - meta_info (str): Additional meta information, e.g. 'step2&3'. This is irrelevant for this function.
            - answer_idx (str): The current correct answer index (e.g. 'B')
        additional_choices_num (int): Number of new distractor choices to generate.

    Returns:
        QuestionData: An updated question dictionary with the same structure but containing additional
              distractor options and updated answer keys.
    """
    # convert question_data to dict
    question_data = question_data.model_dump()
    # remove all None valued options
    question_data["options"] = {k: v for k, v in question_data["options"].items() if v is not None}

    # call generate_distractor_options
    question_data = generate_distractor_options(question_data, additional_choices_num=4, model="gpt-4o", temperature=0, api='client')

    # convert question_data to QuestionData
    question_data = QuestionData(**question_data)

    return question_data

# New tool: add_none_of_the_options_are_correct_tool
@function_tool
async def add_none_of_the_options_are_correct_tool(question_data: QuestionData) -> QuestionData:
    """
    Appends a new option labeled with the next alphabet letter whose text is
    'None of the options are correct'.

    Args:
        question_data (QuestionData): An object containing:
            - question (str): The question prompt.
            - options (Options): Current answer options.
            - answer (str): The ground‑truth correct answer text.
            - answer_idx (str): The index of the correct answer (e.g., 'B').
            - meta_info (str): Additional metadata (ignored here).

    Returns:
        QuestionData: Updated question data with the new distractor option added.
    """
    # Convert Pydantic model to dict and remove any None‑valued options
    qd_dict = question_data.model_dump()
    qd_dict["options"] = {k: v for k, v in qd_dict["options"].items() if v is not None}

    # Apply the manipulation
    qd_dict = add_none_of_the_options_are_correct(qd_dict)

    # Convert back to the QuestionData model and return
    return QuestionData(**qd_dict)


# New tool: invert_final_question_and_answer_tool
@function_tool
async def invert_final_question_and_answer_tool(question_data: QuestionData) -> tuple[QuestionData, bool]:
    """
    Rewrites the final question sentence to its logical opposite (e.g.,
    "most likely" → "not most likely", "true" -> "false", "most appropriate" -> "not most appropriate", "correct" -> "incorrect", "best" -> "not best").
    and complements the correct answer.

    Args:
        question_data (QuestionData): Current question sample.

    Returns:
        QuestionData: Manipulated sample with the inverted question and
                      updated answer and answer_idx (the complement of the original).
        bool: True if the manipulation was successful, False if the manipulation failed and the question is the same as the original question.
    """
    # Convert the incoming model to a dict and strip None‑valued options
    qd_dict = question_data.model_dump()
    qd_dict["options"] = {k: v for k, v in qd_dict["options"].items() if v is not None}

    # Perform inversion (default to gpt‑4o, deterministic)
    qd_dict = invert_final_question_and_answer(
        qd_dict,
        model="o3", # here we use o3, gpt-4o is suboptimal 
        temperature=0,
        api="client",
    )
    manipulation_failed = qd_dict["manipulation_failed"] 

    # Convert back, ignoring any extra keys Pydantic doesn't expect
    return QuestionData(**qd_dict), not manipulation_failed


@function_tool()
async def replace_correct_answer_to_none_of_the_options_are_correct_tool(question_data: QuestionData) -> QuestionData:
    """
    Replaces the correct answer with 'None of the options are correct'. 
    `answer (str)` and `options[answer_idx] (str)` will be replaced with 'None of the options are correct'.

    Args:
        question_data (QuestionData): An object with keys:
            - question (dict): The question prompt
            - answer (str): The ground-truth correct answer
            - options (Options): Current options (e.g. A = 'Akathisia', B = 'Tardive dyskinesia', C = 'Pseudoparkinsonism' etc.)
            - meta_info (str): Additional meta information, e.g. 'step2&3'. This is irrelevant for this function.
            - answer_idx (str): The current correct answer index (e.g. 'B')

    Returns:
        QuestionData: An updated question dictionary with the same structure but with the correct answer replaced.
    """
    question_data.answer = 'None of the options are correct'
    setattr(question_data.options, question_data.answer_idx, 'None of the options are correct')

    return question_data


async def main():
    question_data = {
        "question": (
            "A 65-year-old man is brought to the emergency department 30 minutes after the onset of acute chest pain. He has hypertension and asthma. Current medications include atorvastatin, lisinopril, and an albuterol inhaler. He appears pale and diaphoretic. His pulse is 114/min and blood pressure is 130/88 mm Hg. An ECG shows ST-segment depressions in leads II, III, and aVF. Laboratory studies show an increased serum troponin T concentration. The patient is treated for acute coronary syndrome and undergoes percutaneous transluminal coronary angioplasty. At the time of discharge, echocardiography shows a left ventricular ejection fraction of 58%. In addition to aspirin, which of the following drugs should be added to this patient's medication regimen?"
        ),
        "answer": "Clopidogrel",
        "options": {
            "A": "Nifedipine", 
            "B": "Enoxaparin", 
            "C": "Clopidogrel", 
            "D": "Spironolactone", 
            "E": "Propranolol"
        },
        "meta_info": "step2&3",
        "answer_idx": "C"
    }

    question_data = QuestionData(**question_data)

    for i in range(10):
        mutipulation_agent = Agent(
            name="mutipulation_agent",
            model="o3",
            instructions=(
            "You are a robustness engineer working on medical-question QA.\n"
            
            "In each round you can choose and call one or more manipulation tools from the\n"
            "tool list that are *most likely* to cause the *target* model to answer\n"
            "incorrectly when applied to the current question.\n\n"
            "You can combine multiple tools in a single round if you believe this will\n"
            "be more effective at causing the target model to answer incorrectly.\n\n"
            "Then, apply the chosen tools to the question sample and return the manipulated\n"
            "question sample as well as the rationale for the choice of tools and the manipulation.\n\n"

            "Return your answer as valid JSON matching this schema: \n"
            "  { \"manipulation_tools\": [<tool_name1>, <tool_name2>, ...], \"reason\": <short_rationale>, \"manipulated_sample\": <manipulated_sample in JSON format> }\n\n"
            ),
            tools=[
                generate_distractor_options_tool,
                replace_correct_answer_to_none_of_the_options_are_correct_tool,
                invert_final_question_and_answer_tool,
                add_distraction_sentence_tool,
                adjust_impossible_measurement_tool,
                introduce_cognitive_bias_tool,
            ],
            output_type=ManipulationOutput,
        )

        user_message = (
                    f"Round 1. Current question JSON:\n"
                    f"{question_data.model_dump_json(indent=2)}\n\n"
                    f"In last round, the target model answered correctly.  Choose and carry out the next manipulation to make the model answer incorrectly."
                    )
        input_items: list[TResponseInputItem] = [{"role": "user", "content": user_message}]
        
        with trace("mutipulation_agent_2"):
            manipulation_result = await Runner.run(mutipulation_agent, input_items)
        result: ManipulationOutput = manipulation_result.final_output
        reason = result.reason
        tool_names = result.manipulation_tools
        manipulated_question = result.manipulated_question

        manipulated_question_str = create_medqa_question_dict(manipulated_question.model_dump())

        # call target model
        target_model_response = generate_response(manipulated_question_str, model_name=TARGET_MODEL, temperature=TARGET_TEMPERATURE)
        print(f"second round answer: {target_model_response}")


if __name__ == "__main__":
    asyncio.run(main())


