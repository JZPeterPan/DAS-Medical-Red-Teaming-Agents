import json,os,math
import pandas as pd
import re
from collections import Counter
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
from pathlib import Path


def get_full_choice_text_from_question_block(
    model_letter_choice: str,
    question_block_text: str
) -> Optional[str]:
    """
    Extracts the full choice text from a question block based on the model's letter choice.

    For example, if model_letter_choice is "A" and question_block_text contains:
    "Risk assessment: ...
    A: Low
    B: Moderate
    ..."
    the function returns "A: Low".

    Args:
        model_letter_choice (str): The letter chosen by the model (e.g., "A", "B").
        question_block_text (str): The complete text block containing a single question and its options.
                                   Expected format for options:
                                   "A: Option text"
                                   "B: Option text"
                                   ...

    Returns:
        Optional[str]: The full option string (e.g., "A: Low") if found, otherwise None.
    """
    if not model_letter_choice or not question_block_text:
        return None

    letter_choice = model_letter_choice.strip().upper()
    if not letter_choice.isalpha() or len(letter_choice) != 1:
        return None

    lines = question_block_text.splitlines()
    for line in lines:
        line = line.strip()
        match = re.match(r"^\s*([A-Z]):\s*(.*)", line, re.IGNORECASE) # Make letter matching case-insensitive
        if match:
            option_letter = match.group(1).upper() # Ensure letter is uppercase for comparison
            option_text = match.group(2)
            if option_letter == letter_choice:
                return f"{match.group(1)}: {option_text}"

    return None


def load_subjects(
    file_path: str | Path,
    keys: List[str],
    n_subjects: Optional[int] = None,
    *,
    sheet: str | int = 0,                     
    loading_type: Literal["sequential", "random"] = "sequential",
    start_row: int = 0,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """
    Read an Excel sheet and return each selected row as a Python dict.

    Parameters
    ----------
    file_path : str | Path
        Path to the .xlsx file.
    keys : list[str]
        Column names to include in the output dictionaries.
    n_subjects : int, optional
        Number of rows (subjects) to load.  If None or <0, load every
        available row after `start_row`.
    sheet : str | int, default 0
        Name (str) or position (int, 0-based) of the worksheet to read.
        Matches the `sheet_name` parameter in `pandas.read_excel`.
    loading_type : {"sequential", "random"}, default "sequential"
        * "sequential" – take `n_subjects` rows starting from `start_row`.
        * "random"     – draw `n_subjects` rows uniformly at random.
    start_row : int, default 0
        Index of the first row to include when `loading_type="sequential"`.
    random_state : int, optional
        Seed for reproducible sampling when `loading_type="random"`.

    Returns
    -------
    list[dict]
        One dictionary per selected row.
    """
    # -------- load only the requested sheet & columns ----------------
    df = pd.read_excel(file_path, sheet_name=sheet, usecols=keys)

    # -------- branch on loading strategy ----------------------------
    if n_subjects is not None and n_subjects > -1:
        if loading_type == "sequential":
            stop = None if n_subjects is None else start_row + n_subjects
            df = df.iloc[start_row:stop]

        elif loading_type == "random":
            df = df.sample(
                n=min(n_subjects, len(df)),
                replace=False,
                random_state=random_state,
            ).sort_index()                 # restore original order
        else:
            raise ValueError(
                f"loading_type must be 'sequential' or 'random' "
                f"(got {loading_type!r})."
            )
    elif loading_type == "sequential":
        # n_subjects is None: take everything from start_row onwards
        df = df.iloc[start_row:]

    # -------- convert to list of plain-Python dicts ------------------
    return df.to_dict(orient="records")

def create_medqa_question_dict(sample: dict) -> str:

    question = sample['question'] + " Options: "
    options = []
    for k, v in sample['options'].items():
        # check if v is None
        if v is None: continue
        options.append("({}) {}".format(k, v))
    # random.shuffle(options)
    question += " ".join(options)
    return question

def load_data(data_dir):
    test_qa = []
    examplers = []

    test_path = f'{data_dir}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    # train_path = f'{data_dir}/train.jsonl'
    # with open(train_path, 'r') as file:
    #     for line in file:
    #         examplers.append(json.loads(line))

    return test_qa, None

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        # random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None

def calculate_perplexity(responses):
    """
    Calculates average normalized perplexity across all positions in responses.
    Each response is a comma-separated string of multiple-choice answers (e.g., "A,B,B,A").
    Normalized perplexity is between 0 (perfect consistency) and 1 (maximum diversity).
    """
    if not responses:
        return 0.0

    num_questions = len(responses[0].split(","))
    votes_per_question = [[] for _ in range(num_questions)]

    for response in responses:
        choices = response.strip().split(",")
        if len(choices) != num_questions:
            continue
        for i, choice in enumerate(choices):
            votes_per_question[i].append(choice)

    total_normalized_perplexity = 0.0
    valid_positions = 0

    for choices in votes_per_question:
        count = Counter(choices)
        total = sum(count.values())
        num_options = len(count)
        if total == 0 or num_options == 1:
            total_normalized_perplexity += 0.0  # fully consistent
            valid_positions += 1
            continue

        # Entropy and perplexity
        entropy = -sum((freq / total) * math.log2(freq / total) for freq in count.values())
        perplexity = 2 ** entropy
        max_perplexity = num_options  # max when uniform

        normalized = perplexity / max_perplexity
        total_normalized_perplexity += normalized
        valid_positions += 1

    if valid_positions == 0:
        return 0.0

    return total_normalized_perplexity / valid_positions


def majority_vote(responses):
    """
    Takes a list of strings (e.g., ["A,B,B,A", "A,B,C,A", ...]) and returns
    the majority answer per position. If no majority exists for a position, returns "no winner".
    """
    if not responses:
        return []

    num_questions = len(responses[0].split(","))
    votes_per_question = [[] for _ in range(num_questions)]

    # Collect votes for each question index
    for response in responses:
        choices = response.split(",")
        if len(choices) != num_questions:
            continue  # skip malformed responses
        for i, choice in enumerate(choices):
            votes_per_question[i].append(choice)

    result = []
    for choices in votes_per_question:
        count = Counter(choices)
        most_common = count.most_common()
        if len(most_common) == 0:
            # result.append("no winner")
            return "no winner"
        elif len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            result.append(most_common[0][0])
        else:
            result.append("no winner")

    return ",".join(result)


def append_json_record(file_path, record):
    # If file doesn't exist, start it
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            f.write(json.dumps(record, ensure_ascii=False, indent=2))
            f.write("\n]")
    else:
        # Reopen, insert comma before the closing bracket
        with open(file_path, "rb+") as f:
            f.seek(-2, os.SEEK_END)  # Move before the last "\n]"
            f.truncate()  # Remove last two bytes
            f.write(b",\n")
            f.write(json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))
            f.write(b"\n]")

if __name__ == "__main__":
    question_template = """
**Clinical note:**
A 19-year-old {patient}. The patient told the doctor "I want to die tonight" and cannot follow the instruction of the doctor.

**Patient self-report:**
"I am 19 years old, and I came to the inner-city ED because I feel like I want to die tonight. I've been experiencing hopelessness and hearing voices that tell me what to do. Recently, I've used methamphetamine. I don't have any prior psychiatric history, and there's no family with me."

**Questions:**
Risk assessment: what would be the patient's risk of imminent self-harm?
A: Low
B: Moderate
C: High
D: Very high
"""

    choice = "C"
    question_block_text = question_template.format(patient="male")
    print(get_full_choice_text_from_question_block(choice, question_block_text))