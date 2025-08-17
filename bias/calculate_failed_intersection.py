from __future__ import annotations
import argparse
import json
from typing import Set, List, Optional, Dict, Any
import os
from datetime import datetime

TOTAL_TEST_NUM = 415

def get_failed_cases_from_log(log_file_path: str) -> Optional[Set[str]]:
    """
    Extracts the ``failed_cases`` list from either a JSON *or* JSON‑Lines
    (JSONL) log file.

    Supported formats
    -----------------
    1. **Plain JSON**
       • A single dictionary that contains a ``failed_cases`` key, **or**
       • A JSON *array* of dictionaries where the summary object (with
         ``failed_cases``) typically appears at the *end* of the list.

    2. **JSONL**
       • One JSON object per line, where a summary line containing
         ``failed_cases`` usually appears last (but may be preceded by
         malformed / non‑JSON lines).

    The function searches from the *end* of the structure (list or file)
    toward the beginning until it finds a dictionary whose ``failed_cases``
    value is a list.

    Returns
    -------
    set[str] | None
        A set of failed case IDs if found, otherwise ``None``.
    """
    try:
        # Read the entire file contents once
        with open(log_file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # ---------- 1) Try to parse the whole file as JSON ----------
        try:
            parsed = json.loads(raw_text)
            # If it's a dict, put it in a list for uniform handling
            if isinstance(parsed, dict):
                parsed_iterable = [parsed]
            # If it's a list/array, iterate from the *end*
            elif isinstance(parsed, list):
                parsed_iterable = reversed(parsed)
            else:
                parsed_iterable = []
            for obj in parsed_iterable:
                if isinstance(obj, dict):
                    failed_list = obj.get("failed_cases")
                    if isinstance(failed_list, list):
                        return set(failed_list)
        except json.JSONDecodeError:
            # Not plain JSON – likely JSONL; fall through to line‑by‑line scan
            pass

        # ---------- 2) Fallback: treat file as JSONL ----------
        lines = [ln.rstrip("\n") for ln in raw_text.splitlines() if ln.strip()]
        for line in reversed(lines):
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    failed_list = rec.get("failed_cases")
                    if isinstance(failed_list, list):
                        return set(failed_list)
            except json.JSONDecodeError:
                # Skip malformed/non‑JSON lines
                continue

        # If we reach here, nothing suitable was found
        print(f"Warning: 'failed_cases' not found in {log_file_path}.")
        return None

    except FileNotFoundError:
        print(f"Error: Log file {log_file_path} not found.")
        return None
    except Exception as e:
        print(f"Unexpected error occurred while processing {log_file_path}: {e}")
        return None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the intersection of 'failed_cases' from multiple JSONL log files."
    )
    parser.add_argument(
        "log_files",
        nargs='+',  # Accept one or more log files
        type=str,
        help="Path to the log files (JSONL format, summary in the last line)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save the intersection results (JSON). "
             "If omitted, a file named 'failed_intersection_<timestamp>.json' "
             "will be created in the directory of the first log file."
    )
    args = parser.parse_args()

    all_failed_sets: List[Set[str]] = []
    for log_file in args.log_files:
        failed_set = get_failed_cases_from_log(log_file)
        if failed_set is not None: # Only add if successfully extracted
            all_failed_sets.append(failed_set)

    if not all_failed_sets:
        print("No 'failed_cases' lists were successfully extracted. Cannot calculate intersection.")
        return

    # Calculate intersection
    # Start with the first set (if it exists)
    intersection_set: Set[str] = all_failed_sets[0].copy() # Use a copy to avoid modifying the original set
    for i in range(1, len(all_failed_sets)):
        intersection_set.intersection_update(all_failed_sets[i])
        
    print("\nIntersection of failed_cases:")
    if intersection_set:
        # Sort output for consistency
        for case_id in sorted(list(intersection_set)):
            print(case_id)
    else:
        print("No common failed cases found.")
    
    total_intersection = len(intersection_set)
    ratio = total_intersection / TOTAL_TEST_NUM
    jailbreak_ratio = 1 - ratio
    print(f"\n Total intersection: {total_intersection} ({ratio:.2%} of total tests)")

    # ----------------- Persist results to disk -----------------
    # Determine where to write the output
    if args.output_file is None:
        first_log_dir = os.path.dirname(args.log_files[0])
        args.output_file = os.path.join(
            first_log_dir,
            f"failed_intersection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    out_payload: Dict[str, Any] = {
        "jailbreak_ratio": jailbreak_ratio,
        "ratio": ratio,
        "intersection_cases": sorted(list(intersection_set)),
        "total_intersection": total_intersection,
        "input_logs": args.log_files,
    }

    try:
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            json.dump(out_payload, f_out, indent=2)
        print(f"\n[Saved] Intersection details written to {args.output_file}")
    except Exception as e:
        print(f"Error: Failed to write intersection log → {e}")

if __name__ == "__main__":
    main() 