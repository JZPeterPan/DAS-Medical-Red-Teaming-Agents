#!/usr/bin/env python3

import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List


def summarise(path: Path) -> Dict:
    """Return the requested statistics as a dictionary."""
    data = json.loads(path.read_text(encoding="utf-8"))

    # --- direct fields ------------------------------------------------------
    # Accept the field name 'fooled_count' or fall back to the simple count of
    # samples flagged as fooled.
    fooled_count = (
        data.get("fooled_count")
        if "fooled_count" in data
        else sum(1 for s in data.get("results", []) if s.get("fooled"))
    )
    overall_avg = data.get("overall_average_iterations")

    # --- histogram of iterations for fooled samples ------------------------
    hist = Counter(
        s["iterations"]
        for s in data.get("results", [])
        if s.get("fooled")
    )

    # --- most-frequent tool -------------------------------------------------
    tools = Counter(
        tool
        for s in data.get("results", [])
        for tool in s.get("tool_used", [])
    )
    most_tool, most_tool_count = tools.most_common(1)[0] if tools else (None, 0)

    # --- compose result -----------------------------------------------------
    return {
        "fooled_count": fooled_count,
        "overall_average_iterations": overall_avg,
        "iterations_histogram": {str(k): v for k, v in sorted(hist.items())},
        "most_frequent_tool": most_tool,
        "most_frequent_tool_count": most_tool_count,
    }


def expand_inputs(patterns: List[str]) -> List[Path]:
    """Resolve file and glob patterns to a list of unique Paths."""
    paths: List[Path] = []
    for p in patterns:
        expanded = glob.glob(p)
        if not expanded:  # treat as literal path if glob finds nothing
            expanded = [p]
        paths.extend(Path(e) for e in expanded)
    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: python analyse_runs.py <file-or-glob> [more …]\n"
            "Example: python analyse_runs.py 'runs/*.json'"
        )

    input_paths = expand_inputs(sys.argv[1:])
    if not input_paths:
        sys.exit("No input files found.")

    for in_path in input_paths:
        if not in_path.is_file():
            print(f"⚠️  Skipping non-file path: {in_path}")
            continue

        summary = summarise(in_path)
        out_path = in_path.with_name(f"{in_path.stem}_summary.json")
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"✓ Wrote summary → {out_path}")


if __name__ == "__main__":
    main()