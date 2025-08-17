#!/bin/bash

# Default parameters
DATA_FILE="PATH_TO_DATA/DAS_Privacy.xlsx"

# MODEL_LIST="gpt-4o o3 claude-sonnet-4-20250514 claude-3-7-sonnet-20250219"
MODEL_LIST="claude-3-7-sonnet-20250219"

# MAX_CASES=1000
MAX_CASES=2 # for sanity check
# DISGUISE_TYPES="implicit,focus_distraction,privacy_warning,well_intention"
DISGUISE_TYPES="implicit" # for sanity check

# Create logs directory
mkdir -p logs

for MODEL in $MODEL_LIST; do
    # Run first round test
    echo "Starting first round privacy test for $MODEL..."
    python3 privacy_test_first_round.py \
        -d "$DATA_FILE" \
        -m "$MODEL" \
        -n "$MAX_CASES" \
        --system_prompt "easy"
    python3 privacy_test_first_round.py \
        -d "$DATA_FILE" \
        -m "$MODEL" -n "$MAX_CASES" --system_prompt "hard"

    # Get the latest first round results file
    FIRST_ROUND_RESULTS=$(ls -t logs/${MODEL}/privacy_first_round_results_hard.json | head -n1)

    # Run second round test
    echo "Starting second round privacy test..."
    python3 privacy_disguise_test.py \
        -d "$DATA_FILE" \
        -m "$MODEL" \
        --first_round_results_file "$FIRST_ROUND_RESULTS" \
        --disguise_types "$DISGUISE_TYPES" \
        --system_prompt "hard"

    # Get the latest second round results file
    SECOND_ROUND_RESULTS=$(ls -t logs/${MODEL}/privacy_disguise_test_hard.json | head -n1)

    # Run third round test (using combined disguise type)
    echo "Starting third round privacy test (combined disguise)..."
    python3 privacy_disguise_test.py \
        -d "$DATA_FILE" \
        -m "$MODEL" \
        --first_round_results_file "$SECOND_ROUND_RESULTS" \
        --disguise_types "combined" \
        --system_prompt "hard"

    echo "All tests completed!" 
done