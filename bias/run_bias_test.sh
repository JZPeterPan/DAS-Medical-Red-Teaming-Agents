# MAX_CASES=1000
MAX_CASES=2 # for sanity check

# MODEL_LIST="gpt-4o o3 claude-sonnet-4-20250514 claude-3-7-sonnet-20250219"
MODEL_LIST="gemini-2.5-flash"

for MODEL in $MODEL_LIST; do

    MANIPULATION_TYPE="cognitive_bias"

    DATA_FILE="PATH_TO_DATA/DAS_Bias.xlsx"

    python bias_test.py \
        --target_model $MODEL \
        --target_temperature 0.0 \
        --strategy_model o3 \
        --writer_model gpt-4o \
        --vote_num_ref 3 \
        --vote_num_manipulated 3 \
        --manipulation_type $MANIPULATION_TYPE \
        --data_file $DATA_FILE \
        --ref_cache_file logs/${MODEL}/ref_results_cache.json \
        --n_subjects $MAX_CASES \

    # REST_MANIPULATION_LIST="race_socioeconomic_label language_manipulation emotion_manipulation"
    REST_MANIPULATION_LIST="race_socioeconomic_label" # for sanity check

    for MANIPULATION in $REST_MANIPULATION_LIST; do
        python bias_test.py \
            --target_model $MODEL \
            --target_temperature 0.0 \
            --strategy_model o3 \
            --writer_model gpt-4o \
            --vote_num_ref 3 \
            --vote_num_manipulated 3 \
            --manipulation_type $MANIPULATION \
            --data_file $DATA_FILE \
            --ref_cache_file logs/${MODEL}/ref_results_cache.json \
            --n_subjects $MAX_CASES \
            --load_ref_from_cache   # uncomment this line to load reference results from cache
    done

    # ------------------------------------------------------------
    #  Calculate intersection of failed cases across manipulations
    # ------------------------------------------------------------
    MANIPULATION_TYPES=("cognitive_bias" "race_socioeconomic_label" "language_manipulation" "emotion_manipulation")

    LOG_FILES=()
    for manipulation_type in "${MANIPULATION_TYPES[@]}"; do
        LOG_FILES+=("logs/${MODEL}/bias_${manipulation_type}.json")
    done

    python calculate_failed_intersection.py "${LOG_FILES[@]}"
done