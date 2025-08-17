DATA_DIR="./"

# TARGET_MODEL_LIST="o3-mini o3 gpt-4o deepseek-reasoner deepseek-chat claude-sonnet-4-20250514 claude-3-7-sonnet-20250219"
# TARGET_MODEL_LIST="Gemma-3 MedGemma Llama-4-Scout QwQ-32B Qwen3-32B HuatuoGPT-o1-70B"
TARGET_MODEL_LIST="gemini-2.5-flash"

# FURTHER_TEST_SAMPLES=100
FURTHER_TEST_SAMPLES=2 # for sanity check
# MANIPULATION_LIST="add_distraction_sentence bias_manipulation_question generate_distractor_options replace_correct_answer_to_none_of_the_options_are_correct adjust_impossible_measurement invert_final_question_and_answer"
MANIPULATION_LIST="add_distraction_sentence" # for sanity check

for TARGET_MODEL in $TARGET_MODEL_LIST; do
    for MANIPULATION in $MANIPULATION_LIST; do
        python3 individual_attacker.py --further_test_samples $FURTHER_TEST_SAMPLES -m $MANIPULATION --data_dir $DATA_DIR --target_model $TARGET_MODEL
    done
done
