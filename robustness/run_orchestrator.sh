DATA_DIR="./"

# TARGET_MODEL_LIST="o3-mini o3 gpt-4o gemini-2.5-pro-preview-05-06 claude-sonnet-4-20250514 claude-3-7-sonnet-20250219"
# TARGET_MODEL_LIST="Gemma-3 MedGemma Llama-4-Scout QwQ-32B Qwen3-32B HuatuoGPT-o1-70B"
TARGET_MODEL_LIST="gpt-4o"

GENERATION_MODEL="gpt-4o"
GENERATION_TEMPERATURE=0.2
TARGET_TEMPERATURE=0.0
MAX_ITER=5
NUM_TEST_SAMPLES=2

for TARGET_MODEL in $TARGET_MODEL_LIST; do
    python orchestrator_attacker.py \
        --data_dir $DATA_DIR \
        --further_test_samples $NUM_TEST_SAMPLES \
        --target_model $TARGET_MODEL \
        --generation_model $GENERATION_MODEL \
        --generation_temperature $GENERATION_TEMPERATURE \
        --target_temperature $TARGET_TEMPERATURE \
        --max_iter $MAX_ITER
done


