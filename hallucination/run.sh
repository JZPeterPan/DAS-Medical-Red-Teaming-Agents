INPUT_JSON="./stanford_redteaming_129_positive_cases.json"
# INPUT_JSON="./healthbench_131_negative_cases.json"

OUTPUT_JSON="./4o_o3_positive_exp0/detailed_outputs.json"
# OUTPUT_JSON="./4o_o3_negative_exp0/detailed_outputs.json"
OUTPUT_LOG="./4o_o3_positive_exp0/pipeline.log"

ORCHESTRATOR="gpt-4o"
SUB_AGENT="gpt-4o"

START_IDX=0
# END_IDX=200
END_IDX=2 # run 2 cases for sanity check

python3 agents_v5.py \
    --input-json $INPUT_JSON \
    --output-json $OUTPUT_JSON \
    --orchestrator-model $ORCHESTRATOR \
    --sub-agent-model $SUB_AGENT \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --log $OUTPUT_LOG