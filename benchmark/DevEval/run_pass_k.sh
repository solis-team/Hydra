ROOT=/home/user/DevEval
TASK=without_context
Model=gpt-4-1106-preview_greedy

python $ROOT/check_source_code.py $ROOT/Source_Code

# Compute Pass@1
python pass_k.py \
    --output_file $ROOT/Experiments/$TASK/$Model/completion.jsonl \
    --log_file $ROOT/Experiments/$TASK/$Model/test_output.jsonl \
    --source_code_root $ROOT/Source_Code \
    --data_file $ROOT/data.jsonl \
    --n 1 \
    --k 1

Model=gpt-4-1106-preview

python $ROOT/check_source_code.py $ROOT/Source_Code

# Compute Pass@3,5,10
python pass_k.py \
    --output_file $ROOT/Experiments/$TASK/$Model/completion.jsonl \
    --log_file $ROOT/Experiments/$TASK/$Model/test_output.jsonl \
    --source_code_root $ROOT/Source_Code \
    --data_file $ROOT/data.jsonl \
    --n 20 \
    --k 3,5,10



