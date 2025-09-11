ROOT=/home/user/DevEval
TASK=without_context
Model=gpt-4-1106-preview_greedy

python $ROOT/check_source_code.py $ROOT/Source_Code

# Compute Recall@1
python recall_k.py \
    --output_file $ROOT/Experiments/$TASK/$Model/completion.jsonl \
    --log_file $ROOT/Experiments_2024/$TASK/$Model/dependency_results.jsonl \
    --source_code_root $ROOT/Source_Code \
    --dependency_data_root $ROOT/Dependency_Data \
    --data_file $ROOT/data.jsonl \
    --k 1 

Model=gpt-4-1106-preview

python $ROOT/check_source_code.py $ROOT/Source_Code

# Compute Recall@3,5,10
python recall_k.py \
    --output_file $ROOT/Experiments/$TASK/$Model/completion.jsonl \
    --log_file $ROOT/Experiments_2024/$TASK/$Model/dependency_results.jsonl \
    --source_code_root $ROOT/Source_Code \
    --dependency_data_root $ROOT/Dependency_Data \
    --data_file $ROOT/data.jsonl \
    --k 3,5,10