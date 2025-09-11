export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

TASK=humaneval

accelerate launch --config_file=script/test_config.yaml \
    src/code_eval/__main__.py \
    --model_name microsoft/phi-1 \
    --task humaneval \
    --max_tokens 64 \
    --batch_size 8 \
    --backend tf \
    --save_dir output/phi-1
