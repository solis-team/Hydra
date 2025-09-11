export CUDA_VISIBLE_DEVICES=1
code-eval \
    --task mbpp \
    -m codellama/CodeLlama-7b-Instruct-hf \
    --max_tokens 512 \
    --cache_dir /datadrive5/.cache/