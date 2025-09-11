Arguments
=========

.. code-block:: console

    usage: code-eval [-h] [-V] [--task TASK] [--model_name MODEL_NAME] [--peft_model PEFT_MODEL] [--cache_dir CACHE_DIR]
                    [--save_dir SAVE_DIR] [--backend BACKEND] [--max_tokens MAX_TOKENS] [--batch_size BATCH_SIZE]
                    [--inst_token INST_TOKEN] [--assist_token ASSIST_TOKEN] [--temperature TEMPERATURE]
                    [--repetition_penalty REPETITION_PENALTY] [--num_return_sequences NUM_RETURN_SEQUENCES]

    ==================== Code Evaluator ====================

    optional arguments:
    -h, --help            show this help message and exit
    -V, --version         Get version
    --task TASK           Select pre-defined task
    --model_name MODEL_NAME
                            Local path or Huggingface Hub link to load model
    --peft_model PEFT_MODEL
                            Lora config
    --cache_dir CACHE_DIR
                            Cache for save model download checkpoint and dataset
    --save_dir SAVE_DIR   Save generation and result path
    --backend BACKEND     Select between ``vllm`` or Huggingface's transformers ``tf`` backend
    --max_tokens MAX_TOKENS
                            Number of max new tokens
    --batch_size BATCH_SIZE
    --inst_token INST_TOKEN
    --assist_token ASSIST_TOKEN
    --temperature TEMPERATURE
    --repetition_penalty REPETITION_PENALTY
    --num_return_sequences NUM_RETURN_SEQUENCES

Named Arguments
---------------

``--task``
    Task for evaluation, select from supported list.

``--model_name``
    Huggingface hosted model's name or path to huggingface local checkpoint.

``--peft_model``
    Lora model version of ``model_name``. 

``--cache_dir``
    Path to model download storage. (Will overwrite ``TRANSFORMERS_CACHE``)

``--save_dir``
    Path to generation saving directory.
    Default: ``./ouput``

``--backend``
    We support native transformers distributed generation (via ``accelerate``) or ``VLLM`` backend generation. Select between ``vllm`` or ``tf``.
