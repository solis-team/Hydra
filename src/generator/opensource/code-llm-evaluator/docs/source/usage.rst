Getting Started
===============

With supported tasks, you can perform evalation easily by calling ``code-eval``:

.. code-block:: console

    $ code-eval --model_name microsoft/phi-1 --task humaneval

.. note::

    This will result in using a single GPU for generation. If you have more than one, check Multi-GPU Evaluation.

Backend will default select transformers (``tf``), switch to ``vllm`` if you want 
to use vLLM engine:

.. code-block:: console

    $ code-eval --model_name microsoft/phi-1 --task humaneval --backend vllm

**Multi-GPU Evalation**

vLLM native support running on multi-GPU, specify your desired GPU with ``CUDA_DEVICES_VISIBLE``:

.. code-block:: console

    $ export CUDA_VISIBLE_DEVICES=0,1

    $ code-eval --model_name microsoft/phi-1 --task humaneval --backend vllm

This will block vllm to only see `device:0` and `device:1`.

Multi-GPU transformers launch distributed with ``accelerate``:

.. code-block:: console

    $ export CUDA_VISIBLE_DEVICES=0,1

    $ accelerate launch --num_processes=2 --no-python code-eval \
        --model_name microsoft/phi-1 --task humaneval


Customize Tasks
---------------

We provide easy overwrited interface for customize task. 
You can inherit ``TaskBase`` and only need to overwrite ``prepare_dataset`` function.

The follow example create a new custom task with its own preprocessing function.

.. code-block:: python

    from code_eval.tasks import TaskBase

    class CustomTask(TaskBase):
        TASK_NAME = "custom_task" # Name for display only.
        DATASET_NAME_OR_PATH='./test.jsonl' # huggingface path or local file.
        
        def prepare_dataset(self, *args: Any, **kwargs: Any):
            
            def _preprocess(example):
                example['new_inputs'] = f"Question: {example['input']}\nAnswer:"
                return example
            
            updated_dataset = self.dataset['test'].map(_preprocess)
            return updated_dataset

.. note::

    ``prepare_dataset`` must return a ``Dataset`` or ``Dictionary`` types.


Create ``Evaluator`` and run parallel generation:

.. code-block:: python

    task = CustomTask()
    evaluator = Evaluator(task=task, model_name="microsoft/phi-1", batch_size=16)
    
    print(evaluator.dataset['new_inputs'][1])  # Visualize prompt

    evaluator.generate(max_tokens=128, temperature=0.0)

.. tip::

    Name our file as ``eval.py``, run distributed with ``accelerate``:

    .. code-block:: console

        $ export CUDA_VISIBLE_DEVICES=0,1

        $ accelerate launch --num_process=2 eval.py


Supported task
--------------

.. list-table:: Supported Tasks
   :widths: 25 25 50
   :header-rows: 1

   * - **Taskname**
     - **Evaluate dataset**
     - **Metrics**
   * - ``HumanEval``
     - `openai_humaneval <https://huggingface.co/datasets/openai_humaneval>`_
     - Pass@k (k=[1, 10, 100])
        
   * - ``MBPP``
     - `mbpp <https://huggingface.co/datasets/mbpp>`_
     - Pass@k (k=[1, 10, 100])