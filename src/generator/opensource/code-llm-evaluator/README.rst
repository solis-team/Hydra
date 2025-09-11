=================
CodeLLM Evaluator
=================
Easy to evaluate with fast inference settings CodeLLMs

Overview
========
`CodeLLM Evaluator` provide the ability for fast and efficiently evaluation 
on code generation task. Inspired by `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_ and `bigcode-eval-harness <https://github.com/bigcode-project/bigcode-evaluation-harness>`_,
we designed our framework for multiple use-case, easy to add new metrics and customized task.

**Features:**

* Implemented HumanEval, MBPP benchmarks for Coding LLMs.
* Support for models loaded via `transformers <https://github.com/huggingface/transformers>`_, `DeepSpeed <https://github.com/microsoft/DeepSpeed>`_.
* Support for evaluation on adapters (e.g. LoRA) supported in HuggingFace's `PEFT <https://github.com/huggingface/peft>`_ library.
* Support for inference with distributed native transformers or fast inference with `VLLMs <https://github.com/vllm-project/vllm>`_ backend.
* Easy support for custom prompts, task and metrics.

Setup
=====
Install `code-eval` package from the github repository via `pip`:

.. code-block:: console

    $ git clone https://github.com/FSoft-AI4Code/code-llm-evaluator.git
    $ cd code-llm-evaluator
    $ pip install -e .

Quick-start
===========

To evaluate a supported task in python, you can load our :py:func:`code_eval.Evaluator` to generate
and compute evaluate metrics on the run.

.. code-block:: python

    from code_eval import Evaluator
    from code_eval.task import HumanEval

    task = HumanEval()
    evaluator = Evaluator(task=task)

    output = evaluator.generate(num_return_sequences=3,
                                batch_size=16,
                                temperature=0.9)
    result = evaluator.evaluate(output)


CLI Usage
=========

Inference with Transformers
---------------------------

Load model and generate answer using native transformers (``tf``), pass model local path or
HuggingFace Hub name. We select transformers as default backend, but you can pass ``backend="tf"`` to specify it:

.. code-block:: console

    $ code-eval --model_name microsoft/phi-1 \
        --task humaneval \
        --batch_size 8 \
        --backend hf \

.. tip:: 
    Load LoRA adapters by add ``--peft_model`` argument. The ``--model_name`` must point
    to full model architecture.

    .. code-block:: console

        $ code-eval --model_name microsoft/phi-1 \
            --peft_model <adapters-name> \
            --task humaneval \
            --batch_size 8 \
            --backend hf \


Inference with vLLM engine
--------------------------

We recommend using vLLM engine for fast inference. 
vLLM supported tensor parallel, data parallel or combination of both.
Reference to vLLM documenation for more detail. 

To use ``code-eval`` with vLLM engine, please refer to vLLM engine documents to `instal it <https://docs.vllm.ai/en/latest/getting_started/installation.html>`_.

.. note:: 
    
    You can install vLLM using pip:

    .. code-block:: console

        $ pip install vllm


With model supported by vLLM (See more: `vLLM supported model <https://docs.vllm.ai/en/latest/models/supported_models.html>`_) 
run:

.. code-block:: console

    $ code-eval --model_name microsoft/phi-1 \
        --task humaneval \
        --batch_size 8 \
        --backend vllm

.. tip::

    You can use LoRA with similar syntax.

    .. code-block:: console

        $ code-eval --model_name microsoft/phi-1 \
            --peft_model <adapters-name> \
            --task humaneval \
            --batch_size 8 \
            --backend vllm \

Cite as
=======

.. code-block:: 

    @misc{code-eval,
        author       = {Dung Nguyen Manh},
        title        = {A framework for easily evaluation code generation model},
        month        = 3,
        year         = 2024,
        publisher    = {github},
        version      = {v0.0.1},
        url          = {https://github.com/FSoft-AI4Code/code-llm-evaluator}
    }

