"""Evaluator to load dataset and generate output. Customized config for
generate output.

For example:

.. code-block:: python

    >>> from code_eval import Evaluator, HumanEval

    >>> task = HumanEval()
    >>> evaluator = Evaluator(task=task)

    >>> output = evaluator.generate(temperature=0.9, num_return_sequences=3)
    >>> result = evaluator.evaluate(output)

"""
import os
import sys
import json
import time
import torch.utils
import torch.utils.data
from tqdm import tqdm
from warnings import warn
from typing import Optional, Dict, List

import torch
from torch.utils.data import DataLoader

from code_eval.tasks.base import TaskBase
from code_eval.tasks.mbpp import MBPP

from accelerate import Accelerator, PartialState
from accelerate.utils import gather_object
from transformers import (
    HfArgumentParser,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    DataCollatorWithPadding
)

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE=True
except ImportError:
    VLLM_AVAILABLE=False


class Evaluator:
    """Evaluator class.

        :param task: Evaluation task loader
        :type task: TaskBase
        :param model_name: Selected model for evaluating
        :type model_name: str
        :param peft_model: Adapter model, defaults to None
        :type peft_model: Optional[str], optional
        :param trust_remote_code: Huggingface argument, defaults to False
        :type trust_remote_code: Optional[bool], optional
        :param cache_dir: Downloaded cache directory, defaults to None
        :type cache_dir: Optional[str], optional
        :param batch_size: Generation batch size, defaults to 16
        :type batch_size: Optional[int], optional
        :param save_dir: Saving generation directory, defaults to "./output"
        :type save_dir: Optional[str], optional
    """
   
    def __init__(self,
        task: TaskBase,
        model_name: str,
        peft_model: Optional[str] = None,
        trust_remote_code: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        batch_size: Optional[int] = 16,
        save_dir: Optional[str] = "./output"
    ) -> None:
        self.task = task
        self.TASK_NAME = task.TASK_NAME
        self.DATASET_NAME_OR_PATH = task.DATASET_NAME_OR_PATH
        self.compute_metrics = task.compute_metrics
        self.dataset = task.prepare_dataset()
        print(self.dataset)
       
        self.model_name = model_name
        self.peft_model = peft_model
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

   
    def generate(self,
        backend: Optional[str]="vllm",
        num_return_sequences: Optional[int]=1,
        max_tokens: Optional[int]=256,
        temperature: Optional[float]=0.9,
        top_p: Optional[float]=0.95,
        top_k: Optional[float]=0.,
        do_sample: Optional[bool]=True,
        repetition_penalty: Optional[float]=1.2
        ) -> List:
        """Start backend and generate output

        :param backend: backend to inference model.
            Choose between native ``tf`` (transformers) or ``vllm`` for fast infernce, defaults to "vllm"
        :type backend: Optional[str], optional
        :param num_return_sequences: Model generated n, defaults to 1
        :type num_return_sequences: Optional[int], optional
        :param max_tokens: Max new tokens, defaults to 256
        :type max_tokens: Optional[int], optional
        :param temperature: Model generate temperature, defaults to 0.9
        :type temperature: Optional[float], optional
        :param repetition_penalty: Repetition penalty, defaults to 1.2
        :type repetition_penalty: Optional[float], optional
        :raises NotImplementedError: _description_
        :return: List of generated result, stored in dictionary object
            with ``task_id``, ``prompt`` and ``answer`` key.
        :rtype: List
        """
        print(f"Evaluating task: [{self.TASK_NAME}]")
        print(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")
        print(f"device compute capabilities={torch.cuda.get_device_capability()}")
        print(f"pytorch compute capabilities={torch.cuda.get_arch_list()}")
       
        gen_config = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample = do_sample,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
       
        if backend == "vllm":
            assert VLLM_AVAILABLE, "vllm not installed, try `pip install vllm`"
            self._vllm_initialize(**gen_config)
            self._generate_fn = self._vllm_generate
       
        elif backend == "tf":
            self._distributed_initialize(**gen_config)
            self._generate_fn = self._distributed_generate

        else:
            raise NotImplementedError
       
        self.ds_loader = [self.dataset[i:i+self.batch_size]
                    for i in range(0, len(self.dataset), self.batch_size)]
       
        start_time = time.time()
        result = self._generate_fn()
       
        self.postprocessing(result)
       
        print("=======  Finished {}  =======".format(self.TASK_NAME))
        print("Completion time: %d s", (time.time() - start_time))
       
        # return results
       
    def postprocessing(self, outputs):
        save_path = os.path.join(self.save_dir, f"{self.TASK_NAME}.final.generated.jsonl")
       
        with open(save_path, "w") as writer:
            for idx in range(len(outputs['question'])):
                res = dict(
                    task_id=outputs['task_id'][idx],
                    prompt=outputs['question'][idx],
                    response=outputs['generation'][idx]
                )
           
                json.dump(res, writer)
                writer.write("\n")
   
    def save_result(self, batched_outputs: Dict):
        assert 'question' in batched_outputs.keys()
        assert 'generation' in batched_outputs.keys()
       
        try:
            if self.accelerator.distributed_type == "MULTI_GPU":
                save_path = os.path.join(self.save_dir,
                            f"{self.TASK_NAME}.raw.generated.{self.accelerator.process_index}.jsonl")
            else:
                save_path = os.path.join(self.save_dir, f"{self.TASK_NAME}.final.generated.jsonl")
        except (KeyError, AttributeError):
            save_path = os.path.join(self.save_dir, f"{self.TASK_NAME}.final.generated.jsonl")
       
        with open(save_path, "a") as writer:
            for idx in range(len(batched_outputs['question'])):
                res = dict(
                    task_id=batched_outputs['task_id'][idx],
                    prompt=batched_outputs['question'][idx],
                    response=batched_outputs['generation'][idx]
                )
           
                json.dump(res, writer)
                writer.write("\n")
   
    def _distributed_initialize(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: float,
        do_sample: bool,
        repetition_penalty: float,
        num_return_sequences: int):
        """Initilize native transformers backend"""
        self.accelerator = Accelerator()
        generate_args = dict(
            max_new_tokens=max_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample = do_sample,
            repetition_penalty=repetition_penalty
        )
        self.generation_config = GenerationConfig(**generate_args)
       
        model_kwargs = dict(
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            load_in_8bit=False
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs)
           
        except KeyError: # TODO: except load seq2seq model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **model_kwargs)
       
        if self.peft_model:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.peft_model)
       
        self.model.to(self.accelerator.device)
       
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left"
        )
       
        if not self.tokenizer.pad_token:
            print("Set EOS_TOKEN to PAD_TOKEN")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _distributed_generate(self):
        # ``Accelerate`` distribute data and model
        assert self.accelerator
       
        for i in range(len(self.ds_loader)):
            question = self.ds_loader[i]['question']
            self.ds_loader[i]['question_ids'] = self.tokenizer(question, return_tensors="pt", padding=True)
       
        result = []
        with self.accelerator.split_between_processes(self.ds_loader, apply_padding=True) as batched_prompts:
            index = self.accelerator.process_index
            for batch in tqdm(batched_prompts, desc=f"Process: {index} | Generating", position=index):
                input_ids = batch['question_ids'].to(self.accelerator.device)
                outputs = self.model.generate(**input_ids,
                                            generation_config=self.generation_config,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            eos_token_id=self.tokenizer.eos_token_id)
               
                outputs = [output[len(prompt) :] for prompt, output in zip(input_ids["input_ids"], outputs)]
                batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
               
                batch['generation'] = batch_results
                result.extend(batch['generation'])
                self.save_result(batch)
               
       
        result_gather = gather_object(result)[: len(self.dataset)]
        self.dataset = self.dataset.add_column('generation', result_gather)
        return self.dataset

    def _vllm_initialize(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: float,
        do_sample: bool,
        repetition_penalty: float,
        num_return_sequences: int):
        """Initialize vllm backend

        :return: vLLM's model, sampling parameters and lora config
        :rtype: set
        """
        ngpus = torch.cuda.device_count()
        backend_kwargs = dict(
            disable_log_stats=True,
            tensor_parallel_size=ngpus,
            download_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
       
        self.model = LLM(self.model_name,
            enable_lora=True if self.peft_model else None,
            **backend_kwargs)
       
        self.lora_request = None
        if self.peft_model:
            self.lora_request=LoRARequest("lora", 1, self.peft_model)
           
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            n=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left"
        )
       
        return (self.model, self.sampling_params, self.lora_request)
   
    def _vllm_generate(self):
        result = []
        for batch in tqdm(self.ds_loader, total=len(self.ds_loader), desc="Generating"):
            for qid, q in enumerate(batch['question']):
                if len(self.tokenizer.encode(q)) > (self.tokenizer.model_max_length - 2):
                    batch['question'][qid] = self.tokenizer.decode(self.tokenizer.encode(q)[-(self.tokenizer.model_max_length-self.sampling_params.max_tokens):])

            outputs = self.model.generate(batch['question'],
                                          self.sampling_params,
                                          lora_request=self.lora_request)

            batch['generation'] = [[x.text for x in output.outputs] for output in outputs]
            result.extend(batch['generation'])
            self.save_result(batch)
           
        self.dataset = self.dataset.add_column('generation', result)
        return self.dataset
       
           
    def evaluate():
        pass
   