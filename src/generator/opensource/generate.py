
from typing import Any
from string import ascii_uppercase
import argparse
import os
import json
from datasets import Dataset
import sys

current_dir = os.path.dirname(__file__)
code_eval_path = os.path.join(current_dir, 'code-llm-evaluator', 'src')
if code_eval_path not in sys.path:
    sys.path.insert(0, code_eval_path)

from code_eval import Evaluator
from code_eval.tasks.base import TaskBase
from transformers import AutoTokenizer



system_prompt = """You are a helpful coding assistant."""


class Benchmark(TaskBase):
    def __init__(self, task_name, dataset_path, model_name, system_prompt=None) -> None:
        self.TASK_NAME = task_name
        self.DATASET_NAME_OR_PATH = dataset_path
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        super().__init__()
   
    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:
        samples = []
        with open(self.DATASET_NAME_OR_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        dataset = Dataset.from_list(samples)

        def _preprocess(example):
            messages = []

            if self.system_prompt:
                messages = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                ]
            prompt = example["prompt"]
            messages.append({"role": "user", "content": prompt})
            example['question'] = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            example['task_id'] = example['id']
            return example
       
        updated_dataset = dataset.map(_preprocess)
        return updated_dataset
   
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, choices=["RepoExec", "DevEval"], required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--num_return_sequences", type=int, default=5)

    parser.add_argument('--do_sample', action="store_true")
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--has_example', action="store_true")
    parser.add_argument('--fc', action="store_true")
   
    opt = parser.parse_args()

    data_path = f"data/prompt/{opt.task_name}_prompt.jsonl"
    task_name = opt.task_name
    task = Benchmark(task_name=task_name, dataset_path=data_path, system_prompt=system_prompt,
               model_name=opt.model)

    save_dir = "data/generation"
    os.makedirs(save_dir, exist_ok=True)
    evaluator = Evaluator(task=task,
                        model_name=opt.model,
                        batch_size=opt.batch_size,
                        save_dir=save_dir,
                        cache_dir=opt.cache_dir,
                        peft_model = opt.lora_path, 
                        trust_remote_code=True)
   
    print("="*25 + "Test sample" + "="*25)
    print(evaluator.dataset['question'][0])
    print(len(evaluator.dataset['question']))
    print("="*61 )

    evaluator.generate(backend='vllm',
                    max_tokens=opt.max_tokens,
                    num_return_sequences=opt.num_return_sequences,
                    temperature=opt.temperature,
                    do_sample= opt.do_sample,
                    top_p = opt.top_p,
                    top_k = opt.top_k)