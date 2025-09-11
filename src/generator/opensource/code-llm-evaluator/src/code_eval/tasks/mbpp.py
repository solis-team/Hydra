_CITATION = """

"""
import sys
sys.path.append("..")
from typing import List, Optional

from datasets import Dataset

from code_eval.tasks.base import TaskBase


class MBPP(TaskBase):
    """MBPP benchmark loader."""
    TASK_NAME = "mbpp"
    DATASET_NAME_OR_PATH = 'mbpp'
    
    def __init__(self, 
        inst_token: Optional[str]="",
        assist_token: Optional[str]="",
        mode: Optional[str]=None) -> None:
        
        self.stop_words = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]
        super().__init__()

        self.k = [1, 10, 100]
        self.mode = mode        
        self.dataset = self.dataset["test"]  # split = "test"
        self.inst_token = inst_token
        self.assist_token = assist_token
        
    
    def prepare_dataset(self, **kwargs) -> Dataset:
        """Preprocess MBPP datasets. With different mode "no-context" and normal.
        
        Supported `mode`:
        - Default: 
        ```question = <instruction_token> + <prompt> + <assistant_token> + <context>```
        - Mode `no-context`: 
        ```question = <instruction_token> + <prompt> + <assistant_token> ```
        """
        
        key_column = "text"
        context_column = "test_list"
        if self.mode == "no-context":
            context_column = ""
            
        def _preprocess(examples):
            TEMPLATE = '"""\n{}\n"""'
            model_inputs = dict(task_id=[], question=[])
            
            for idx in range(len(examples[key_column])):
                question = examples[key_column][idx]
                context = examples[context_column][idx] if context_column else ""
                
                context = "\n".join(context)
                
                # MODEL INPUTS HERE
                model_inputs['question'].append(TEMPLATE.format(
                    self.inst_token + question + context + self.assist_token
                ))
            
            model_inputs['task_id'] = examples['task_id']
            
            return model_inputs
        
        preprossed_ds = self.dataset.map(_preprocess, batched=True,
                                        remove_columns=self.dataset.column_names)
        
        return preprossed_ds
    
    def compute_metrics(self):
        """Execute generated output to compute results"""
        pass