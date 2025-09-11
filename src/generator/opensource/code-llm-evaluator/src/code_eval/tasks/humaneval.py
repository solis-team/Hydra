_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""
import sys
sys.path.append("..")
from typing import List, Optional

from datasets import Dataset

from code_eval.tasks.base import TaskBase


class HumanEval(TaskBase):
    """HumanEval benchmark loader."""
    TASK_NAME = "humaneval"
    DATASET_NAME_OR_PATH = 'codeparrot/instructhumaneval'
    def __init__(self, 
        inst_token: Optional[str]="",
        assist_token: Optional[str]="",
        mode: Optional[str]=None) -> None:
        
        self.stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>", "\n>>>"]
        super().__init__()

        self.k = [1, 10, 100]
        self.mode = mode        
        self.dataset = self.dataset["test"]  # split = "test"
        self.inst_token = inst_token
        self.assist_token = assist_token
        
    
    def prepare_dataset(self, **kwargs) -> Dataset:
        """Preprocess HumanEval datasets. With different mode `instruct`, 
        `instruct-no-context`, and normal humaneval.
        
        Detail return input of each mode:
        - Default: 
            `<instruction_token> + <prompt> + <assistant_token>`
        - Mode `instruct`: 
            `<instruction_token> + <instruction> + <assistant_token> + <context>`
        - Mode `instruct-no-context`: 
            `<instruction_token> + <instruction> + <assistant_token>`
        """
        
        context_column = ""
        if self.mode == "instruct":
            key_column = "instruction"
            context_column = "context"
        elif self.mode == "instruct-no-context":
            key_column = "instruction"
        else:
            key_column = "prompt"
            
        def _preprocess(examples):
            model_inputs = dict(task_id=[], question=[])
            
            for idx in range(len(examples[key_column])):
                question = examples[key_column][idx]
                context = examples[context_column][idx] if context_column else ""
                
                # MODEL INPUTS HERE
                model_inputs['question'].append(
                    self.inst_token + question + self.assist_token + context
                )
            
            model_inputs['task_id'] = examples['task_id']
            
            return model_inputs
        
        preprossed_ds = self.dataset.map(_preprocess, batched=True,
                                        remove_columns=self.dataset.column_names)
        
        return preprossed_ds
    
    def postprocess_generation(self, generation):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)
    
    def compute_metrics(self):
        """Execute generated output to compute results"""
        pass