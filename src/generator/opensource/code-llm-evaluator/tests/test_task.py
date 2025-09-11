'''Test task'''
import os
import unittest
from src.code_eval.tasks import HumanEval

class TestHumanEval(unittest.TestCase):
    modes = [None, 'instruct', 'instruct-no-context']
    def test_loader(self):
        # test download -> create new cache dir
        os.environ['TRANSFORMERS_CACHE'] = "./tmp"
        task = HumanEval()
        print(task)
        
    def test_mode_process(self):
        for mode in self.modes:
            task = HumanEval(mode=mode)
            processed_ds = task.prepare_dataset()
            
            for i in range(3):
                print(processed_ds[i]['question'])
            
            self.assertIsNotNone(processed_ds)
    
    def test_add_inst_token(self):
        inst_token = "<inst_token>"
        assist_token = "<assist_token>"
        
        task = HumanEval(inst_token=inst_token,
                         assist_token=assist_token)
        
        processed_ds = task.prepare_dataset()
        for i in range(3):
            print(processed_ds[i]['question'])
        
        