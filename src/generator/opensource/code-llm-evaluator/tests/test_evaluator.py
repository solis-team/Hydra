import unittest
from src.code_eval.evaluator import Evaluator
from src.code_eval.tasks import HumanEval

unittest.TestLoader.sortTestMethodsUsing = None # forge to execute by order

class TestEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.task = HumanEval()
        self.model_name = "microsoft/phi-1"
        self.batch_size = 16
        return super().setUp()
    
    def test_evaluator_init(self):
        evaluator = Evaluator(self.task,
                              model_name=self.model_name,
                              batch_size=self.batch_size)

        return evaluator
        
    def test_evaluator_generate(self):
        # trick to gen 5 first test
        evaluator = self.test_evaluator_init()
        
        evaluator.dataset = evaluator.dataset.select(range(5))
        output = evaluator.generate()
    
    def test_evaluator_compute_metric(self):
        pass