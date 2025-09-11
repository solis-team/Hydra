from .base import TaskBase
from .humaneval import HumanEval
from .mbpp import MBPP

SUPPORTED_TASKS = ["humaneval", "multiple", "mbpp"]

__all__ = ["TaskBase", "HumanEval", "MBPP"]