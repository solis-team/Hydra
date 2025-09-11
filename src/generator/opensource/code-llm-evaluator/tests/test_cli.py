import unittest
import subprocess


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_cli(self):
        subprocess.run(["bash", "script/run.sh"])
        # TODO: assert result