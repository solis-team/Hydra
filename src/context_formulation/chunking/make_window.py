import code
import itertools
import functools
from socket import SOL_ALG

from utils import Tools, FilePathBuilder, CONSTANTS
from collections import defaultdict
import json
import os

def get_repos(base_dir, benchmark):        

    if benchmark == "RepoExec":
        return os.listdir(base_dir)

    if benchmark == "DevEval":
        categories = os.listdir(base_dir)
        repos = []
        for category in categories:
            category_path = os.path.join(base_dir, category)
            for repo in os.listdir(category_path):
                repo_path = os.path.join(category, repo)
                repos.append(repo_path)
        return repos
    return []

class RepoWindowMaker:
    def __init__(self,repo, window_size, slice_size, repo_base_dir=FilePathBuilder.repo_base_dir) -> None:
        self.repo = repo
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = 1 if window_size // slice_size == 0 else window_size // slice_size
        self.repo_base_dir = repo_base_dir
        self.source_code_files = Tools.iterate_repository(repo, repo_base_dir)
        
    def _buid_windows_for_a_file(self, fpath_tuple, code):
        code_windows = []
        code_lines = code.splitlines()
        delta_size = self.window_size // 2
        for line_no in range(0, len(code_lines), self.slice_step): # line_no starts from 0
            start_line_no = max(0, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            if not window_lines:  # all empty lines
                continue
            window_text = '\n'.join(window_lines)
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'repo': self.repo,
                    'slice_size': self.slice_size,
                }
            })
        return code_windows
    
    def _merge_windows_with_same_context(self, code_windows):
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, metadata_list in merged_code_windows.items():
            json_lines.append({
                'context': context,
                'metadata': metadata_list
            })
        return json_lines

    def build_windows(self):
        all_code_windows = []
        for fpath_tuple, code in self.source_code_files.items():
            all_code_windows += self._buid_windows_for_a_file(fpath_tuple, code)
        merged_code_windows = self._merge_windows_with_same_context(all_code_windows)
        print(f'build {len(merged_code_windows)} windows for {self.repo} with window size {self.window_size} and slice {self.slice_size}')
        output_path = "cache/{}/window/repos/{}_ws{}_ss{}.jsonl".format(self.repo_base_dir, self.repo, self.window_size, self.slice_size)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Tools.dump_pickle(merged_code_windows, output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in merged_code_windows:
                f.write(json.dumps(sample) + '\n')

        
        output_path = FilePathBuilder.repo_windows_path(self.repo_base_dir, self.repo, self.window_size, self.slice_size)
        Tools.dump_pickle(merged_code_windows, output_path)

class CurrentFileWindowMaker:
    def __init__(self, idx, solution, fpath_tuple, repo, window_size, slice_size, solution_position = None, repo_base_dir=FilePathBuilder.repo_base_dir):
        assert solution is not None or solution_position is not None, "Either solution or solution_position must be provided"
        self.idx = idx
        self.solution = solution
        self.fpath_tuple = fpath_tuple
        self.repo = repo
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = 1 if window_size // slice_size == 0 else window_size // slice_size
        self.solution_position = solution_position
        self.base_dir = repo_base_dir
        self.fpath = os.path.join(self.base_dir, *fpath_tuple)
        self.source_code = Tools.read_code(self.fpath)

    def _merge_windows_with_same_context(self, code_windows):
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, metadata_list in merged_code_windows.items():
            json_lines.append({
                'context': context,
                'metadata': metadata_list
            })
        return json_lines
    
    def build_window(self):
        if self.solution_position:
            code_lines = self.source_code.splitlines()
            code_lines = code_lines[:self.solution_position[0] - 1] + code_lines[self.solution_position[1]:]
            if not code_lines:
                print(f"Solution position {self.solution_position} is out of range for the code in file {self.fpath}. ID: {self.idx}")
        else:
            if self.solution in self.source_code:
                print("Solution found.")
                self.source_code = self.source_code.replace(self.solution, "")
            else:
                print(f"Solution not found in the code. ID: {self.idx}")
            code_lines = self.source_code.splitlines()
        delta_size = self.window_size // 2
        code_windows = []
        for line_no in range(0, len(code_lines), self.slice_step): # line_no starts from 0
            start_line_no = max(0, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            if not window_lines:  # all empty lines
                continue
            window_text = '\n'.join(window_lines)
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath_tuple': self.fpath_tuple,
                    'line_no': line_no,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'repo': self.repo,  # repo is the first element of fpath_tuple
                    'slice_size': self.slice_size,
                }
            })
        merged_code_windows = self._merge_windows_with_same_context(code_windows)
        print(f'Build {len(merged_code_windows)} windows for sample {self.idx} with window size {self.window_size} and slice {self.slice_size}')
        output_path = "cache/{}/window/current-files/{}_ws{}_ss{}.jsonl".format(self.base_dir, self.idx, self.window_size, self.slice_size)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Tools.dump_pickle(merged_code_windows, output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in merged_code_windows:
                f.write(json.dumps(sample) + '\n')

class BaselineWindowMaker:
    '''the retrieve-and-generate approach'''
    def __init__(self, benchmark, repo, window_size, tasks, repo_base_dir=FilePathBuilder.repo_base_dir):
        self.benchmark = benchmark
        self.repo = repo
        self.window_size = window_size
        self.tasks = tasks
        self.source_code = Tools.iterate_repository(repo, repo_base_dir)
        self.repo_base_dir = repo_base_dir

    def build_window(self):
        code_windows = []
        for task in self.tasks:
            task_id_tuple = task['metadata']['task_id'].split('/')
            if "/".join(task_id_tuple[:-1]) != self.repo:
                continue
            fpath_tuple = tuple(task['metadata']['fpath_tuple'])
            line_no = task['metadata']['line_no']
            original_code = self.source_code[fpath_tuple]
            code_lines = original_code.splitlines()
            context_start_lineno = task['metadata']['context_start_lineno']
            start_line_no = max(context_start_lineno, line_no - self.window_size)
            window_lines = [i for i in code_lines[start_line_no:line_no]]
            code_windows.append({
                'context': '\n'.join(window_lines),
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,  # line_no starts from 0
                    'task_id': task['metadata']['task_id'],
                    'start_line_no': start_line_no,
                    'end_line_no': line_no,
                    'window_size': self.window_size,
                    'context_start_lineno': context_start_lineno,
                    'repo': self.repo,
                    'target_function_prompt': task['metadata']['target_function_prompt'],
                    'function_signature': task['metadata']['function_signature'],
                    'id': task['metadata']['id']
                }
            })
        print(f'build {len(code_windows)} baseline windows for {self.repo} with window size {self.window_size}')
        output_path = FilePathBuilder.search_first_window_path(self.repo_base_dir, CONSTANTS.rg, self.repo, self.window_size)
        Tools.dump_pickle(code_windows, output_path)

class GroundTruthWindowMaker:
    '''Use for oracle evaluation'''
    def __init__(self, benchmark, repo, window_size, tasks, repo_base_dir=FilePathBuilder.repo_base_dir):
        self.benchmark = benchmark
        self.repo = repo
        self.window_size = window_size
        self.tasks = tasks
        self.source_code = Tools.iterate_repository(repo, repo_base_dir)
        self.repo_base_dir = repo_base_dir

    def build_window(self):
        code_windows = []
        delta_size = self.window_size // 2
        for task in self.tasks:
            task_id_tuple = task['metadata']['task_id'].split('/')
            if "/".join(task_id_tuple[:-1]) != self.repo:
                continue
            fpath_tuple = tuple(task['metadata']['fpath_tuple'])
            line_no = task['metadata']['line_no']
            original_code = self.source_code[fpath_tuple]
            code_lines = original_code.splitlines()
            context_start_lineno = task['metadata']['context_start_lineno']
            start_line_no = max(context_start_lineno, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            code_windows.append({
                'context': '\n'.join(window_lines),
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,  
                    'task_id': task['metadata']['task_id'],
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'context_start_lineno': context_start_lineno,
                    'repo': self.repo,
                    'target_function_prompt': task['metadata']['target_function_prompt'],
                    'function_signature': task['metadata']['function_signature'],
                    'id': task['metadata']['id']
                }
            })
        print(f'build {len(code_windows)} ground truth windows for {self.repo} with window size {self.window_size}')
        output_path = FilePathBuilder.search_first_window_path(self.repo_base_dir, CONSTANTS.gt, self.repo, self.window_size) # Old version: CONSTANTS.rg
        Tools.dump_pickle(code_windows, output_path)

class PredictionWindowMaker:
    def __init__(self, repo, window_size, prediction_path, window_path_builder, repo_base_dir=FilePathBuilder.repo_base_dir):
        self.repo = repo
        self.window_size = window_size
        self.prediction_path = prediction_path
        self.source_code = Tools.iterate_repository(repo, repo_base_dir)
        self.predictions = Tools.load_jsonl(prediction_path)
        self.window_path_builder = window_path_builder
    
    def build_window(self, type='centered'):
        code_windows = []
        delta_size = self.window_size // 2
        for prediction in self.predictions:
            task_id_tuple = prediction['metadata']['task_id'].split('/')
            if "/".join(task_id_tuple[:-1]) != self.repo:
                continue
            fpath_tuple = tuple(prediction['metadata']['fpath_tuple'])
            line_no = prediction['metadata']['line_no']  # line_no in prediction file starts from 0
            original_code = self.source_code[fpath_tuple]
            code_lines = original_code.splitlines()
            context_start_lineno = prediction['metadata']['context_start_lineno']
            start_line_no = max(context_start_lineno, line_no - delta_size)
            for sample in [prediction['choices'][i]['text'] for i in range(len(prediction['choices']))]:
                # TODO actually only one sample is generated
                sample_lines = [i for i in sample.splitlines() if i.strip()]
                new_code_lines = code_lines[:line_no] + sample_lines
                end_line_no = min(len(new_code_lines), line_no + self.window_size - delta_size)
                window_lines = [i for i in new_code_lines[start_line_no:end_line_no] if i.strip()]
                if not window_lines:  # all empty lines
                    continue
                code_windows.append({
                    'context': '\n'.join(window_lines),
                    'metadata': {
                        'fpath_tuple': fpath_tuple,
                        'line_no': line_no,  # line_no starts from 0
                        'prediction': sample,
                        'task_id': prediction['metadata']['task_id'],
                        'start_line_no': start_line_no,
                        'end_line_no': end_line_no,
                        'window_size': self.window_size,
                        'context_start_lineno': context_start_lineno,
                        'repo': self.repo
                    }
                })
        print(f'build {len(code_windows)} prediction windows for {self.repo} with window size {self.window_size}')
        output_path = self.window_path_builder(self.prediction_path, self.repo, self.window_size)
        Tools.dump_pickle(code_windows, output_path)

class MakeWindowWrapper:
    def __init__(self, benchmark, repos, window_sizes, slice_sizes, repo_base_dir=FilePathBuilder.repo_base_dir):
        self.repos = repos
        self.repo_base_dir = repo_base_dir
        self.window_sizes = window_sizes
        self.slice_sizes = slice_sizes

        self.benchmark = benchmark

        if benchmark == CONSTANTS.line_benchmark:
            self.task_file_path = FilePathBuilder.random_line_completion_benchmark
        elif benchmark == CONSTANTS.api_benchmark:
            self.task_file_path = FilePathBuilder.api_completion_benchmark
        elif benchmark == CONSTANTS.short_line_benchmark:
            self.task_file_path = FilePathBuilder.short_random_line_completion_benchmark
        elif benchmark == CONSTANTS.short_api_benchmark:
            self.task_file_path = FilePathBuilder.short_api_completion_benchmark
        else:
            self.task_file_path = benchmark

    def window_for_repo_files(self):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            for repo in self.repos:
                repo_window_maker = RepoWindowMaker(repo, window_size, slice_size, self.repo_base_dir)
                repo_window_maker.build_windows()
    
    def window_for_current_files(self):
        samples = Tools.load_jsonl(self.task_file_path)
        for sample in samples:
            idx = sample["metadata"]["id"]
            solution = sample["metadata"].get("ground_truth", None)
            solution_position = sample["metadata"].get("solution_position", None)
            if "fpath_tuple" in sample["metadata"].keys():
                fpath_tuple = sample["metadata"]["fpath_tuple"]
            else:
                project = sample["project"].split("/")[1:]
                module = sample["module"].split(".")
                fname = module.pop(-1) + ".py"
                module.append(fname)
                fpath_tuple = tuple(project + module)
            if "repo" in sample.keys():
                repo = sample["repo"]
            else:
                repo = fpath_tuple[0]
            for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
                current_file_window_maker = CurrentFileWindowMaker(idx, solution, fpath_tuple, repo, window_size, slice_size, solution_position, self.repo_base_dir)
                current_file_window_maker.build_window()

    def window_for_baseline_and_ground(self):
        tasks = Tools.load_jsonl(self.task_file_path)
        for window_size in self.window_sizes:
            for repo in self.repos:
                baseline_window_maker = BaselineWindowMaker(self.benchmark, repo, window_size, tasks, self.repo_base_dir)
                ground_window_maker = GroundTruthWindowMaker(self.benchmark, repo, window_size, tasks, self.repo_base_dir)
                baseline_window_maker.build_window()
                ground_window_maker.build_window()

    def window_for_prediction(self, mode, prediction_path_template):
        for window_size, slice_size in itertools.product(self.window_sizes, self.slice_sizes):
            prediction_path = prediction_path_template.format(window_size=window_size, slice_size=slice_size)
            for repo in self.repos:
                window_path_builder = functools.partial(FilePathBuilder.gen_first_window_path, self.repo_base_dir, mode)
                pred_window_maker = PredictionWindowMaker(repo, window_size, prediction_path, window_path_builder, self.repo_base_dir)
                pred_window_maker.build_window()



def run(benchmark_type):
    """Run the window making process for the specified benchmark type."""
    if benchmark_type == "RepoExec":
        base_dir = "benchmark/RepoExec/test-apps"
        benchmark_file = "data/temp/RepoExec_benchmark.jsonl"
        repo_base_dir = "benchmark/RepoExec/test-apps"
    elif benchmark_type == "DevEval":
        base_dir = "benchmark/DevEval/Source_Code"
        benchmark_file = "data/temp/DevEval_benchmark.jsonl"
        repo_base_dir = "benchmark/DevEval/Source_Code"
    else:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}. Must be 'RepoExec' or 'DevEval'.")

    repos = get_repos(base_dir=base_dir, benchmark=benchmark_type)

    if not repos:
        print(f"No repositories found in {repo_base_dir}. Please check the directory.")
        return
    else:
        print(f"Got {len(repos)} repos. Example: {repos[0]}")

    a = MakeWindowWrapper(
        benchmark=benchmark_file, 
        repos=repos, 
        window_sizes=[20], 
        slice_sizes=[2], 
        repo_base_dir=repo_base_dir
    )
    a.window_for_repo_files()
    a.window_for_current_files()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Make windows for benchmark data")
    parser.add_argument("--benchmark", type=str, choices=["RepoExec", "DevEval"], 
                       required=True, help="Benchmark type: RepoExec or DevEval")
    
    args = parser.parse_args()
    run(args.benchmark)

