from pathlib import Path
import json
import subprocess
import psutil
from subprocess import run
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser
import textwrap
from func_timeout import func_set_timeout
import func_timeout
import shutil
import argparse
import fcntl
from convert import convert_format_with_data_file
import time

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--log_file', type=Path)
    parser.add_argument('--source_code_root', type=Path, default=Path('Source_Code'))
    parser.add_argument('--data_file', type=Path, default=Path('data.jsonl')) # data.jsonl
    parser.add_argument('--k', type=str, default='1,3,5,10') # k in pass_at_k
    parser.add_argument('--n', type=int, default=1) # number of completions per task
    return parser.parse_args()


# def adjust_indent(code, new_indent):
#     dedented_code = textwrap.dedent(code)
#     indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
#     return indented_code

@func_set_timeout(30)
def execution_tests(args, data):
    project_path = os.path.join(args.source_code_root, data['project_path'])
    command = ['python', 'setup.py', 'pytest', '--addopts']    
    all_tests = data['tests']
    failed_tests = []
    passed_tests = []
    
    for test in all_tests:
        process = subprocess.Popen(command + [test], cwd=project_path, 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                 text=True)
        try:
            while True:
                process_id = process.pid
                process_memory = psutil.Process(process_id).memory_info().rss
                if process_memory > 5 * 1024 * 1024 * 1024: 
                    process.terminate()
                    process.wait()
                    failed_tests.append(f"{test}: OOM - Out of Memory (exceeded 5GB)")
                    break
                
                return_code = process.poll()
                if return_code is not None:
                    stdout, stderr = process.communicate()
                    
                    if return_code != 0:
                        error_content = f"Test: {test}\n"
                        if stderr.strip():
                            error_content += f"STDERR:\n{stderr}\n"
                        if stdout.strip():
                            error_content += f"STDOUT:\n{stdout}\n"
                        
                        full_output = stderr + stdout
                        if "Traceback" in full_output:
                            lines = full_output.split('\n')
                            traceback_lines = []
                            in_traceback = False
                            for line in lines:
                                if "Traceback" in line:
                                    in_traceback = True
                                if in_traceback:
                                    traceback_lines.append(line)
                                    if line.strip() and not line.startswith(' ') and any(x in line for x in ['Error:', 'Exception:', 'AssertionError:', 'ValueError:', 'TypeError:']):
                                        break
                            
                            if traceback_lines:
                                error_content = f"TRACEBACK for {test}:\n" + '\n'.join(traceback_lines) + f"\n\nFULL_OUTPUT:\n{error_content}"
                        
                        failed_tests.append(error_content)
                        break
                    else:
                        passed_tests.append(test)
                        break
                        
        except Exception as e:
            error_content = f"EXCEPTION in execution_tests for {test}:\n"
            error_content += f"Exception: {str(e)}\n"
            error_content += f"Exception type: {type(e).__name__}\n"
            
            import traceback
            error_content += f"Python traceback:\n{traceback.format_exc()}"
            
            failed_tests.append(error_content)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    

    if len(failed_tests) == 0:
        return 'Pass', f'All {len(passed_tests)} tests passed: {", ".join(passed_tests)}'
    else:
        error_summary = f"Failed {len(failed_tests)} out of {len(all_tests)} tests:\n"
        error_summary += "\n" + "="*50 + "\n".join(failed_tests)
        if len(passed_tests) > 0:
            error_summary += f"\n\nPassed tests: {', '.join(passed_tests)}"
        return 'Error', error_summary


def compute_pass_at_k(n, c, k):
    """
    n: total number of completions per task
    c: number of completions that pass all tests
    k: k in pass_at_k
    """
    if n - c < k:
        return 1
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))


def SetUp_evaluation(args, data, completion):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_backup_' + head_tail[1])

    if not os.path.exists(completion_path):
        raise FileNotFoundError(f"Source file not found: {completion_path}")
    
    try:
        shutil.copy2(completion_path, completion_tmp_path)
    except Exception as e:
        raise Exception(f"Failed to create backup of {completion_path}: {e}")

    try:
        with open(completion_path, 'r', encoding='utf-8') as f:
            file_lines = f.readlines()
    except Exception as e:
        if os.path.exists(completion_tmp_path):
            shutil.copy2(completion_tmp_path, completion_path)
            os.remove(completion_tmp_path)
        raise Exception(f"Failed to read file {completion_path}: {e}")

    sos, eos = data['body_position'][0]-1, data['body_position'][1]
    file_lines = file_lines[:sos] + ['\n', completion, '\n'] + file_lines[eos:]
    try:
        with open(completion_path, 'w', encoding='utf-8') as f:
            f.write(''.join(file_lines))
    except Exception as e:
        if os.path.exists(completion_tmp_path):
            shutil.copy2(completion_tmp_path, completion_path)
            os.remove(completion_tmp_path)
        raise Exception(f"Failed to write modified content to {completion_path}: {e}")



def TearDown_evaluation(args, data):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_backup_' + head_tail[1])
    i = 1
    while i < 2:
        if i == 1:
            i += 1
            continue
        if os.path.exists(completion_tmp_path):
            try:
                shutil.copy2(completion_tmp_path, completion_path)
                os.remove(completion_tmp_path)
            except Exception as e:
                print(f"Warning: Could not restore file {completion_path} from backup: {e}")
                try:
                    if os.path.exists(completion_tmp_path):
                        shutil.move(completion_tmp_path, completion_path)
                except Exception as e2:
                    print(f"Critical: Failed to restore {completion_path}: {e2}")
                    try:
                        if os.path.exists(completion_tmp_path):
                            os.remove(completion_tmp_path)
                    except:
                        pass
        else:
            print(f"Warning: Backup file not found: {completion_tmp_path}")
        



def check_correctness(args, data):
    completion = data['completion']
    if completion == "    pass\n":
        return 'Error', 'Empty completion - only pass statement'
    #completion = adjust_indent(completion, data['indent'])
    
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    original_content = None
    try:
        with open(completion_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        return 'Error', f'Cannot read original file {completion_path}: {str(e)}'
    
    try:
        SetUp_evaluation(args, data, completion)
    except Exception as e:
        return 'Error', f'Setup failed: {str(e)}'
    
    flag, content = 'Error', 'Unknown error'
    try:
        result = execution_tests(args, data)
        if isinstance(result, tuple):
            flag, content = result
        else:
            flag, content = result, ''
    except func_timeout.exceptions.FunctionTimedOut:
        flag, content = 'TimeOut', 'Function execution timed out after 30 seconds'
    except Exception as e:
        flag, content = 'Error', f'Test execution failed: {str(e)}'
    finally:
        try:
            TearDown_evaluation(args, data)
            try:
                with open(completion_path, 'r', encoding='utf-8') as f:
                    restored_content = f.read()
                if restored_content != original_content:
                    with open(completion_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
            except Exception as verify_e:
                print(f"Cannot verify restoration of {completion_path}: {verify_e}")
                if original_content:
                    try:
                        with open(completion_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)
                        print(f"Emergency restore of {completion_path} from memory backup")
                    except Exception as emergency_e:
                        print(f"Emergency restore failed for {completion_path}: {emergency_e}")

        except Exception as e:
            print(f"TearDown failed for {data.get('namespace', 'unknown')}: {e}")
            if original_content:
                try:
                    with open(completion_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    print(f"Emergency restore successful for {completion_path}")
                except Exception as emergency_e:
                    print(f"Emergency restore failed for {completion_path}: {emergency_e}")
    
    return flag, content
    


def report_results(args, benchmark_data):
    if not os.path.exists(args.log_file):
        raise ValueError(f'{args.log_file} does not exist')
    passed_completion = {}
    with open(args.log_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            if 'pass' in js:
                js['Result'] = js['pass']
            if js['Result'] == 'Pass':
                namespace, completion = js['namespace'], js['completion']
                if namespace not in passed_completion:
                    passed_completion[namespace] = set()
                passed_completion[namespace].add(completion)

    results = {}
    generation_file = args.output_file.parent / 'generation.jsonl'
    with open(generation_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in benchmark_data:
                continue
            if namespace not in results:
                results[namespace] = 0
            if namespace in passed_completion and completion in passed_completion[namespace]:
                results[namespace] += 1
            
    k_list = [int(k) for k in args.k.split(',')]
    for k in k_list:
        if k > args.n:
            continue
        pass_at_k = np.mean([compute_pass_at_k(args.n, pass_num, k) for namespace, pass_num in results.items()])
        print(f'pass_at_{k}: {pass_at_k*100}%')


def load_finished_data(args):
    finished_data = {}
    if os.path.exists(args.log_file):   
        with open(args.log_file, 'r') as f:
            for line in f:
                js = json.loads(line)
                namespace, idx = js['namespace'], js['idx']
                if namespace not in finished_data:
                    finished_data[namespace] = set()
                finished_data[namespace].add(idx)
    return finished_data


def main(args):
    finished_data = load_finished_data(args)

    todo_output_data = []
    generation_file = args.output_file.parent / 'generation.jsonl'
    if not generation_file.exists():
        print(f"Converting {args.output_file} to {generation_file}...")
        convert_format_with_data_file(str(args.output_file), str(args.data_file), str(generation_file))
        
    time.sleep(5)

    print(f"Finished data has {len(finished_data)} namespaces")
    total_finished = sum(len(completions) for completions in finished_data.values())
    print(f"Total finished completions: {total_finished}")
    
    with open(generation_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, idx = js['namespace'], js['idx']
            if namespace not in finished_data:
                todo_output_data.append(js)
                finished_data[namespace] = set()
                finished_data[namespace].add(idx)
            elif idx not in finished_data[namespace]:
                todo_output_data.append(js)
                finished_data[namespace].add(idx)
    #del finished_data
    print("TODO Completions: ", len(todo_output_data))
    
    benchmark_data = {}
    with open(args.data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js

    with open(args.log_file, 'a') as f:
        for output in tqdm(todo_output_data):
            if output['namespace'] in benchmark_data:
                data = benchmark_data[output['namespace']]
                data['completion'] = output['completion']
                result = check_correctness(args, data)
                
                if isinstance(result, tuple):
                    flag, content = result
                else:
                    flag, content = result, ''
                
                output['Result'] = flag
                output['content'] = content  
                f.write(json.dumps(output) + '\n')
                f.flush()

    report_results(args, benchmark_data)


def test_ground_truth(args):
    data = open(args.data_file, 'r').readlines()
    output_f = open('samples.jsonl', 'w')

    for line in tqdm(data):
        js = json.loads(line)
        tests = set(js['tests'])
        js['tests'] = list(tests)
        try:
            result = execution_tests(args, js)
            if isinstance(result, tuple):
                flag, content = result
            else:
                flag, content = result, ''
        except func_timeout.exceptions.FunctionTimedOut:
            flag, content = 'TimeOut', 'Function execution timed out after 30 seconds'
        
        if flag == 'Error':
            print(js['namespace'])
            js['Result'] = flag
            js['content'] = content
            output_f.write(json.dumps(js) + '\n')


if __name__ == '__main__':
    args = get_parser()
    if args.output_file is None:
        test_ground_truth(args)
    else:
        main(args)