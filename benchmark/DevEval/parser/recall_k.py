import json
from subprocess import run
from tqdm import tqdm
import os
from argparse import ArgumentParser
import textwrap
from add_func_call import process


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--log_file', type=str) 
    parser.add_argument('--k', type=str)
    parser.add_argument('--source_code_root', type=str)
    parser.add_argument('--dependency_data_root', type=str)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--dependency_tmp_dir', type=str, default='dependency_data_tmp')
    return parser.parse_args()

def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code


def compute_recall(generated_dependency, reference_dependency):
    reference = []
    for _type, _list in reference_dependency.items():
        reference.extend(_list)
    if generated_dependency is None:
        return 0
    prediction = []
    for _type, _list in generated_dependency.items():
        prediction.extend(_list)
    reference = set(reference)
    prediction = set(prediction)
    recall = len(reference.intersection(prediction)) / len(reference)
    return recall
    

def report_results(args, k_list, output_data, benchmark_data):
    if not os.path.exists(args.log_file):
        raise ValueError("Output file not found")
    
    parse_results = dict()
    with open(args.log_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in parse_results:
                parse_results[namespace] = dict()
            parse_results[namespace][completion] = js['generated_dependency']

    results = {}
    for namespace, outputs in output_data.items():
        for output in outputs:
            completion = output['completion']
            if namespace in parse_results and namespace in benchmark_data:
                generated_dependency = parse_results[namespace][completion]
                data = benchmark_data[namespace]
                reference_dependency = data['dependency']
                recall = compute_recall(generated_dependency, reference_dependency)
                if namespace not in results:
                    results[namespace] = []
                results[namespace].append(recall)

    for k in k_list:
        recall = 0
        for namespace, recall_list in results.items():
            recall += max(recall_list[:k])
        recall /= len(results) # average the recall of samples
        print(f"Recall@{k}: {recall*100}%\n")


def SetUp_evaluation(args, data):
    completion = adjust_indent(data['completion'], data['indent'])
    completion_path = os.path.join(args.source_code_root, data['completion_path'])
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])

    # rename the original completion file as tmp_completion
    run(['cp', completion_path, completion_tmp_path])

    # write the new completion file
    sos, eos = data['body_position'][0]-1, data['body_position'][1]
    file_lines = []
    with open(completion_path, 'r') as f:
        file_lines = f.readlines()
    file_lines = file_lines[:sos] + ['\n', completion, '\n'] + file_lines[eos:]
    with open(completion_path, 'w') as f:
        f.write(''.join(file_lines))


def parse_dependency(args, data):
    project_root = os.path.join(args.source_code_root, data['project_path'])
    file_to_parse = os.path.join(args.source_code_root, data['completion_path'])
    output_path = os.path.join(args.dependency_tmp_dir, data['project_path'])
    analyzer_result_path = os.path.join(args.dependency_data_root, data['project_path'], 'analyzer_result.pkl')
    try:
        process(target_object=project_root, func_object_root=project_root, func_path=file_to_parse,
                analyzer_result=analyzer_result_path, target_root=output_path)
    except Exception as e:
        return False
    return True


def extract_dependency(args, data):
    generated_dependency = {'intra_class': [], 'intra_file': [], 'cross_file': []}
    dependency_path = os.path.join(args.dependency_tmp_dir, data['completion_path'].replace('.py', '.json'))
    if not os.path.exists(dependency_path):
        return generated_dependency
    with open(dependency_path, 'r') as f:
        dependency_data = json.load(f)
    if data['namespace'] not in dependency_data:
        return generated_dependency
    attributes = dependency_data[data['namespace']]
    for _item in attributes['in_class']:
        generated_dependency['intra_class'].append(_item['name'])
    for _item in attributes['in_file']:
        generated_dependency['intra_file'].append(_item['name'])
    for _item in attributes['in_object']:
        generated_dependency['cross_file'].append(_item['name'])
    return generated_dependency


def TearDown_evaluation(args, data):
    completion_path = os.path.join(args.source_code_root, data['completion_path'])
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    dependency_tmp_path = os.path.join(args.dependency_tmp_dir, data['project_path'])

    run(['mv', completion_tmp_path, completion_path])
    run(['rm', '-rf', dependency_tmp_path])


def is_standalone(data):
    dependency = data['dependency']
    if len(dependency['intra_class']) + len(dependency['intra_file']) + len(dependency['cross_file']) == 0:
        return True
    return False


def load_finished_data(args):
    finished_data = dict()
    if os.path.exists(args.log_file):
        with open(args.log_file, 'r') as f:
            for line in f:
                js = json.loads(line)
                if js['namespace'] not in finished_data:
                    finished_data[js['namespace']] = set()
                finished_data[js['namespace']].add(js['completion'])
    return finished_data


def main():
    args = get_parser()

    # Parse the k values
    k_list = []
    for _k in args.k.split(','):
        k_list.append(int(_k))
    max_k = max(k_list)

    # Load the completion data and finished data
    finished_data = load_finished_data(args)
    output_data = dict()
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            if namespace not in output_data:
                output_data[namespace] = []
            if len(output_data[namespace]) < max_k: # only consider max_k completions
                output_data[namespace].append(js)
    
    benchmark_data = {}
    with open(args.data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js
    
    # Skip the finished data, deuplicate completions, and standalone completions
    todo_output_data = []
    for namespace, outputs in output_data.items():
        assert len(outputs) == max_k, print(len(outputs))
        if namespace in benchmark_data:
            data = benchmark_data[namespace]
            if is_standalone(data):
                continue
            for output in outputs:   # only consider max_k completions
                completion = output['completion']
                if namespace not in finished_data:
                    todo_output_data.append(output)
                    finished_data[namespace] = set()
                    finished_data[namespace].add(completion)
                elif completion not in finished_data[namespace]:
                    todo_output_data.append(output)
                    finished_data[namespace].add(completion)
    print(f"TODO Completions: {len(todo_output_data)}\n")

    # release memory
    del finished_data
            
    with open(args.log_file, 'a') as f:
        for output in tqdm(todo_output_data):
            if output['completion'] == "    pass\n":
                output['generated_dependency'] = None
            else:
                data = benchmark_data[output['namespace']]
                data['completion'] = output['completion']
                SetUp_evaluation(args, data)
                if parse_dependency(args, data) == True:
                    generated_dependency = extract_dependency(args, data)
                    output['generated_dependency'] = generated_dependency
                else:
                    output['generated_dependency'] = None
                TearDown_evaluation(args, data)
            f.write(json.dumps(output) + '\n')
            f.flush()
    
    report_results(args, k_list, output_data, benchmark_data)

if __name__ == '__main__':
    main()