from datasets import load_dataset
import jsonlines, json
import os
from utils import parser, get_node_by_kind
from codetext.parser import PythonParser
import textwrap
import argparse
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--dataset_name', default="Fsoft-AIC/RepoExec", type=str)
argument_parser.add_argument('--n_samples', help='full_context | medium_context | short_context', type=int, default=5)
argument_parser.add_argument('--subset', help='full_context | medium_context | short_context', type=str,
                    choices=["full_context", "medium_context", "short_context"], default="full_context")
# parser.add_argument('--run_gt', help='Execute the grouth truth solution', action="store_true")
argument_parser.add_argument('--gendir', help='Raw Gen folder', type=str, default=None)

argument_parser.add_argument('--is_gpt', action="store_true")
argument_parser.add_argument('--is_api', action="store_true")
args = argument_parser.parse_args()


current_path = os.path.dirname(os.path.abspath(__file__))
temp_path = os.path.join(current_path, "temp.py")

n_samples = args.n_samples
set_name = args.subset
is_api_call = args.is_api
is_gpt = args.is_gpt or is_api_call
src = args.gendir
src_rs = os.path.join(src, "repoexec.final.generated.jsonl")
save_rs = os.path.join(src, "predictions/processed_generations.json")
if not os.path.exists(save_rs):
    os.makedirs(os.path.dirname(save_rs), exist_ok=True)
    
datasrc = load_dataset(args.dataset_name)[set_name]

with open(src_rs, "r") as f:
    samples = [json.loads(line) for line in f]
    samples_sorted = sorted(samples, key=lambda x: x['task_id'])
    generations = [sample['response'] for sample in samples_sorted]

def get_actual_solution(dp):
    root = parser.parse(bytes(dp["check"], "utf8"))
    root_node = root.root_node

    function_nodes = PythonParser.get_function_list(root_node)
    for function_node in function_nodes:
        entry_point = PythonParser.get_function_metadata(function_node, dp["check"])["identifier"]

        if entry_point == dp["entry_point"]:
            return function_node.text.decode()
    return None


def code_parser(response, target_func_prompt, function_signature):
    def normalize_indentation(code_text):
        lines = code_text.split('\n')
        if not lines:
            return ""
            
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return ""
            
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
        
        normalized_lines = []
        for line in lines:
            if line.strip():
                normalized_lines.append(line[min_indent:])
            else:
                normalized_lines.append(line)
                
        return '\n'.join(normalized_lines)
    #Find python block.-.
    python_blocks = []
    start_idx = 0
    while True:
        start = response.find("```python", start_idx)
        if start == -1:
            break
        
        start += len("```python")
        end = response.find("```", start)
        
        if end == -1:
            normalized_code = normalize_indentation(response[start:].rstrip())
            python_blocks.append((start, len(response), normalized_code))
            break
        else:
            normalized_code = normalize_indentation(response[start:end].rstrip())
            python_blocks.append((start, end, normalized_code))
            start_idx = end + 3
    
    if python_blocks:
        parsed_code = None
        for i in range(len(python_blocks)):  
            candidate_code = python_blocks[i][2]
            if function_signature in candidate_code:
                parsed_code = candidate_code
                break

        if parsed_code is None:
            parsed_code = python_blocks[0][2]
    else:
        parsed_code = response

    main_check = 'if __name__ == "__main__":'
    main_check_alt = "if __name__ == '__main__':" 
    
    if main_check in parsed_code:
        parsed_code = parsed_code[:parsed_code.find(main_check)].rstrip()
    elif main_check_alt in parsed_code:
        parsed_code = parsed_code[:parsed_code.find(main_check_alt)].rstrip()
    
    lines = parsed_code.split('\n')
    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line.startswith('import ') and not stripped_line.startswith('from '):
            filtered_lines.append(line)
    parsed_code = '\n'.join(filtered_lines)
    
    if not parsed_code.strip().startswith("from ") and not parsed_code.strip().startswith("import ") and not parsed_code.strip().startswith("def "):
        
        if (parsed_code.startswith("\n ") or 
            parsed_code.strip().startswith(" ") or 
            parsed_code.startswith("\t") or
            parsed_code.lstrip().startswith('"""') or 
            parsed_code.lstrip().startswith("'''") or
            parsed_code.lstrip().startswith('r"""') or
            parsed_code.lstrip().startswith("r'''")):
            lines = parsed_code.strip().split('\n')
            indented_lines = []
            is_raw_docstring = parsed_code.lstrip().startswith('r"""') or parsed_code.lstrip().startswith("r'''")
            for i, line in enumerate(lines):
                if line.strip():  
                    if i == 0 and is_raw_docstring:
                        if 'r"""' in line:
                            line = line.replace('r"""', '"""', 1)
                        elif "r'''" in line:
                            line = line.replace("r'''", "'''", 1)
                    if not line.startswith('    '):
                        indented_lines.append('    ' + line)
                    else:
                        indented_lines.append(line)
                else:
                    indented_lines.append(line)  
            indented_code = '\n'.join(indented_lines)
            result = target_func_prompt + indented_code
        else:
            result = parsed_code
    else:
        result = parsed_code
    if result.startswith(' ') or result.startswith('\t'):
        lines = result.split('\n')
        def_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                def_line_index = i
                break
        if def_line_index >= 0:
            def_indent = len(lines[def_line_index]) - len(lines[def_line_index].lstrip())
            if def_indent > 0:
                result = '\n'.join(
                    line[def_indent:] if line.strip() else line 
                    for line in lines
                )
        else:
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                if min_indent > 0:
                    result = '\n'.join(line[min_indent:] if line.strip() else line for line in lines)
    result = '\n'.join(line for line in result.split('\n') if line.strip())
    return result

    
def gpt_code_parser(response):
    if "```python" in response:
        parsed_code = response[response.index("```python") + len("```python"):]
        return parsed_code[:parsed_code.rfind("```")]
    elif "```" in response:
        parsed_code = response[response.index("```") + len("```"):]
        return parsed_code[:parsed_code.rfind("```")]
    else:
        return response

print(datasrc)
# exit()

wrong_process = 0
processed_generations = []
num_gen_tasks = len(generations)
print(num_gen_tasks)


actual_id = 0
for task_id, generation in enumerate(generations):
    if num_gen_tasks > len(datasrc) and task_id not in datasrc["id"]:
        continue
    
    all_test = []
    all_predictions = []

    if datasrc[actual_id]["solution"] not in datasrc[actual_id]["check"]:
        actual_solution = get_actual_solution(datasrc[actual_id])
    else:
        actual_solution = datasrc[actual_id]["solution"]

    for gen_id, gen_rs in enumerate(generation[:n_samples]):
        if "[/INST]" in gen_rs:
            gen_rs = gen_rs.split("[/INST]")[1].strip()

        if is_gpt:
            if is_api_call:
                if "</think>" in gen_rs["prediction"]:
                    gen_rs["prediction"] = gen_rs["prediction"].split("</think>")[-1]#.replace("        ", "    ")
                if is_instruct:
                    gen_rs = gpt_code_parser(gen_rs["prediction"])
                    print("hello")
                else:
                    if not gpt_code_parser(gen_rs["prediction"]).strip("\n").startswith("    "):
                        gen_rs = datasrc[actual_id]["target_function_prompt"] + textwrap.indent(gpt_code_parser(gen_rs["prediction"]), prefix="    ")
                    else:
                        gen_rs = datasrc[actual_id]["target_function_prompt"] + gpt_code_parser(gen_rs["prediction"])
            else:
                solution_body = None
                if datasrc[actual_id]["target_function_prompt"].strip() in gen_rs:
                    solution_body = gen_rs[gen_rs.index(datasrc[actual_id]["target_function_prompt"].strip()) + len(datasrc[actual_id]["target_function_prompt"].strip()): ]

                    if not solution_body.startswith("\n    ") and not solution_body.startswith("def") and not solution_body.startswith("\ndef"):
                        solution_body = "\n    " + solution_body.strip()
                    else:
                        solution_body = None
                
                if solution_body is not None:
                    gen_rs = datasrc[actual_id]["target_function_prompt"].strip() + solution_body

                
        #solution_fn = None
        try:
            solution_fn = code_parser(gen_rs, datasrc[actual_id]["target_function_prompt"].strip(), datasrc[actual_id]["function_signature"].strip())
        except Exception as e:
            print(e)
            solution_fn = ""
            print(actual_id, "cannot find solution for {} for task id {}".format(datasrc[actual_id]["entry_point"], datasrc[actual_id]["id"]))


        if solution_fn is None:
            solution_fn = ""
            print(actual_id, "cannot find solution for {} for task id {}".format(datasrc[actual_id]["entry_point"], datasrc[actual_id]["id"]))


        all_predictions.append(solution_fn)
        test_case = datasrc[actual_id]["check"]

        assert actual_solution in test_case
        test_case = test_case.replace(actual_solution, solution_fn)

        all_test.append(test_case)
        
    processed_generations.append({
        "task_id": actual_id,
        "project": datasrc[actual_id]["project"],
        "module":  datasrc[actual_id]["module"],
        "predictions": all_predictions,
        "test": all_test
        })
    if actual_id == 1:
        print(all_predictions[1])
    actual_id += 1


with jsonlines.open(save_rs, mode='w') as writer:
    writer.write_all(processed_generations)
