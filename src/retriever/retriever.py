from collections import defaultdict
from similar_context.bm25 import retrieve_bm25
from similar_context.unixcoder import retrieve_unixcoder
from DAR.infer import DARInferenceModel
import os
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm

def dar_retriever(example):
    model_dir = "model"
    
    model = DARInferenceModel(model_dir)
    candidate = example['candidate']
    selected = defaultdict(lambda: {'class': {}, 'function': {}, 'variable': {}})
    
    for candidate_type in ['class', 'function', 'variable']:
        if candidate_type in candidate:
            texts = []
            meta = []
            for candidate_id, candidate_info in candidate[candidate_type].items():
                text = candidate_info.get('DAR_sample', '')
                texts.append(text)
                meta.append((candidate_id, candidate_info))
            
            if texts:
                preds = model.predict_batch(texts)
                for i, (pred, score) in enumerate(preds):
                    if pred == 1:
                        candidate_id, candidate_info = meta[i]
                        file_path = candidate_info['relative_path']
                        selected[file_path][candidate_type][candidate_id] = {
                            'relative_path': file_path,
                            'DAR_sample': candidate_info.get('DAR_sample', ''),
                            'source_code': candidate_info.get('source_code', ''),
                            'dar_score': score
                        }
    return dict(selected)

def hybrid_retriever(example):
    bm25_results = retrieve_bm25(example, top_k=5)
    dar_results = dar_retriever(example)
    merged = defaultdict(lambda: {'class': {}, 'function': {}, 'variable': {}})
    
    for results in [bm25_results, dar_results]:
        for file_path, types in results.items():
            for candidate_type in ['class', 'function', 'variable']:
                for candidate_id, candidate_info in types[candidate_type].items():
                    merged[file_path][candidate_type][candidate_id] = candidate_info
    
    return dict(merged)

def format_prompt(example, retriever_type):
    """Unified prompt formatting for all retriever types"""
    query = example['target_function_prompt']
    component_type = example['type']
    current_file_path = example['relative_path']
    
    # Start timing
    start_time = time.perf_counter()
    
    if retriever_type == 'bm25':
        results = retrieve_bm25(example, top_k=10)
    elif retriever_type == 'unixcoder':
        results = retrieve_unixcoder(example, top_k=10)
    elif retriever_type == 'dar':
        results = dar_retriever(example)
    elif retriever_type == 'hybrid':
        results = hybrid_retriever(example)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    # print(results)
    
    # End timing
    end_time = time.perf_counter()
    latency = end_time - start_time

    prompt_elements = [
        "You are a Python programmer working with a repository. Here is all the context you may find useful to complete the function:"
    ]

    current_file_results = {}
    other_file_results = {}

    for file_path, file_results in results.items():
        if file_path == current_file_path:
            current_file_results = file_results
        else:
            other_file_results[file_path] = file_results

    for file_path, file_results in other_file_results.items():
        prompt_elements.append(f"#FILE: {file_path}")
        for candidate_type in ['class', 'variable', 'function']:
            if file_results[candidate_type]:
                for candidate_id, candidate_info in file_results[candidate_type].items():
                    prompt_elements.append(candidate_info['source_code'])
                    prompt_elements.append("")

    if current_file_results:
        prompt_elements.append(f"#CURRENT FILE: {current_file_path}")
        import_stmts = example.get('import_statements', [])
        if import_stmts:
            for stmt in import_stmts:
                prompt_elements.append(stmt)
            prompt_elements.append("")
        for candidate_type in ['class', 'variable', 'function']:
            if current_file_results[candidate_type]:
                for candidate_id, candidate_info in current_file_results[candidate_type].items():
                    prompt_elements.append(candidate_info['source_code'])
                    prompt_elements.append("")

    prompt_elements.append("Based on the information above, please complete the function in the current file:")
    t_f_p = example["target_function_prompt"] if component_type == "function" else example["target_method_prompt"]
    prompt_elements.append(t_f_p)

    prompt = "\n".join(prompt_elements)
    return {
        'id': example.get('id'),
        'prompt': prompt,
        'latency': latency
    }

def run_prompt_pipeline(input_file, output_file, retriever_type):
    """Process JSONL file with specified retriever type"""
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Processing with {retriever_type}"):
            line = line.strip()
            if not line:
                continue
            
            example = json.loads(line)
            processed_example = format_prompt(example, retriever_type)
            f_out.write(json.dumps(processed_example) + '\n')
    
    print(f"Processed examples saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run retriever pipeline on benchmark data")
    parser.add_argument('--benchmark', choices=['RepoExec', 'DevEval'], help='Benchmark to process')
    parser.add_argument('--retriever', choices=['hybrid', 'bm25', 'unixcoder', 'dar'], help='Retriever type to use')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    
    if args.benchmark == 'RepoExec':
        input_file = os.path.join(str(repo_root), 'data', 'processed_benchmarks', 'processed_RepoExec.jsonl')
        output_file = os.path.join(str(repo_root), 'data', 'prompt', 'RepoExec_prompt.jsonl')
    else:
        input_file = os.path.join(str(repo_root), 'data', 'processed_benchmarks', 'processed_DevEval.jsonl')
        output_file = os.path.join(str(repo_root), 'data', 'prompt', 'DevEval_prompt.jsonl')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    run_prompt_pipeline(input_file, output_file, args.retriever)

if __name__ == "__main__":
    main()
