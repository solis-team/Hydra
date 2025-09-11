from rank_bm25 import BM25Okapi
import json
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

sep = "/"


def retrieve_bm25(example, top_k=10):
    query = example['target_function_prompt']
    candidate = example['candidate']
    
    corpus_data = []
    for candidate_type in ['class', 'function', 'variable']:
        if candidate_type in candidate:
            for candidate_id, candidate_info in candidate[candidate_type].items():
                text = candidate_info.get('source_code', '')
                metadata = {
                    'type': candidate_type,
                    'id': candidate_id,
                    'relative_path': candidate_info['relative_path'],
                    'source_code': candidate_info.get('source_code', ''),
                    'DAR_sample': candidate_info.get('DAR_sample', '')
                }
                corpus_data.append((text, metadata))
    
    if not corpus_data:
        return {}
    
    corpus_texts = [item[0] for item in corpus_data]
    tokenized_corpus = [text.split() for text in corpus_texts]
    tokenized_query = query.split()
    
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    scores = bm25.get_scores(tokenized_query)
    
    scored_items = []
    for i, (text, metadata) in enumerate(corpus_data):
        scored_items.append({
            'score': scores[i],
            'text': text,
            'metadata': metadata
        })
    
    top_results = sorted(scored_items, key=lambda x: x['score'], reverse=True)[:top_k]
    
    similarity_scores = defaultdict(lambda: {'class': {}, 'function': {}, 'variable': {}})
    
    for item in top_results:
        metadata = item['metadata']
        file_path = metadata['relative_path']
        candidate_type = metadata['type']
        candidate_id = metadata['id']
        
        similarity_scores[file_path][candidate_type][candidate_id] = {
            'relative_path': file_path,
            'source_code': metadata.get('source_code', ''),
            'DAR_sample': metadata.get('DAR_sample', ''),
            'bm25_score': item['score']
        }
    
    return dict(similarity_scores)



def create_prompt_with_bm25_chunking_context(example, input_dir = "cache/test-gen-repository/window", imported_context=True, benchmark="RepoExec"):
    def format_prompt(windows_path, current_fpath, query, input_fpath_tuple, import_file_tuples, output_path=None, imported_context=imported_context, t_f_p=None):
        #print(input_fpath_tuple)
        input_module = sep.join(input_fpath_tuple)
        updated_samples = []
        anchor_text = query
        if anchor_text is None:
            print("Warning: No anchor text found in the example.", anchor_text)
        if imported_context:
            with open(windows_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    metadata = data.get("metadata")
                    # print(metadata)
                    # break
                    if len(metadata) == 1:
                        if metadata[0]["fpath_tuple"] != input_fpath_tuple and metadata[0]["fpath_tuple"] in import_file_tuples:  # if the chunk is not from the same file but is from an import file, keep it
                            updated_samples.append(data)
                    elif len(metadata) > 1:
                        new_metadata = [
                            meta for meta in metadata
                            if meta["fpath_tuple"] != input_fpath_tuple and meta["fpath_tuple"] in import_file_tuples
                        ]
                        if new_metadata:
                            data["metadata"] = new_metadata 
                            updated_samples.append(data)

        else:
            with open(windows_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    metadata = data.get("metadata")
                    if len(metadata) == 1:
                        if metadata[0]["fpath_tuple"] != input_fpath_tuple:  # if the chunk is not from the same file but is from an import file, keep it
                            updated_samples.append(data)
                    elif len(metadata) > 1:
                        new_metadata = [
                            meta for meta in metadata
                            if meta["fpath_tuple"] != input_fpath_tuple 
                        ]
                        if new_metadata:
                            data["metadata"] = new_metadata
                            updated_samples.append(data)

        with open(current_fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data["metadata"] = [{"fpath_tuple": input_fpath_tuple}]
                #print(input_fpath_tuple)
                updated_samples.append(data) # Add the current file's context
        # with open("windows.jsonl", 'w', encoding='utf-8') as f:
        #     for sample in updated_samples:
        #         f.write(json.dumps(sample) + "\n")
        # return
                
        if updated_samples and anchor_text:
            tokenized_corpus = [context["context"].split() for context in updated_samples]
            tokenized_query = anchor_text.split()
            
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)

            for i in range(len(updated_samples)):
                updated_samples[i]["bm25_score"] = scores[i]

        top_10 = sorted(updated_samples, key=lambda x: x["bm25_score"], reverse=True)[:10]
        modules_dict = defaultdict(list)
        for sample in top_10:
            for metadata in sample["metadata"]:
                #print(metadata)
                module_name = "/".join(metadata["fpath_tuple"])
                modules_dict[module_name].append(sample)

        prompt_elements = ["You are a Python programmer working with a repository. Here is all the context you may find useful to complete the function:"
        ]

        same_modules = []
        count = 0
        for module_name, samples in modules_dict.items():
            if module_name != input_module:
                prompt_elements.append(f"#FILE: {module_name}")
                for i, sample in enumerate(samples):
                    count += 1
                    prompt_elements.append(f"##CHUNK {i+1}")
                    prompt_elements.append(sample['context'])
                    prompt_elements.append("")
            else:
                same_modules.extend(samples)
        prompt_elements.append(f"#CURRENT FILE: {input_module}")
        for i, sample in enumerate(same_modules):
            prompt_elements.append(f"##CHUNK {i+1}")
            prompt_elements.append(sample['context'])
            prompt_elements.append("")    
        prompt_elements.append("Based on the information above, please complete the function in the current file:")
        prompt_elements.append(t_f_p)
        prompt = "\n".join(prompt_elements)
        
        # if count + len(same_modules) > 10:
        #     print(f"Different modules: {count}, Same module: {len(same_modules)}")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
        return prompt
    input_fpath_tuple: list[str] = example["metadata"]["fpath_tuple"]
    repo = example["metadata"]["fpath_tuple"][0] if benchmark == "RepoExec" else example["metadata"]["fpath_tuple"][1]
    if benchmark == "DevEval":
        catergory = example["metadata"]["fpath_tuple"][0]
    id = example["metadata"]["id"]
    query = example["metadata"]["target_function_prompt"]
    t_f_p = example["metadata"]["target_function_prompt"] if example["metadata"]["type"] == "function" else example["metadata"]["target_method_prompt"]
    import_file = example["import_file"]
    import_file_tuples = [file.split('/') for file in import_file]
    windows_path = f"{input_dir}/repos/{repo}_ws20_ss2.jsonl" if benchmark == "RepoExec" else f"{input_dir}/repos/{catergory}/{repo}_ws20_ss2.jsonl"
    current_fpath = f"{input_dir}/current-files/{id}_ws20_ss2.jsonl"
    example["prompt"] = format_prompt(windows_path, current_fpath, query, input_fpath_tuple, import_file_tuples, None, imported_context, t_f_p)
    return example



def run_bm25_chunking(benchmark:str, imported_context:bool):
    def _run(samples, output_path, input_dir="cache/test-gen-repository/window", benchmark=benchmark):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in tqdm(samples):
                data = create_prompt_with_bm25_chunking_context(sample, input_dir, imported_context, benchmark)
                f.write(json.dumps(data) + "\n")
        print(f"BM25 prompts saved to {output_path}")

    if benchmark == "RepoExec":
        with open('data/temp/RepoExec_benchmark.jsonl', 'r') as file:
            samples = [json.loads(line) for line in file]
        output_path = "data/prompt/RepoExec_prompt.jsonl"
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _run(samples, output_path, input_dir="cache/benchmark/RepoExec/test-apps/window", benchmark=benchmark)
    elif benchmark == "DevEval":
        with open('data/temp/DevEval_benchmark.jsonl', 'r') as file:
            samples = [json.loads(line) for line in file]
        output_path = "data/prompt/DevEval_prompt.jsonl"
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        _run(samples, output_path, input_dir="cache/benchmark/DevEval/Source_Code/window", benchmark=benchmark)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="RepoExec", help="Benchmark to run: RepoExec or DevEval")
    parser.add_argument("--imported_context", action="store_true", help="Whether to include only imported context or not")
    args = parser.parse_args()
    run_bm25_chunking(args.benchmark, args.imported_context)
