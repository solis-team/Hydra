import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

sep = "/"

class UniXcoderEmbedder:
    def __init__(self, model_name="microsoft/unixcoder-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load UniXcoder model and tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def get_embedding(self, text, max_length=512):
        """Get embedding for a single text"""
        # Tokenize and encode
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state
            # Apply attention mask and mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * mask_expanded
            embeddings = torch.sum(embeddings, dim=1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        return embeddings.cpu().numpy()
    
    def get_batch_embeddings(self, texts, batch_size=64, max_length=512):
        """Get embeddings for multiple texts in batches"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.get_embedding(text, max_length)
                batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        return np.vstack(all_embeddings)


# ================================================================================
# STRUCTURED CONTEXT FUNCTIONS
# ================================================================================

def retrieve_unixcoder(example, top_k=10):
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
    
    embedder = UniXcoderEmbedder()
    
    corpus_texts = [item[0] for item in corpus_data]
    context_embeddings = embedder.get_batch_embeddings(corpus_texts, batch_size=64)
    query_embedding = embedder.get_embedding(query)
    similarities = cosine_similarity(query_embedding, context_embeddings)[0]
    
    scored_items = []
    for i, (text, metadata) in enumerate(corpus_data):
        scored_items.append({
            'score': similarities[i],
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
            'unixcoder_score': float(item['score'])
        }
    
    return dict(similarity_scores)


# ================================================================================
# CHUNKING CONTEXT FUNCTIONS
# ================================================================================

def create_prompt_with_unixcoder_chunking_context(example, input_dir = "cache/test-gen-repository/window", imported_context=True, benchmark="RepoExec"):
    def format_prompt(windows_path, current_fpath, query, input_fpath_tuple, import_file_tuples, output_path=None, imported_context=imported_context, t_f_p=None):
        input_module = sep.join(input_fpath_tuple)
        updated_samples = []
        anchor_text = query
        if imported_context:
            with open(windows_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    metadata = data.get("metadata")
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
        with open(current_fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data["metadata"] = [{"fpath_tuple": input_fpath_tuple}]
                updated_samples.append(data) # Add the current file's context

            # with open("windows.jsonl", 'w', encoding='utf-8') as f:
            #     for sample in updated_samples:
            #         f.write(json.dumps(sample) + "\n")
            # return
        
        if updated_samples and anchor_text:
            print("Computing UniXcoder embeddings...")
            embedder = UniXcoderEmbedder()
            
            # Get all contexts
            contexts = [sample["context"] for sample in updated_samples]
            
            # Compute embeddings for all contexts
            context_embeddings = embedder.get_batch_embeddings(contexts)
            
            # Compute query embedding
            query_embedding = embedder.get_embedding(anchor_text)
            
            # Compute cosine similarities
            similarities = cosine_similarity(query_embedding, context_embeddings)[0]
            
            # Add similarity scores to samples
            for i, sample in enumerate(updated_samples):
                sample["unixcoder_score"] = float(similarities[i])
            
            print(f"Computed embeddings for {len(updated_samples)} samples")

        top_10 = sorted(updated_samples, key=lambda x: x.get("unixcoder_score", 0), reverse=True)[:10]
        modules_dict = defaultdict(list)
        for sample in top_10:
            for metadata in sample["metadata"]:
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

def run_unixcoder_chunking(benchmark:str, imported_context:bool):
    def _run(samples, output_path, input_dir="cache/test-gen-repository/window", benchmark=benchmark):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in tqdm(samples):
                data = create_prompt_with_unixcoder_chunking_context(sample, input_dir, imported_context, benchmark=benchmark)
                f.write(json.dumps(data) + "\n")
        print(f"UniXcoder prompts saved to {output_path}")

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
    run_unixcoder_chunking(args.benchmark, args.imported_context)
