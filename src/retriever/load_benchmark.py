import os
import json
import glob
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import datasets
from abc import ABC, abstractmethod


class Benchmark_Loader(ABC):
    """Base class for benchmark dataset loaders."""
    
    def __init__(self, graphs_base_dir: str, output_dir: str = None):
        """
        Initialize the benchmark loader.
        
        Args:
            graphs_base_dir: Directory containing dependency graphs
            output_dir: Directory to save processed files
        """
        self.graphs_base_dir = graphs_base_dir
        self.output_dir = output_dir or "data"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_dependency_graph(self, graph_path: str) -> Dict[str, Any]:
        """Load dependency graph from JSON file."""
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading dependency graph from {graph_path}: {e}")
            return {}
    
    def load_external_knowledge(self, repo_dir: str) -> Dict[str, Any]:
        """Load external knowledge from JSON file in repo directory."""
        external_path = os.path.join(repo_dir, "external_knowledge.json")
        try:
            with open(external_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading external knowledge from {external_path}: {e}")
            return {}
    
    def get_import_statements(self, external_knowledge: Dict[str, Any], relative_path: str) -> List[str]:
        """Get import statements for a given relative path."""
        file_data = external_knowledge.get(relative_path, {})
        return file_data.get('import_statements', [])
    
    def find_dependency_graph_path(self, repo_name: str) -> Optional[str]:
        """Find dependency graph file for a given repository name."""
        pattern = os.path.join(self.graphs_base_dir, "**", repo_name, "dependency_graph.json")
        matches = glob.glob(pattern, recursive=True)
        
        if matches:
            return matches[0]
        
        pattern = os.path.join(self.graphs_base_dir, "**", "dependency_graph.json")
        all_graphs = glob.glob(pattern, recursive=True)
        
        for graph_path in all_graphs:
            if repo_name in graph_path:
                return graph_path
        
        return None
    
    def format_target_component(self, comp_data: Dict[str, Any]) -> str:
        """Format target component with name and description only."""
        result = {
            'name': comp_data['id'],
            'description': comp_data.get('docstring', 'DOCSTRING')
        }
        
        signature = comp_data.get('signature', '')
        if signature:
            result['signature'] = signature
        
        return json.dumps(result, ensure_ascii=False)
    
    def format_function_class_component(self, comp_data: Dict[str, Any]) -> str:
        """Format function or class component as JSON text."""
        result = {
            'name': comp_data['id'],
            'description': comp_data.get('docstring', 'DOCSTRING'),
            'code': comp_data.get('source_code', '')
        }
        
        if comp_data['component_type'] in ['function', 'class']:
            signature = comp_data.get('signature', '')
            if signature:
                result['signature'] = signature
        
        return json.dumps(result, ensure_ascii=False)
    
    def create_dar_sample(self, target_comp: Dict[str, Any], candidate_comp: Dict[str, Any]) -> str:
        target_text = self.format_target_component(target_comp)
        if candidate_comp['component_type'] == 'variable':
            context_text = candidate_comp.get('source_code', '')
        else:
            context_text = self.format_function_class_component(candidate_comp)
        
        return f"{target_text}</s>{context_text}"
    
    def process_candidates(self, target_comp: Dict[str, Any], components: Dict[str, Any], 
                          candidate_ids: List[str], comp_type: str) -> Dict[str, Dict[str, Any]]:
        """Process candidate components of a specific type."""
        candidates = {}
        
        for candidate_id in candidate_ids:
            if candidate_id in components:
                candidate_comp = components[candidate_id]
                
                candidates[candidate_id] = {
                    'relative_path': candidate_comp.get('relative_path', ''),
                    'source_code': candidate_comp.get('source_code', ''),
                    'DAR_sample': self.create_dar_sample(target_comp, candidate_comp)
                }
        
        return candidates
    
    @abstractmethod
    def load_dataset(self) -> Any:
        """Load the benchmark dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_output_filename(self) -> str:
        """Get the output filename for processed data. Must be implemented by subclasses."""
        pass
    
    def process_dataset(self, max_samples: Optional[int] = None):
        """Process the entire dataset and save to JSONL format."""
        dataset = self.load_dataset()
        if dataset is None:
            print("Failed to load dataset")
            return
        
        if max_samples and hasattr(dataset, '__len__'):
            dataset = dataset[:max_samples] if isinstance(dataset, list) else dataset.select(range(min(max_samples, len(dataset))))
            print(f"Processing first {len(dataset)} samples")
        
        processed_samples = []
        failed_count = 0
        
        print("Processing samples...")
        if isinstance(dataset, list):
            iterator = tqdm(dataset)
        else:
            iterator = tqdm(dataset)
        
        for sample in iterator:
            processed_sample = self.process_sample(sample)
            
            if processed_sample:
                processed_samples.append(processed_sample)
            else:
                failed_count += 1
        
        print(f"Successfully processed: {len(processed_samples)} samples")
        print(f"Failed to process: {failed_count} samples")
        
        if not processed_samples:
            print("No samples were successfully processed")
            return
        
        self.save_processed_benchmark(processed_samples)
        
        total_candidates = 0
        for sample in processed_samples:
            if 'candidate' in sample:
                for comp_type in ['class', 'function', 'variable']:
                    if comp_type in sample['candidate']:
                        total_candidates += len(sample['candidate'][comp_type])
        
        avg_candidates = total_candidates / len(processed_samples) if processed_samples else 0
        print(f"Average candidates per sample: {avg_candidates:.2f}")
    
    def save_processed_benchmark(self, processed_samples: List[Dict[str, Any]]):
        """Save processed samples to JSONL file."""
        output_path = os.path.join(self.output_dir, self.get_output_filename())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in processed_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"Saved {len(processed_samples)} processed samples to {output_path}")


class RepoExec_Loader(Benchmark_Loader):
    """Loader for RepoExec benchmark dataset."""
    
    def __init__(self, graphs_base_dir: str, output_dir: str = None,
                 dataset_name: str = "Fsoft-AIC/RepoExec", split: str = "full_context"):
        """
        Initialize RepoExec loader.
        
        Args:
            graphs_base_dir: Directory containing dependency graphs
            output_dir: Directory to save processed files
            dataset_name: HuggingFace dataset name
            split: Dataset split to load
        """
        super().__init__(graphs_base_dir, output_dir)
        self.dataset_name = dataset_name
        self.split = split
    
    def extract_repo_name(self, project: str) -> str:
        """Extract repository name from project field."""
        return project.split('/')[1]
    
    def construct_component_id(self, entry_point: str, module: str) -> str:
        """Construct component ID from entry_point and module."""
        module_path = module.replace('.', '/')
        return [f"{entry_point}@{module_path}.py", f"{entry_point}@src/{module_path}.py"]
    
    def load_dataset(self) -> datasets.Dataset:
        """Load RepoExec dataset from HuggingFace."""
        try:
            print(f"Loading dataset {self.dataset_name}, split {self.split}...")
            dataset = datasets.load_dataset(self.dataset_name, split=self.split)
            print(f"Loaded {len(dataset)} samples from RepoExec")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single RepoExec sample and create the required format."""
        try:
            task_id = sample['id']
            target_function_prompt = sample.get('target_function_prompt')
            project = sample['project']
            module = sample['module']
            entry_point = sample['entry_point']
            
            repo_name = self.extract_repo_name(project)
            graph_path = self.find_dependency_graph_path(repo_name)
            
            if not graph_path:
                print(f"Warning: No dependency graph found for repo {repo_name}")
                # print(project)
                return None

            # print(graph_path)

            components = self.load_dependency_graph(graph_path)
            repo_dir = os.path.dirname(graph_path)
            external_knowledge = self.load_external_knowledge(repo_dir)
            
            target_component_ids = self.construct_component_id(entry_point, module)
            target_component_id = target_component_ids[0]

            if target_component_id not in components:
                module_path = module.replace('.', '/') + '.py'
                if target_component_ids[1] in components:
                    target_component_id = target_component_ids[1]
                elif module_path in components:
                    target_component_id = module_path
                else:
                    print(f"Warning: Target component {target_component_id} not found in dependency graph")
                    return None
            
            target_comp = components[target_component_id]
            target_relative_path = target_comp.get('relative_path', '')
            
            outgoing_calls = target_comp.get('outgoing_calls', {"class": [], "function": [], "variable": []})
            noise = target_comp.get('noise', {"class": [], "function": [], "variable": []})
            
            all_candidates = {
                'class': outgoing_calls.get('class', []) + noise.get('class', []),
                'function': outgoing_calls.get('function', []) + noise.get('function', []),
                'variable': outgoing_calls.get('variable', []) + noise.get('variable', [])
            }
            
            candidate_dict = {
                'class': self.process_candidates(target_comp, components, all_candidates['class'], 'class'),
                'function': self.process_candidates(target_comp, components, all_candidates['function'], 'function'),
                'variable': self.process_candidates(target_comp, components, all_candidates['variable'], 'variable')
            }
            
            import_statements = self.get_import_statements(external_knowledge, target_relative_path)
            
            processed_sample = {
                'id': task_id,
                'target_function_prompt': target_function_prompt,
                'target_method_prompt': " ",
                'relative_path': target_relative_path,
                "type": "function",
                'candidate': candidate_dict,
                'import_statements': import_statements
            }
            
            return processed_sample
            
        except Exception as e:
            print(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            return None
    
    def get_output_filename(self) -> str:
        """Get the output filename for RepoExec processed data."""
        return "processed_RepoExec.jsonl"


class DevEval_Loader(Benchmark_Loader):
    """Loader for DevEval benchmark dataset."""
    
    def __init__(self, graphs_base_dir: str, output_dir: str = None,
                 data_path: str = None):
        """
        Initialize DevEval loader.
        
        Args:
            graphs_base_dir: Directory containing dependency graphs
            output_dir: Directory to save processed files
            data_path: Path to the local DevEval data.jsonl file
        """
        super().__init__(graphs_base_dir, output_dir)
        if data_path is None:
            repo_root = Path(__file__).resolve().parents[2]
            self.data_path = os.path.join(str(repo_root), 'benchmark', 'DevEval', 'data.jsonl')
        else:
            self.data_path = data_path
    
    def extract_repo_name(self, project_path: str) -> str:
        """Extract repository name from project_path field."""
        return project_path.split('/')[-1]
    
    def construct_component_id_from_namespace(self, namespace: str, completion_path: str, sample_type: str) -> str:
        """Construct component ID from namespace and completion_path."""  
        if sample_type == "function":
            function_name = namespace.split('.')[-1]
            path_parts = completion_path.split('/')
            if len(path_parts) >= 3:
                relative_path = '/'.join(path_parts[2:])  
            else:
                relative_path = path_parts[-1]

            return [f"{function_name}@{relative_path}", f"{function_name}@src/{relative_path}"]
        elif sample_type == "method":
            namespace_parts = namespace.split('.')
            if len(namespace_parts) >= 2:
                method_part = '.'.join(namespace_parts[-2:]) 
                path_parts = namespace_parts[:-2] 
                if path_parts:
                    file_path = '/'.join(path_parts) + '.py'  
                    init_path = '/'.join(path_parts) + '/__init__.py'
            return [f"{method_part}@{file_path}", f"{method_part}@src/{file_path}", f"{method_part}@{init_path}", f"{method_part}@src/{init_path}"]
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load DevEval dataset from local JSONL file."""
        try:
            print(f"Loading DevEval dataset from {self.data_path}...")
            samples = []
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num}: {e}")
                        continue
            
            print(f"Loaded {len(samples)} samples from DevEval")
            return samples
            
        except Exception as e:
            print(f"Error loading DevEval dataset: {e}")
            return None
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single DevEval sample and create the required format."""
        try:
            namespace = sample.get('namespace', '')
            sample_type = sample.get('type', '')
            project_path = sample.get('project_path', '')
            completion_path = sample.get('completion_path', '')
            target_function_prompt = sample.get('target_function_prompt', '')
            target_method_prompt = sample.get('target_method_prompt', '')

            task_id = namespace
            
            repo_name = self.extract_repo_name(project_path)
            graph_path = self.find_dependency_graph_path(repo_name)
            
            if not graph_path:
                print(f"Warning: No dependency graph found for repo {repo_name}")
                return None
            
            components = self.load_dependency_graph(graph_path)
            if not components:
                print(f"Warning: Empty dependency graph for repo {repo_name}")
                return None
            
            repo_dir = os.path.dirname(graph_path)
            external_knowledge = self.load_external_knowledge(repo_dir)
            
            target_component_ids = self.construct_component_id_from_namespace(namespace, completion_path, sample_type)
            if target_component_ids[0] in components:
                target_component_id = target_component_ids[0]
            if target_component_ids[0] not in components:
                function_name = namespace.split('.')[-1]
                relative_path = '/'.join(completion_path.split('/')[2:])
                
                possible_ids = [
                    target_component_ids[1],
                    target_component_ids[2],
                    target_component_ids[3],
                    f"{function_name}@{relative_path}"
                ]
                
                found_component = None
                for possible_id in possible_ids:
                    if possible_id in components:
                        target_component_id = possible_id
                        found_component = components[possible_id]
                        break
                
                if not found_component:
                    print(f"Warning: Target component not found for {namespace}")
                    print(f"Checked IDs:", target_component_ids)
                    return None
            
            target_comp = components[target_component_id]
            target_relative_path = target_comp.get('relative_path', '')
            
            outgoing_calls = target_comp.get('outgoing_calls', {"class": [], "function": [], "variable": []})
            noise = target_comp.get('noise', {"class": [], "function": [], "variable": []})
            
            all_candidates = {
                'class': outgoing_calls.get('class', []) + noise.get('class', []),
                'function': outgoing_calls.get('function', []) + noise.get('function', []),
                'variable': outgoing_calls.get('variable', []) + noise.get('variable', [])
            }
            
            candidate_dict = {
                'class': self.process_candidates(target_comp, components, all_candidates['class'], 'class'),
                'function': self.process_candidates(target_comp, components, all_candidates['function'], 'function'),
                'variable': self.process_candidates(target_comp, components, all_candidates['variable'], 'variable')
            }
            
            import_statements = self.get_import_statements(external_knowledge, target_relative_path)
            
            processed_sample = {
                'id': task_id,
                'target_function_prompt': target_function_prompt,
                'target_method_prompt': target_method_prompt,
                'relative_path': target_relative_path,
                'type': sample_type,
                'candidate': candidate_dict,
                'import_statements': import_statements
            }
            
            return processed_sample
            
        except Exception as e:
            print(f"Error processing sample {sample.get('namespace', 'unknown')}: {e}")
            return None
    
    def get_output_filename(self) -> str:
        """Get the output filename for DevEval processed data."""
        return "processed_DevEval.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Process a benchmark and generate processed outputs.")
    parser.add_argument('--benchmark', choices=['RepoExec', 'DevEval'], help='Benchmark to process (RepoExec or DevEval)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    graphs_base_dir = os.path.join(str(repo_root), 'data', 'parser_output', args.benchmark)
    output_dir = os.path.join(str(repo_root), 'data', 'processed_benchmarks')

    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(f"Processing {args.benchmark} Dataset")
    print("=" * 60)
    
    if args.benchmark == 'RepoExec':
        loader = RepoExec_Loader(
            graphs_base_dir=graphs_base_dir,
            output_dir=output_dir
        )
    else:  
        loader = DevEval_Loader(
            graphs_base_dir=graphs_base_dir,
            output_dir=output_dir
        )
    
    dataset = loader.load_dataset()
    
    #loader.process_dataset()
    loader.process_dataset(max_samples=2)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
