import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def load_dependency_graphs(parser_output_dir: str) -> Dict[str, Any]:
    """
    Load all dependency_graph.json files from subdirectories.
    
    Args:
        parser_output_dir: Directory containing repo subdirectories with dependency_graph.json
        
    Returns:
        Dict containing all components from all repositories
    """
    all_components = {}
    
    for root, dirs, files in os.walk(parser_output_dir):
        if "dependency_graph.json" in files:
            dep_graph_path = os.path.join(root, "dependency_graph.json")
            print(f"Loading dependency graph from: {dep_graph_path}")
            
            try:
                with open(dep_graph_path, 'r', encoding='utf-8') as f:
                    components = json.load(f)
                    
                for comp_id, comp_data in components.items():
                    if comp_id in all_components:
                        print(f"Warning: Duplicate component ID {comp_id}, overwriting...")
                    all_components[comp_id] = comp_data
                    
            except Exception as e:
                print(f"Error loading {dep_graph_path}: {e}")
                continue
    
    print(f"Loaded {len(all_components)} total components from all repositories")
    return all_components


def group_functions_by_out_count(components: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Group functions and classes by their out_count.
    
    Args:
        components: Dictionary of all components
        
    Returns:
        Dict mapping out_count to list of function/class component IDs
    """
    groups = defaultdict(list)
    
    for comp_id, comp_data in components.items():
        if comp_data['component_type'] in ['function', 'class']:
            out_count = comp_data.get('out_count', 0)
            groups[out_count].append(comp_id)
    
    return dict(groups)


def select_target_functions(groups: Dict[int, List[str]]) -> List[str]:
    """
    Select one random function/class from each out_count group as target.
    
    Args:
        groups: Dict mapping out_count to list of component IDs
        
    Returns:
        List of selected target function/class IDs
    """
    target_functions = []
    
    for out_count, comp_ids in groups.items():
        if comp_ids:  
            selected = random.choice(comp_ids)
            target_functions.append(selected)
            print(f"Selected target function from group {out_count}: {selected}")
    
    return target_functions


def format_function_class_component(comp_data: Dict[str, Any]) -> str:
    """
    Format function or class component as JSON text.
    
    Args:
        comp_data: Component data dictionary
        
    Returns:
        JSON formatted string
    """
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


def format_target_component(comp_data: Dict[str, Any]) -> str:
    """
    Format target component (function/class) with name, description, signature only.
    
    Args:
        comp_data: Component data dictionary
        
    Returns:
        JSON formatted string
    """
    result = {
        'name': comp_data['id'],
        'description': comp_data.get('docstring', 'DOCSTRING')
    }
    
    signature = comp_data.get('signature', '')
    if signature:
        result['signature'] = signature
    
    return json.dumps(result, ensure_ascii=False)


def create_samples_for_target(target_id: str, components: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create training samples for a target function.
    
    Args:
        target_id: ID of the target function/class
        components: Dictionary of all components
        
    Returns:
        List of training samples
    """
    samples = []
    target_comp = components[target_id]
    
    target_text = format_target_component(target_comp)
    
    outgoing_calls = target_comp.get('outgoing_calls', {"class": [], "function": [], "variable": []})
    noise = target_comp.get('noise', {"class": [], "function": [], "variable": []})
    
    for comp_type in ['class', 'function', 'variable']:
        for comp_id in outgoing_calls[comp_type]:
            if comp_id in components:
                comp_data = components[comp_id]
                
                if comp_data['component_type'] == 'variable':
                    context_text = comp_data.get('source_code', '')
                else:
                    context_text = format_function_class_component(comp_data)
                
                sample = {
                    'text': f"{target_text}</s>{context_text}",
                    'label': 1
                }
                samples.append(sample)
    
    for comp_type in ['class', 'function', 'variable']:
        for comp_id in noise[comp_type]:
            if comp_id in components:
                comp_data = components[comp_id]
                
                if comp_data['component_type'] == 'variable':
                    context_text = comp_data.get('source_code', '')
                else:
                    context_text = format_function_class_component(comp_data)
                
                sample = {
                    'text': f"{target_text}</s>{context_text}",
                    'label': 0
                }
                samples.append(sample)
    
    return samples


def split_data(samples: List[Dict[str, Any]], train_ratio: float = 0.8, 
              valid_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Split samples into train/valid/test sets.
    
    Args:
        samples: List of all samples
        train_ratio: Ratio for training set
        valid_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_samples, valid_samples, test_samples)
    """
    random.shuffle(samples)
    
    total = len(samples)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    
    train_samples = samples[:train_end]
    valid_samples = samples[train_end:valid_end]
    test_samples = samples[valid_end:]
    
    return train_samples, valid_samples, test_samples


def downsample_train_data(train_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Downsample class 0 samples to match class 1 count for balanced training.
    
    Args:
        train_samples: List of training samples
        
    Returns:
        List of downsampled training samples
    """
    label_0_samples = [sample for sample in train_samples if sample['label'] == 0]
    label_1_samples = [sample for sample in train_samples if sample['label'] == 1]
    
    print(f"Original train data: Label 0={len(label_0_samples)}, Label 1={len(label_1_samples)}")
    
    if len(label_0_samples) <= len(label_1_samples):
        print("No downsampling needed (class 0 <= class 1)")
        return train_samples
    
    downsampled_label_0 = random.sample(label_0_samples, len(label_1_samples))
    
    downsampled_samples = downsampled_label_0 + label_1_samples
    random.shuffle(downsampled_samples)
    
    print(f"Downsampled train data: Label 0={len(downsampled_label_0)}, Label 1={len(label_1_samples)}")
    return downsampled_samples


def save_samples_to_jsonl(samples: List[Dict[str, Any]], output_path: str):
    """
    Save samples to JSONL format.
    
    Args:
        samples: List of samples to save
        output_path: Path to output JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(samples)} samples to {output_path}")


def create_dataset(parser_output_dir: str, data_output_dir: str):
    """
    Main function to create the dataset.
    
    Args:
        parser_output_dir: Directory containing parsed repository data
        data_output_dir: Directory to save train/valid/test data
    """
    print(f"Creating dataset from: {parser_output_dir}")
    print(f"Output directory: {data_output_dir}")
    
    components = load_dependency_graphs(parser_output_dir)
    
    if not components:
        print("No components found. Exiting.")
        return
    
    groups = group_functions_by_out_count(components)
    print(f"Found {len(groups)} different out_count groups")
    
    target_functions = select_target_functions(groups)
    print(f"Selected {len(target_functions)} target functions")
    
    all_samples = []
    for target_id in target_functions:
        print(f"Creating samples for target: {target_id}")
        samples = create_samples_for_target(target_id, components)
        all_samples.extend(samples)
        print(f"  Created {len(samples)} samples")
    
    print(f"Total samples created: {len(all_samples)}")
    
    if not all_samples:
        print("No samples created. Exiting.")
        return
    
    train_samples, valid_samples, test_samples = split_data(all_samples)
    
    print(f"Data split: Train={len(train_samples)}, Valid={len(valid_samples)}, Test={len(test_samples)}")
    
    save_samples_to_jsonl(train_samples, os.path.join(data_output_dir, 'train.jsonl'))
    save_samples_to_jsonl(valid_samples, os.path.join(data_output_dir, 'valid.jsonl'))
    save_samples_to_jsonl(test_samples, os.path.join(data_output_dir, 'test.jsonl'))
    
    downsampled_train_samples = downsample_train_data(train_samples)
    save_samples_to_jsonl(downsampled_train_samples, os.path.join(data_output_dir, 'train_downsampling.jsonl'))
    
    print("Dataset creation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DAR training dataset from parsed repositories")
    parser.add_argument("--parser_output", type=str, 
                       default=None,
                       help="Directory containing parsed repository data")
    parser.add_argument("--data_output", type=str, 
                       default="data",
                       help="Directory to save train/valid/test data (default: data)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    parser_output_dir = os.path.abspath(args.parser_output)
    data_output_dir = os.path.abspath(args.data_output)
    
    if not os.path.exists(parser_output_dir):
        print(f"Error: Parser output directory does not exist: {parser_output_dir}")
        exit(1)
    
    create_dataset(parser_output_dir, data_output_dir)
