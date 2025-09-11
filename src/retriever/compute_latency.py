import json
import argparse
import statistics
from pathlib import Path


def compute_latency_stats(prompt_file):
    """    
    Args:
        prompt_file (str): Path to the JSONL file containing prompts with latency data
        
    Returns:
        dict: Dictionary containing min, max, mean, and median latency values
    """
    latencies = []
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if 'latency' in data:
                    latencies.append(data['latency'])
                else:
                    print(f"Warning: No latency field found in line {line_num}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
    
    if not latencies:
        print("No latency data found in the file!")
        return None
    
    stats = {
        'count': len(latencies),
        'min_ms': min(latencies) * 1000,
        'max_ms': max(latencies) * 1000,
        'mean_ms': statistics.mean(latencies) * 1000,
        'median_ms': statistics.median(latencies) * 1000,
        'min_sec': min(latencies),
        'max_sec': max(latencies),
        'mean_sec': statistics.mean(latencies),
        'median_sec': statistics.median(latencies)
    }
    
    return stats


def print_latency_stats(stats, benchmark_name):
    """
    Print formatted latency statistics
    
    Args:
        stats (dict): Statistics dictionary from compute_latency_stats
        benchmark_name (str): Name of the benchmark for display
    """
    if stats is None:
        return
    
    print(f"\n{'='*50}")
    print(f"Latency Statistics for {benchmark_name}")
    print(f"{'='*50}")
    print(f"Total samples: {stats['count']}")
    print(f"Min latency:   {stats['min_ms']:.4f} ms ({stats['min_sec']:.8f} seconds)")
    print(f"Max latency:   {stats['max_ms']:.4f} ms ({stats['max_sec']:.8f} seconds)")
    print(f"Mean latency:  {stats['mean_ms']:.4f} ms ({stats['mean_sec']:.8f} seconds)")
    print(f"Median latency: {stats['median_ms']:.4f} ms ({stats['median_sec']:.8f} seconds)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute latency statistics from retriever prompt files"
    )
    parser.add_argument(
        '--benchmark', 
        choices=['RepoExec', 'DevEval'], 
        required=True,
        help='Benchmark to analyze (RepoExec or DevEval)'
    )
    parser.add_argument(
        '--prompt_file',
        type=str,
        help='Custom path to prompt file (optional, will use default location if not specified)'
    )
    
    args = parser.parse_args()
    
    if args.prompt_file:
        prompt_file = args.prompt_file
    else:
        repo_root = Path(__file__).resolve().parents[2]
        prompt_file = repo_root / 'data' / 'prompt' / f'{args.benchmark}_prompt.jsonl'
    
    if not Path(prompt_file).exists():
        print(f"Error: Prompt file not found at {prompt_file}")
        print("Make sure you have run the retriever pipeline first.")
        return
    
    print(f"Analyzing latency data from: {prompt_file}")
    
    stats = compute_latency_stats(prompt_file)
    print_latency_stats(stats, args.benchmark)

if __name__ == "__main__":
    main()
