# Reproducing Experimental Results

This document provides comprehensive instructions for reproducing the experimental results in our repository-level code generation research, including data extraction, benchmarking, and evaluation across different research questions.

## Table of Contents

- [Setup and Prerequisites](#setup-and-prerequisites)
- [Benchmarks](#benchmarks)
- [Data Extraction and Organization](#data-extraction-and-organization)
- [Model Training](#model-training)
- [Research Questions Reproduction](#research-questions-reproduction)
  - [RQ1: Structure-Aware vs Chunking-Based Indexing](#rq1-structure-aware-vs-chunking-based-indexing)
  - [RQ2: Different Retrieval Approaches](#rq2-different-retrieval-approaches)
  - [RQ3: Comparison with State-of-the-Art](#rq3-comparison-with-state-of-the-art)
  - [RQ4: Computational Cost Analysis](#rq4-computational-cost-analysis)
- [Code Generation](#code-generation)

### Structured Context Extraction

Before running any experiments that require structured context processing, you must first extract and organize repository information using our AST-based parser:

```bash
bash src/context_formulation/structured_indexer/run.sh --dataset <RepoExec|DevEval>
```

**Output Location**: After execution, the parsed output is automatically saved to:
```
data/parser_output/<dataset_name>/
```

This step creates structured representations of repositories including:
- Abstract Syntax Tree (AST) information
- Code block hierarchies
- Dependency relationships
- Function and class definitions

## Model Training

**Note**: If you plan to run experiments with the DAR (Dependency-Aware Retriever) or Hybrid retriever, you must first complete the training process.

For comprehensive instructions on training the Dependency-Aware Retriever (DAR) models used in this research, please refer to our detailed training documentation:

**[Training Guide](Training.md)**

The training guide covers:
- Dataset construction methodology
- Model architecture and configuration
- Training procedures and hyperparameters
- Evaluation metrics and model selection
- Hardware requirements and optimization

## Research Questions Reproduction

## Research Questions Reproduction

### RQ1: Structure-Aware vs Chunking-Based Indexing

**Research Question**: *How effective is Structure-Aware Indexing compared to Chunking-Based Indexing?*

This experiment compares two different context formulation approaches: traditional chunking-based methods versus our structure-aware indexing approach. To ensure fair comparison, we evaluate each context formulation method using the same retriever (either BM25 or UniXcoder), with each approach retrieving 10 chunks/code units consistently.

#### Fair Comparison with BM25 Retriever

**Chunking-Based Context + BM25 (10 chunks):**
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context chunking --retriever bm25
```

**Structure-Aware Context + BM25 (10 code units):**
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever bm25
```

#### Fair Comparison with UniXcoder Retriever

**Chunking-Based Context + UniXcoder (10 chunks):**
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context chunking --retriever unixcoder
```

**Structure-Aware Context + UniXcoder (10 code units):**
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever unixcoder
```


### RQ2: Different Retrieval Approaches

**Research Question**: *How do different retrieval approaches affect repository-level code generation performance?*

This experiment evaluates four different retrieval methods using structured context to understand their impact on generation quality.

#### BM25 Retriever
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever bm25
```

#### UniXcoder Retriever
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever unixcoder
```

#### Dependency-Aware Retriever (DAR)
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever dar
```

#### Hybrid Retriever
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever hybrid
```

### RQ3: Comparison with State-of-the-Art

**Research Question**: *How effective is our method compared to existing state-of-the-art approaches for repository-level code generation?*

#### Our Method (Hybrid Retriever)
Our proposed method combines multiple retrieval strategies:
```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever hybrid
```


### Generated Prompts
All prompt generation results are saved to:
```
data/prompt/<benchmark>_prompt.jsonl
```

#### Baseline Comparisons
For comprehensive evaluation, we compare against state-of-the-art baselines using their official implementations:

- **RepoCoder**: 
  - Repository: https://github.com/microsoft/CodeT

- **RLCoder**: 
  - Repository: https://github.com/DeepSoftwareAnalytics/RLCoder

- **RepoFormer**: 
  - Repository: https://github.com/amazon-science/Repoformer

### RQ4: Computational Cost Analysis

**Research Question**: *How does the computational cost of running Hydra compare to state-of-the-art approaches for repository-level code generation?*

This experiment evaluates the computational efficiency and latency characteristics of different retrieval methods to understand their practical deployment considerations.

#### Our Method (Hybrid Retriever) - Latency Analysis

First, generate prompts with the hybrid retriever to capture latency measurements:

```bash
bash src/retriever/create_prompt.sh --benchmark <RepoExec|DevEval> --context structured --retriever hybrid
```

Then analyze the retrieval latency using our measurement tool:

```bash
python src/retriever/compute_latency.py --benchmark <RepoExec|DevEval>
```

This workflow automatically embeds `time.perf_counter()` measurements during the retrieval process and then extracts comprehensive latency statistics.

#### Baseline Methods - Latency Measurement

For state-of-the-art baseline comparisons, researchers should instrument their retrieval implementations with timing measurements:

**Required Implementation**: Insert `time.perf_counter()` calls around retrieval operations in baseline methods to capture computational costs:

#### Comprehensive Analysis Workflow

**Step 1: Generate Prompts with Latency Capture**
```bash
bash src/retriever/create_prompt.sh --benchmark RepoExec --context structured --retriever hybrid
bash src/retriever/create_prompt.sh --benchmark DevEval --context structured --retriever hybrid
```

**Step 2: Analyze Computational Performance**
```bash
python src/retriever/compute_latency.py --benchmark RepoExec
python src/retriever/compute_latency.py --benchmark DevEval
```

#### Performance Metrics Reported

The analysis provides detailed performance characteristics:
- **Average latency** per retrieval operation (in milliseconds)
- **Min/Max latency** bounds for performance characterization
- **Median latency** for robust central tendency measurement

**Output**: Latency statistics are displayed in milliseconds with detailed breakdown of performance characteristics for deployment considerations.

## Code Generation

### Generation Pipeline Overview

Our code generation pipeline supports two distinct approaches: **opensource** models using local inference and **closedsource** models via API endpoints. Each approach is optimized for different deployment scenarios and research requirements.

### Opensource Generation

#### Framework Integration
For opensource model evaluation, we utilize the **code-llm-evaluator** framework developed by:
- **Author**: Dung Manh Nguyen (email: dungnm.workspace@gmail.com)
- **Repository**: https://github.com/FSoft-AI4Code/code-llm-evaluator


#### Opensource Generation Command
```bash
cd src/generator/opensource

python generate.py \
  --model <model_path_or_name> \
  --split <benchmark_split> \
  --data <dataset_path> \
  --task_name <RepoExec|DevEval> \
  --max_tokens 2048 \
  --batch_size 5 \
  --cache_dir <cache_directory> \
  --do_sample \
  --num_return_sequences 5 \
  --top_p 0.95 \
  --top_k -1 \
  --temperature 0.2
```

**Key Parameters:**
- `--model`: HuggingFace model identifier (e.g., `Qwen/Qwen2.5-Coder-7B-Instruct`)
- `--split`: Dataset split corresponding to retrieval method (`<benchmark>_<retriever>`)
- `--data`: Path to the generated prompt dataset
- `--max_tokens`: Maximum generation length
- `--num_return_sequences`: Number of candidate solutions per prompt

**Note**: Outputs are automatically saved to the standardized location `data/generation/<benchmark>/<benchmark>.final.generated.jsonl`

**Example Execution:**
```bash
python generate.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --split deveval_hybrid \
  --data data/prompt/DevEval_prompt.jsonl \
  --task_name DevEval \
  --max_tokens 2048
```


### Closedsource Generation

#### API Configuration
For closedsource models (GPT-4, Claude, etc.), configure the following environment variables:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_BASE_URL="your-base-url"  
export OPENAI_MODEL="your-model-name"      
```

#### Closedsource Generation Command
```bash
cd src/generator/closedsource

python generate.py \
  --benchmark <RepoExec|DevEval> \
  --input_file <prompt_dataset_path> \
  --output_file <generation_output_path> \
  --model_name gpt-4.1-mini-2025-04-14 \
  --max_tokens 2048 \
  --temperature 0.2
```

**Example Execution:**
**Example Execution:**
```bash
python generate.py --benchmark DevEval
```


### Generation Output Locations

Both opensource and closedsource generation approaches save outputs to the same standardized location:
```
data/generation/<benchmark>/<benchmark>.final.generated.jsonl
```

## Evaluation and Metrics Calculation

After generating outputs using either opensource or closedsource approaches, you need to compute the necessary evaluation metrics for each benchmark. Please refer to the respective benchmark documentation for detailed instructions on metric calculation:

### RepoExec Evaluation
For RepoExec benchmark evaluation and metrics calculation, refer to:
**`benchmark/RepoExec/README.md`**


### DevEval Evaluation  
For DevEval benchmark evaluation and metrics calculation, refer to:
**`benchmark/DevEval/README.md`**





