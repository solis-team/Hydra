# Hydra

This repository contains the replication package of the paper **"Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond"**.

## Abstract

Large language models for code (CodeLLMs) have demonstrated remarkable success in standalone code completion and generation, yet their effectiveness diminishes in repository-level settings where cross-file dependencies and structural context are essential. Existing Retrieval-Augmented Generation (RAG) approaches often borrow strategies from NLP, relying on chunking-based indexing and similarity-based retrieval that overlook structural relationships and miss functionally relevant dependencies.

We present **Hydra**, a repository-level code generation framework that treats code as structured code rather than natural language. Our approach introduces: (i) structure-aware indexing that preserves code structure and dependencies, (ii) a lightweight dependency-aware retriever (DAR) that identifies true dependencies, and (iii) hybrid retrieval combining dependency-aware and similarity-based methods.

Extensive experiments on DevEval and RepoExec benchmarks show that HyDra achieves state-of-the-art performance, surpassing the strongest baseline by over 5% in Pass@1 and enabling smaller models to match larger ones.

## Research Questions

1. **RQ1**: How effective is Structure-Aware Indexing compared to Chunking-Based Indexing?
2. **RQ2**: How do different retrieval approaches affect repository-level code generation performance?
3. **RQ3**: How effective is Hydra compared to existing state-of-the-art approaches for repository-level code
generation?
4. **RQ4**: How does the computational cost of running Hydra compare to state-of-the-art approaches for repository-level code generation?

## Quick Start

### Prerequisites

Create a new conda environment and install dependencies:

```bash
# Create conda environment
conda create -n hydra python=3.10.0
conda activate hydra

# Install required packages
pip install -r requirements.txt
```

### Setup and Installation

**Important**: You must complete the following setup steps before running any experiments.

1. **Extract benchmark data:**
   ```bash
   cd data
   unzip temp.zip
   
   # Extract RepoExec benchmark
   cd ../benchmark/RepoExec
   unzip test-apps.zip
   
   # Extract DevEval benchmark
   cd ../DevEval
   tar -xzf data.tar.gz
   wget https://huggingface.co/datasets/LJ0815/DevEval/resolve/main/Source_Code.tar.gz
   tar -xvzf Source_Code.tar.gz
   ```

2. **Prepare structured context (required for experiments):**
   ```bash
   # For RepoExec benchmark
   bash src/context_formulation/structured_indexer/run.sh --dataset RepoExec
   
   # For DevEval benchmark  
   bash src/context_formulation/structured_indexer/run.sh --dataset DevEval
   ```

## Documentation
**Important** Before reproducing experiments, you must first train the DAR (Dependency-Aware Retriever) model. 

For detailed instructions and comprehensive guides, please refer to:
- **[Training.md](docs/Training.md)** - DAR (Dependency-Aware Retriever) training guide including:
  - Dataset construction methodology
  - Model architecture and training procedures

- **[Reproduce.md](docs/Reproduce.md)** - Complete experimental reproduction guide including:
  - Benchmark setup and data preparation
  - Research questions reproduction (RQ1-RQ4)
  - Code generation pipeline
  - Evaluation and metrics calculation


## Citation

If you found this repository to be useful, please cite:

```bibtex
@misc{leanh2026treatcodenaturallanguage,
      title={Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond}, 
      author={Minh Le-Anh and Huyen Nguyen and Khanh An Tran and Nam Le Hai and Linh Ngo Van and Nghi D. Q. Bui and Bach Le},
      year={2026},
      eprint={2602.11671},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.11671}, 
}
```
