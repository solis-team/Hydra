<h1 align="center">Do Not Treat Code as Natural Language: Implications for Repository-Level Code Generation and Beyond</h1>

<p align="center">
  <strong>Repository-Level Code Generation</strong> • <strong>Structure-Aware Indexing</strong> • <strong>Dependency-Aware Retrieval</strong>
</p>

<p align="center">
  <a href="https://python.org/"><img alt="Python version" src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" /></a>
  <a href="./LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" /></a>
  <a href="https://arxiv.org/abs/2602.11671"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.11671-b31b1b?style=flat-square" /></a>
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> •
  <a href="#documentation"><strong>Documentation</strong></a> •
  <a href="https://arxiv.org/abs/2602.11671"><strong>Paper</strong></a>
</p>

## DAR Model is available at: https://huggingface.co/solis-soict/Hydra-DAR !

## Introduction

Large language models for code (CodeLLMs) have demonstrated remarkable success in standalone code completion and generation, yet their effectiveness diminishes in repository-level settings where cross-file dependencies and structural context are essential. Existing Retrieval-Augmented Generation (RAG) approaches often borrow strategies from NLP, relying on chunking-based indexing and similarity-based retrieval that overlook structural relationships and miss functionally relevant dependencies.

We present **Hydra**, a repository-level code generation framework that treats code as structured code rather than natural language. Our approach introduces: (i) **structure-aware indexing** that preserves code structure and dependencies, (ii) a lightweight **dependency-aware retriever (DAR)** that identifies true dependencies, and (iii) **hybrid retrieval** combining dependency-aware and similarity-based methods.

Extensive experiments on DevEval and RepoExec benchmarks show that Hydra achieves state-of-the-art performance, surpassing the strongest baseline by over 5% in Pass@1 and enabling smaller models to match larger ones.

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

## Main Results

Comparison of Hydra with prior retrieval-based approaches and no-context baselines. Results are reported in Pass@1/3/5.

### GPT-4.1 mini

| Method | RepoExec Pass@1 | RepoExec Pass@3 | RepoExec Pass@5 | DevEval Pass@1 | DevEval Pass@3 | DevEval Pass@5 |
|--------|-----------------|-----------------|-----------------|----------------|---------------|----------------|
| No Context | 21.58 | 24.42 | 25.63 | 19.72 | 23.19 | 24.71 |
| RepoCoder | 22.20 | 26.08 | 27.89 | 17.48 | 23.15 | 25.70 |
| RepoFormer | 39.15 | 42.42 | 43.94 | 30.89 | 34.21 | 35.40 |
| RLCoder | 38.14 | 42.17 | 43.38 | 29.46 | 32.76 | 34.14 |
| **Hydra** | **43.55** | **45.72** | **46.48** | **31.91** | **35.56** | **36.99** |

### QwenCoder-1.5B-Instruct

| Method | RepoExec Pass@1 | RepoExec Pass@3 | RepoExec Pass@5 | DevEval Pass@1 | DevEval Pass@3 | DevEval Pass@5 |
|--------|-----------------|-----------------|-----------------|----------------|---------------|----------------|
| No Context | 5.75 | 8.31 | 9.30 | 3.53 | 5.20 | 5.97 |
| RepoCoder | 7.15 | 11.72 | 14.37 | 4.54 | 8.08 | 9.81 |
| RepoFormer | 11.15 | 16.42 | 18.87 | 5.58 | 7.94 | 8.99 |
| RLCoder | 14.87 | 21.04 | **23.94** | 9.34 | 12.90 | 14.47 |
| **Hydra** | **15.72** | **21.30** | 23.38 | **10.71** | **14.50** | **16.05** |

### QwenCoder-7B-Instruct

| Method | RepoExec Pass@1 | RepoExec Pass@3 | RepoExec Pass@5 | DevEval Pass@1 | DevEval Pass@3 | DevEval Pass@5 |
|--------|-----------------|-----------------|-----------------|----------------|---------------|----------------|
| No Context | 13.30 | 17.04 | 18.03 | 7.10 | 9.16 | 10.03 |
| RepoCoder | 14.82 | 21.99 | 25.07 | 6.39 | 10.63 | 12.82 |
| RepoFormer | 17.69 | 25.04 | 28.45 | 10.41 | 13.68 | 14.90 |
| RLCoder | 20.17 | 23.69 | 27.61 | 13.00 | 17.67 | 19.61 |
| **Hydra** | **23.32** | **31.32** | **34.36** | **17.27** | **22.44** | **24.44** |

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
