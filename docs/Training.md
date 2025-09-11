# Training Guide — DAR (Dependency-Aware Retriever)

This document outlines the complete training methodology for our Dependency-Aware Retriever (DAR) model. We provide detailed instructions for dataset construction, model training, and experimental reproduction to enable research reproducibility.

## Overview

The DAR training pipeline consists of three core stages:
1. **Dataset Construction**: Repository collection, parsing, and data generation
2. **Model Training**: Binary classification training for dependency relevance prediction  
3. **Evaluation**: Model validation and performance assessment

---

## 1. Dataset Construction

### 1.1 Repository Collection and Processing

To construct our training dataset, we systematically collected approximately 3,000 Python repositories from The Stack dataset. The dataset construction process follows a rigorous multi-stage pipeline:

**Stage 1: Repository Parsing**
We employ an Abstract Syntax Tree (AST) parser to extract structured representations from each repository:

```bash
python src/context_formulation/structured_indexer/ast_parser.py \
  --repo_path /path/to/repository \
  --output_dir /path/to/parser_output/<repo_name>
```

This parser generates:
- `dependency_graph.json`: Complete dependency relationships between code components
- `external_knowledge.json`: Import statements and external dependencies

**Stage 2: Data Generation and Splitting**
The parsed repository data is then processed to create training instances:

```bash
python src/retriever/DAR/data_preprocessing/creation.py \
  --parser_output /path/to/parser_output \
  --data_output /path/to/training_dataset \
  --seed 42
```

This process:
- Extracts function-context pairs from dependency graphs
- Generates positive samples (actual dependencies) and negative samples (noise)
- Performs stratified splitting into training/validation/test sets
- Creates balanced downsampled versions for class imbalance mitigation

Each sample follows the format:
```json
{"text": "{\"name\": \"target_function\", \"description\": \"...\"}</s>{\"name\": \"context_component\", \"code\": \"...\"}", "label": 1}
```

Where `label: 1` indicates relevant dependency and `label: 0` indicates noise.


**Stage 3: Final Dataset (Hydra-Dataset)**
The complete processing pipeline produced our final training dataset, available at `Hydra-2025/hydra` on Hugging Face. This dataset includes four splits:
- `train`: Full training set (~80% of total samples)
- `train_downsampling`: Balanced training set for class imbalance mitigation
- `valid`: Validation set (~10% of total samples)  
- `test`: Test set (~10% of total samples)

In our research, we use the downsampled version (`train_downsampling`) by default to address class imbalance issues.

---

## 2. Model Architecture and Training

### 2.1 Model Architecture

Our DAR model (`CustomCodeClassifier`) implements a binary classifier built upon pre-trained code representations:

- **Base Encoder**: Microsoft UniXcoder (`microsoft/unixcoder-base`)
- **Classification Head**: Multi-layer feedforward network with batch normalization and dropout
- **Output**: Binary classification for dependency relevance prediction

### 2.2 Training Configuration

**To reproduce the results, you need to use the default training command (with downsampling):**
```bash
python src/retriever/DAR/training/train.py \
  --datasource huggingface \
  --huggingface_dataset_name Hydra-2025/hydra \
  --downsample 
```

This uses the following updated default parameters from `train.py`:

- `--datasource local`
- `--data_dir data/hydra-dataset`
- `--model_name_or_path microsoft/unixcoder-base`
- `--use_fast_tokenizer`
- `--max_seq_length 512`
- `--pad_to_max_length`
- `--num_train_epochs 10`
- `--per_device_train_batch_size 32`
- `--per_device_eval_batch_size 32`
- `--learning_rate 2e-5`
- `--weight_decay 0.01`
- `--gradient_accumulation_steps 16`
- `--run_name dar-classification`
- `--logging_steps 100`
- `--seed 42`


**Training Command (with local dataset):**

You can also train with your own dataset

```bash
python src/retriever/DAR/training/train.py \
  --datasource local \
  --data_dir /path/to/your/dataset \
  --downsample 
```
```bash
python src/retriever/DAR/training/train.py \
  --datasource local \
  --data_dir data/hydra-dataset \
  --downsample 
```

### 2.3 Training Arguments Reference

**Data Configuration:**
- `--datasource`: Data source type (`local` for JSONL files, `huggingface` for HF datasets)
- `--data_dir`: Directory containing training data files
- `--downsample`: Use balanced training set (`train_downsampling.jsonl`)

**Model Configuration:**
- `--model_name_or_path`: Base pre-trained model (default: `microsoft/unixcoder-base`)
- `--model_dir`: Output directory for trained model weights
- `--max_seq_length`: Maximum input sequence length (default: 512)

**Training Hyperparameters:**
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Training batch size per GPU/device
- `--per_device_eval_batch_size`: Evaluation batch size per GPU/device  
- `--learning_rate`: Optimizer learning rate (recommended: 1e-5 to 5e-5)
- `--weight_decay`: L2 regularization coefficient
- `--gradient_accumulation_steps`: Gradient accumulation for effective larger batch sizes

**Technical Options:**
- `--pad_to_max_length`: Enable static padding to maximum sequence length
- `--use_fast_tokenizer`: Use optimized tokenizer implementation
- `--seed`: Random seed for reproducibility (default: 42)

---

## 3. Evaluation and Model Selection

### 3.1 Evaluation Metrics

Our training pipeline employs comprehensive evaluation metrics:
- **Primary Metric**: `RecallTrue` (recall on positive class) - used for best model selection
- **Secondary Metrics**: Accuracy, Precision, F1-score for both classes
- **Per-Class Analysis**: Separate metrics for positive (relevant) and negative (noise) samples

### 3.2 Model Loading and Inference

```python
from src.retriever.DAR.training.model import CustomCodeClassifier
from transformers import AutoTokenizer

model = CustomCodeClassifier.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

inputs = tokenizer("sample_text", return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
```
