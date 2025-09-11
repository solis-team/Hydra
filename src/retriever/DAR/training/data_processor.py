import logging
import os
from datasets import load_dataset
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)

def load_and_process_data(data_args, model_args, tokenizer):
    logger.info("Loading dataset...")
    
    if data_args.dataset_source == "local_dataset":
        raw_datasets = load_local_datasets(data_args)
    elif data_args.dataset_source == "huggingface":
        raw_datasets = load_huggingface_datasets(data_args)
    else:
        raise ValueError(f"Unknown dataset_source: {data_args.dataset_source}")
    
    logger.info(f"Training dataset: {len(raw_datasets['train'])} samples")
    logger.info(f"Validation dataset: {len(raw_datasets['validation'])} samples")
    if "test" in raw_datasets:
        logger.info(f"Test dataset: {len(raw_datasets['test'])} samples")
    
    text_column_name = "text"
    label_column_name = "label"
    
    label_to_id = {0: 0, 1: 1}
    id_to_label = {0: 0, 1: 1}

    def preprocess_function(examples):
        texts = [str(t) for t in examples[text_column_name]]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
            max_length=data_args.max_seq_length,
        )
        
        if label_column_name in examples:
            tokenized["labels"] = examples[label_column_name]
        
        return tokenized
    
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name, label_column_name],  
        desc="Running tokenizer on dataset",
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # logger.info(f"Dataset columns after preprocessing: {raw_datasets['train'].column_names}")
    # logger.info(f"First sample: {raw_datasets['train'][0]}")
    
    return raw_datasets, data_collator, label_to_id, id_to_label


def load_local_datasets(data_args):
    if data_args.local_dataset_path:
        base_path = data_args.local_dataset_path
        train_file = os.path.join(
            base_path,
            "train_downsampling.jsonl" if data_args.train_downsample else "train.jsonl"
        )
        validation_file = os.path.join(base_path, "valid.jsonl")
        test_file = os.path.join(base_path, "test.jsonl")
    else:
        train_file = data_args.train_file
        if data_args.train_downsample:
            base_dir = os.path.dirname(train_file)
            downsampling_file = os.path.join(base_dir, "train_downsampling.jsonl")
            if os.path.exists(downsampling_file):
                train_file = downsampling_file
                logger.info(f"Using downsampled training data: {train_file}")
        validation_file = data_args.validation_file
        test_file = data_args.test_file
    
    data_files = {
        "train": train_file,
        "validation": validation_file,
        "test": test_file
    }
    
    logger.info(f"Loading local dataset from: {data_files}")
    raw_datasets = load_dataset("json", data_files=data_files)
    return raw_datasets


def load_huggingface_datasets(data_args):
    if not data_args.huggingface_dataset_name:
        raise ValueError("huggingface_dataset_name is required when using huggingface dataset source")
    
    raw_datasets = load_dataset(data_args.huggingface_dataset_name)
    
    if data_args.train_downsample and "train_downsampling" in raw_datasets:
        raw_datasets["train"] = raw_datasets["train_downsampling"]
        logger.info("Using train_downsampling split for training")


    return raw_datasets
