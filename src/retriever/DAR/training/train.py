import os
import logging
import torch
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoConfig, 
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

from model import CustomCodeClassifier
from utils import seed_all, compute_detailed_metrics
from data_processor import load_and_process_data
from arguments import DataTrainingArguments, ModelArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ModelCallback(TrainerCallback):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.best_metric = -float('inf')
        
    def on_epoch_end(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None or metrics is None:
            return
        
        print(f"\nEpoch {int(state.epoch)} Metrics:")
        print(f"Acc: {metrics.get('eval_accuracy', 0):.4f}")
        print(f"Recall: {metrics.get('eval_recall', 0):.4f}")
        print(f"Precision: {metrics.get('eval_precision', 0):.4f}")
        print(f"RecallTrue: {metrics.get('eval_recall_true', 0):.4f}")
        print(f"PrecisionTrue: {metrics.get('eval_precision_true', 0):.4f}")
        print(f"F1True: {metrics.get('eval_f1_true', 0):.4f}")
        print(f"RecallFalse: {metrics.get('eval_recall_false', 0):.4f}")
        print(f"PrecisionFalse: {metrics.get('eval_precision_false', 0):.4f}")
        print(f"F1False: {metrics.get('eval_f1_false', 0):.4f}")
        
        current_metric = metrics.get('eval_recall_true', 0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            model.save_pretrained(self.model_dir)
            logger.info(f"New best model with RecallTrue {current_metric:.4f}")


def save_final_model(model, tokenizer, config, model_dir, model_name_or_path, max_seq_length):
    model.save_pretrained(model_dir)
    model.config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Final model saved: {model_dir}")


def run(datasource, 
        data_dir=None, 
        huggingface_dataset_name=None,
        downsample=False,
        model_name_or_path="microsoft/unixcoder-base",
        max_seq_length=512,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42,
        model_dir=None,
        run_name="dar-classification",
        logging_steps=100,
        gradient_accumulation_steps=1,
        pad_to_max_length=True,
        use_fast_tokenizer=True):
    """
    Run training with specified parameters.
    
    Args:
        datasource: "local" or "huggingface"
        data_dir: Path to folder containing .jsonl files (for local datasource)
        huggingface_dataset_name: Name of HuggingFace dataset (for huggingface datasource)
        downsample: Whether to use train_downsampling instead of train
        model_name_or_path: Base model to use
        max_seq_length: Maximum sequence length
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size for training
        per_device_eval_batch_size: Batch size for evaluation
        learning_rate: Learning rate
        weight_decay: Weight decay
        seed: Random seed
        model_dir: Directory to save model (default: DAR/model)
        run_name: Name for the training run
        logging_steps: Log every N steps
        gradient_accumulation_steps: Number of gradient accumulation steps
        pad_to_max_length: Whether to pad sequences to max length
        use_fast_tokenizer: Whether to use fast tokenizer
    """
    
    seed_all(seed)
    
    if model_dir is None:
        model_dir = "model"
    
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Model directory: {model_dir}")
    

    if datasource == "local":
        if not data_dir:
            raise ValueError("data_dir is required for local datasource")
        
        data_args = DataTrainingArguments(
            dataset_source="local_dataset",
            local_dataset_path=data_dir,
            train_downsample=downsample,
            max_seq_length=max_seq_length,
            pad_to_max_length=pad_to_max_length
        )
        logger.info(f"Using local dataset from: {data_dir}")
        
    elif datasource == "huggingface":
        if not huggingface_dataset_name:
            raise ValueError("huggingface_dataset_name is required for huggingface datasource")
        
        data_args = DataTrainingArguments(
            dataset_source="huggingface",
            huggingface_dataset_name=huggingface_dataset_name,
            train_downsample=downsample,
            max_seq_length=max_seq_length,
            pad_to_max_length=pad_to_max_length
        )
        logger.info(f"Using HuggingFace dataset: {huggingface_dataset_name}")
        
    else:
        raise ValueError(f"Unknown datasource: {datasource}")
    
    model_args = ModelArguments(
        model_name_or_path=model_name_or_path,
        use_fast_tokenizer=use_fast_tokenizer
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer
    )
    
    logger.info("Loading and processing data...")
    processed_datasets, data_collator, label_to_id, id_to_label = load_and_process_data(
        data_args, model_args, tokenizer
    )
    
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        finetuning_task="text-classification"
    )
    config.name_or_path = model_name_or_path
    config.problem_type = "single_label_classification"
    config.label2id = label_to_id
    config.id2label = id_to_label
    
    model = CustomCodeClassifier(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        disable_tqdm=False,
    )
    
    model_callback = ModelCallback(model_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_detailed_metrics,
        callbacks=[model_callback]
    )
    # for batch in trainer.get_train_dataloader():
    #     print("First batch keys:", list(batch.keys()))
    #     print({k: v.shape for k, v in batch.items()})
    #     break

    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training completed. Metrics: {train_result.metrics}")
    
    logger.info("Evaluating on validation set...")
    val_metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {val_metrics}")
    
    if "test" in processed_datasets:
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(processed_datasets["test"])
        logger.info(f"Test metrics: {test_metrics}")
    
    save_final_model(model, tokenizer, config, model_dir, model_name_or_path, max_seq_length)
    
    logger.info(f"Training completed successfully! Model saved in: {model_dir}")
    return model_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DAR model")
    parser.add_argument("--datasource", choices=["local", "huggingface"], default="local",
                       help="Data source type")
    parser.add_argument("--data_dir", type=str, help="Data directory (for local datasource)", default='/kaggle/input/test-train/data')
    parser.add_argument("--huggingface_dataset_name", type=str, 
                       help="HuggingFace dataset name (for huggingface datasource)")
    parser.add_argument("--downsample", action="store_true", 
                       help="Use downsampled training data")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/unixcoder-base",
                       help="Base model name or path")
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True,
                       help="Use fast tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--pad_to_max_length", action="store_true", default=True,
                       help="Pad sequences to maximum length")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                       help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--model_dir", type=str, default=None,
                       help="Model output directory (default: model)")
    parser.add_argument("--run_name", type=str, default="dar-classification",
                       help="Name for the training run")
    parser.add_argument("--logging_steps", type=int, default=1,
                       help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args, _ = parser.parse_known_args()
    
    if args.datasource == "local" and not args.data_dir:
        parser.error("--data_dir is required when using local datasource")
    if args.datasource == "huggingface" and not args.huggingface_dataset_name:
        parser.error("--huggingface_dataset_name is required when using huggingface datasource")
    
    model_path = run(
        datasource=args.datasource,
        data_dir=args.data_dir,
        huggingface_dataset_name=args.huggingface_dataset_name,
        downsample=args.downsample,
        model_name_or_path=args.model_name_or_path,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        model_dir=args.model_dir,
        run_name=args.run_name,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        pad_to_max_length=args.pad_to_max_length,
        use_fast_tokenizer=args.use_fast_tokenizer
    )
    
    print(f"Training completed! Model saved at: {model_path}")