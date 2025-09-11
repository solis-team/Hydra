import transformers
transformers.logging.set_verbosity_error()

import os
import json
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
from transformers import AutoTokenizer, AutoConfig, AutoModel
import sys

current_dir = os.path.dirname(__file__)
training_dir = os.path.join(current_dir, 'training')
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

from model import CustomCodeClassifier


def load_inference_model(model_dir: str, device: str = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    config = AutoConfig.from_pretrained(model_dir)
    model = CustomCodeClassifier.from_pretrained(model_dir, config=config)

    model.to(device)
    model.eval()
    
    return model, tokenizer, {"max_seq_length": 512}


def infer_batch(texts: List[str], model_dir: str, threshold: float = 0.25, 
                device: str = None, model=None, tokenizer=None, model_info=None) -> List[Tuple[int, float]]:
    """
    Perform inference on a batch of texts.
    
    Args:
        texts: List of text strings to classify
        model_dir: Directory containing the saved model (used if model is None)
        threshold: Threshold for binary classification (default: 0.25)
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        model: Pre-loaded model (optional, will load if None)
        tokenizer: Pre-loaded tokenizer (optional, will load if None)
        model_info: Pre-loaded model info (optional, will load if None)
        
    Returns:
        List of tuples (predicted_class, probability_score) for each input text
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    if model is None or tokenizer is None or model_info is None:
        model, tokenizer, model_info = load_inference_model(model_dir, device)
    
    if not texts:
        return [(0, None)] * len(texts)
    
    max_length = model_info.get("max_seq_length", 512)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    predictions = []
    
    if isinstance(logits, torch.Tensor) and len(logits.shape) == 2:
        probs = F.softmax(logits, dim=1)
        for p in probs:
            score = p[1].item()  
            pred_class = 1 if score >= threshold else 0
            predictions.append((pred_class, score))
    else:
        predictions = [(0, None)] * len(texts)
    
    return predictions


def infer_single(text: str, model_dir: str, threshold: float = 0.25, 
                device: str = None) -> Tuple[int, float]:
    """
    Perform inference on a single text.
    
    Args:
        text: Text string to classify
        model_dir: Directory containing the saved model
        threshold: Threshold for binary classification (default: 0.25)
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        
    Returns:
        Tuple of (predicted_class, probability_score)
    """
    results = infer_batch([text], model_dir, threshold, device)
    return results[0]


class DARInferenceModel:
    """
    A class for holding a loaded model for multiple inference calls.
    This is more efficient when you need to run inference multiple times.
    """
    
    def __init__(self, model_dir: str, device: str = None):
        """
        Initialize the inference model.
        
        Args:
            model_dir: Directory containing the saved model
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_dir = model_dir
        self.device = device
        self.model, self.tokenizer, self.model_info = load_inference_model(model_dir, device)
    
    def predict(self, texts: Union[str, List[str]], threshold: float = 0.25) -> Union[Tuple[int, float], List[Tuple[int, float]]]:
        """
        Predict labels for input text(s).
        
        Args:
            texts: Single text string or list of text strings
            threshold: Threshold for binary classification (default: 0.25)
            
        Returns:
            Single tuple (class, score) if input is string, list of tuples if input is list
        """
        if isinstance(texts, str):
            return infer_batch([texts], self.model_dir, threshold, 
                             self.device, self.model, self.tokenizer, self.model_info)[0]
        else:
            return infer_batch(texts, self.model_dir, threshold, 
                             self.device, self.model, self.tokenizer, self.model_info)
    
    def predict_batch(self, texts: List[str], threshold: float = 0.25) -> List[Tuple[int, float]]:
        """
        Predict labels for a batch of texts.
        
        Args:
            texts: List of text strings
            threshold: Threshold for binary classification (default: 0.25)
            
        Returns:
            List of tuples (predicted_class, probability_score)
        """
        return infer_batch(texts, self.model_dir, threshold, 
                         self.device, self.model, self.tokenizer, self.model_info)
