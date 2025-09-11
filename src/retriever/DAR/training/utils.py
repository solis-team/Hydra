import numpy as np
import evaluate
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import random
import os
import torch


def seed_all(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed_value


    

def compute_detailed_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    
    recall_true = recall[1] if len(recall) > 1 else 0
    precision_true = precision[1] if len(precision) > 1 else 0
    f1_true = f1[1] if len(f1) > 1 else 0
    recall_false = recall[0] if len(recall) > 0 else 0
    precision_false = precision[0] if len(precision) > 0 else 0
    f1_false = f1[0] if len(f1) > 0 else 0
    
    overall_recall = recall.mean()
    overall_precision = precision.mean()
    
    return {
        "accuracy": acc,
        "recall": overall_recall,
        "precision": overall_precision,
        "recall_true": recall_true,
        "precision_true": precision_true,
        "f1_true": f1_true,
        "recall_false": recall_false,
        "precision_false": precision_false,
        "f1_false": f1_false
    }


