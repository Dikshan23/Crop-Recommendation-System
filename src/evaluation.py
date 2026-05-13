"""
src/evaluation.py
=================
From-scratch model evaluation metrics.
"""
import numpy as np
from collections import defaultdict


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix from scratch.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label names
    
    Returns:
        matrix: np.ndarray of shape (n_classes, n_classes)
        label_list: List of class labels in order
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    n_classes = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        i = label_to_idx[true]
        j = label_to_idx[pred]
        matrix[i, j] += 1
    
    return matrix, labels


def classification_metrics_from_cm(cm, labels):
    """
    Calculate per-class and macro metrics from confusion matrix.
    """
    n_classes = len(labels)
    
    # Per-class metrics
    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Support
        support = cm[i, :].sum()
        
        per_class[label] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'support': support
        }
    
    # Macro averages
    macro_precision = np.mean([v['precision'] for v in per_class.values()])
    macro_recall = np.mean([v['recall'] for v in per_class.values()])
    macro_f1 = np.mean([v['f1_score'] for v in per_class.values()])
    
    # Weighted averages
    total_support = cm.sum()
    weighted_precision = sum(v['precision'] * v['support'] for v in per_class.values()) / total_support
    weighted_recall = sum(v['recall'] * v['support'] for v in per_class.values()) / total_support
    weighted_f1 = sum(v['f1_score'] * v['support'] for v in per_class.values()) / total_support
    
    # Overall accuracy
    accuracy = np.trace(cm) / total_support
    
    return {
        'accuracy': round(accuracy, 4),
        'macro_avg': {
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1_score': round(macro_f1, 4)
        },
        'weighted_avg': {
            'precision': round(weighted_precision, 4),
            'recall': round(weighted_recall, 4),
            'f1_score': round(weighted_f1, 4)
        },
        'per_class': per_class
    }


def evaluate_model(model, X_test, y_test, label_names=None):
    """
    Comprehensive model evaluation.
    """
    y_pred = model.predict(X_test)
    
    cm, labels = confusion_matrix(y_test, y_pred, label_names)
    metrics = classification_metrics_from_cm(cm, labels)
    
    # Add confusion matrix to results
    metrics['confusion_matrix'] = cm
    metrics['labels'] = labels
    
    # Identify problematic classes (low F1)
    problematic = [
        label for label, scores in metrics['per_class'].items()
        if scores['f1_score'] < 0.5
    ]
    metrics['low_performance_classes'] = problematic
    
    # Class balance analysis
    supports = [v['support'] for v in metrics['per_class'].values()]
    metrics['class_balance'] = {
        'min_samples': min(supports),
        'max_samples': max(supports),
        'imbalance_ratio': round(max(supports) / min(supports), 2)
    }
    
    return metrics