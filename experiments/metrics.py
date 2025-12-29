"""
Evaluation Metrics for AML Detection

Comprehensive metrics for evaluating anti-money laundering models,
accounting for severe class imbalance (~2% illicit transactions).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels [num_samples]
        y_pred: Predicted labels [num_samples]
        y_prob: Prediction probabilities [num_samples] (optional)
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = (y_true == y_pred).mean()
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    
    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # False Positive Rate (important for AML - alerts cost money)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Probability-based metrics (if available)
    if y_prob is not None:
        # Only compute if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find the optimal classification threshold.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        metric: Metric to optimize ("f1", "recall", "precision")
        
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_value = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == 'f1':
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            value = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            value = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = thresh
    
    return best_threshold, best_value


def compute_ppp_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_percentile: float = 1.0
) -> float:
    """
    Compute Precision at Prioritized Percentage (PPP) metric.
    
    This is similar to the metric used in Jullum et al. (2020):
    "What percentage of the top k% predictions are actually illicit?"
    
    This is important for AML because investigators can only
    review a limited number of cases per day.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        k_percentile: Top percentage to consider (e.g., 1.0 = top 1%)
        
    Returns:
        Precision at top k%
    """
    n = len(y_true)
    k = max(1, int(n * k_percentile / 100))
    
    # Get indices of top k predictions
    top_k_indices = np.argsort(y_prob)[-k:]
    
    # Compute precision in top k
    precision = y_true[top_k_indices].mean()
    
    return precision


def compute_lift(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_percentile: float = 1.0
) -> float:
    """
    Compute Lift at top k%.
    
    Lift = (Precision at k%) / (Base illicit rate)
    
    Shows how much better than random the model is at finding
    illicit transactions in the top k%.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        k_percentile: Top percentage to consider
        
    Returns:
        Lift value (>1 is better than random)
    """
    ppp = compute_ppp_metric(y_true, y_prob, k_percentile)
    base_rate = y_true.mean()
    
    if base_rate == 0:
        return 0.0
    
    return ppp / base_rate


class MetricsTracker:
    """
    Tracks metrics across multiple evaluation rounds.
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        
    def update(self, metrics: Dict[str, float], prefix: str = ""):
        """Add metrics from one round."""
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics."""
        summary = {}
        for key, values in self.history.items():
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'last': values[-1] if values else 0.0
            }
        return summary
    
    def get_best_round(self, metric: str) -> int:
        """Get the round with the best value for a metric."""
        if metric not in self.history:
            return 0
        return int(np.argmax(self.history[metric]))


def evaluate_model(
    model,  # nn.Module
    data,   # PyG Data
    mask: torch.Tensor,
    device: torch.device,
    thresholds: List[float] = [0.3, 0.5, 0.7]
) -> Dict[str, float]:
    """
    Evaluate a GNN model on specified data split.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        mask: Boolean mask for evaluation nodes
        device: Device for computation
        thresholds: Thresholds to evaluate
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    with torch.no_grad():
        data = data.to(device)
        logits, _ = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
    
    # Filter to labeled nodes
    eval_mask = mask & (data.y != -1)
    
    y_true = data.y[eval_mask].cpu().numpy()
    y_prob = probs[eval_mask].cpu().numpy()
    
    # Compute metrics at each threshold
    all_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob, thresh)
        for key, value in metrics.items():
            all_metrics[f'{key}@{thresh}'] = value
    
    # Optimal threshold metrics
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_prob, 'f1')
    all_metrics['optimal_threshold'] = opt_thresh
    all_metrics['optimal_f1'] = opt_f1
    
    # PPP metrics (important for practical AML)
    for k in [0.5, 1.0, 2.0, 5.0]:
        all_metrics[f'ppp@{k}%'] = compute_ppp_metric(y_true, y_prob, k)
        all_metrics[f'lift@{k}%'] = compute_lift(y_true, y_prob, k)
    
    return all_metrics


if __name__ == "__main__":
    # Test metrics
    print("=" * 60)
    print("Testing Evaluation Metrics")
    print("=" * 60)
    
    # Create dummy predictions (simulating imbalanced data)
    np.random.seed(42)
    n = 1000
    
    # 2% illicit rate
    y_true = np.zeros(n)
    y_true[:20] = 1  # 20 illicit out of 1000
    np.random.shuffle(y_true)
    
    # Simulated model probabilities
    y_prob = np.random.beta(2, 5, n)  # Mostly low probabilities
    y_prob[y_true == 1] = np.random.beta(5, 2, 20)  # Higher for illicit
    
    y_pred = (y_prob > 0.5).astype(int)
    
    print("\nDataset:")
    print(f"  Total: {n}, Illicit: {y_true.sum():.0f} ({y_true.mean()*100:.1f}%)")
    
    print("\nBasic Metrics:")
    metrics = compute_metrics(y_true, y_pred, y_prob)
    for key, value in metrics.items():
        if not key.startswith('true_') and not key.startswith('false_'):
            print(f"  {key}: {value:.4f}")
    
    print("\nOptimal Threshold:")
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_prob, 'f1')
    print(f"  Threshold: {opt_thresh:.2f}, F1: {opt_f1:.4f}")
    
    print("\nPPP Metrics (Practical AML):")
    for k in [1.0, 2.0, 5.0]:
        ppp = compute_ppp_metric(y_true, y_prob, k)
        lift = compute_lift(y_true, y_prob, k)
        print(f"  Top {k}%: PPP={ppp:.4f}, Lift={lift:.2f}x")
