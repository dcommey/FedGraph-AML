"""
XGBoost Baseline for AML Detection

Implements the Jullum et al. (2020) baseline approach:
- XGBoost on tabular node features
- No graph topology (treats each transaction independently)

This is the baseline we aim to beat with graph-based approaches.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
from typing import Dict, Optional, Tuple
import torch


class XGBoostAMLBaseline:
    """
    XGBoost baseline for AML detection.
    
    This replicates the Jullum et al. (2020) approach where
    transactions are treated as independent tabular data.
    
    Key limitation: Ignores graph structure, missing:
    - Transaction flow patterns (cycles, fan-in/fan-out)
    - Temporal dependencies between transactions
    - Cross-institution patterns
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 10.0,
        random_state: int = 42
    ):
        """
        Initialize XGBoost baseline.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            scale_pos_weight: Weight for positive (illicit) class
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'tree_method': 'hist',  # Faster for large datasets
            'n_jobs': -1
        }
        self.model = None
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features [num_samples, num_features]
            y_train: Training labels [num_samples]
            X_val: Optional validation features
            y_val: Optional validation labels
            verbose: Print training progress
        """
        self.model = xgb.XGBClassifier(**self.params)
        
        eval_set = []
        # Only use validation set if it has both classes
        if X_val is not None and y_val is not None:
            if len(np.unique(y_val)) > 1:
                eval_set = [(X_val, y_val)]
            elif verbose:
                print("  Warning: Validation set has only one class, skipping eval_set")
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if eval_set else None,
            verbose=verbose
        )
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        proba = self.predict_proba(X)
        pred = (proba >= threshold).astype(int)
        
        metrics = {
            'f1': f1_score(y, pred, zero_division=0),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0),
            'roc_auc': roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.0,
            'pr_auc': average_precision_score(y, proba) if len(np.unique(y)) > 1 else 0.0,
        }
        
        return metrics
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[int, float]:
        """Get top-k most important features."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_k]
        
        return {int(idx): float(importance[idx]) for idx in indices}


def train_xgboost_baseline(
    data,  # PyG Data object
    client_mask: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> Tuple[XGBoostAMLBaseline, Dict[str, float]]:
    """
    Train XGBoost baseline on PyG data.
    
    Args:
        data: PyTorch Geometric Data object
        client_mask: Optional mask for client-specific training
        verbose: Print progress
        
    Returns:
        Tuple of (trained model, test metrics)
    """
    # Extract features and labels
    X = data.x.numpy()
    y = data.y.numpy()
    
    # Apply client mask if provided
    if client_mask is not None:
        client_mask = client_mask.numpy()
        train_mask = data.train_mask.numpy() & client_mask
        val_mask = data.val_mask.numpy() & client_mask
        test_mask = data.test_mask.numpy() & client_mask
    else:
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()
    
    # Filter out unlabeled data (y == -1)
    labeled_train = train_mask & (y != -1)
    labeled_val = val_mask & (y != -1)
    labeled_test = test_mask & (y != -1)
    
    X_train, y_train = X[labeled_train], y[labeled_train]
    X_val, y_val = X[labeled_val], y[labeled_val]
    X_test, y_test = X[labeled_test], y[labeled_test]
    
    if verbose:
        print(f"Training XGBoost: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test")
        print(f"Class balance (train): {(y_train == 1).sum()} illicit / {(y_train == 0).sum()} licit")
    
    # Train model
    model = XGBoostAMLBaseline()
    model.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    if verbose:
        print(f"\nTest Results:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
    
    return model, metrics


if __name__ == "__main__":
    # Test XGBoost baseline
    print("=" * 60)
    print("Testing XGBoost Baseline")
    print("=" * 60)
    
    import sys
    sys.path.append('..')
    from data.elliptic_loader import EllipticDataset
    
    # Load data
    dataset = EllipticDataset()
    data = dataset.load()
    
    # Train and evaluate
    model, metrics = train_xgboost_baseline(data, verbose=True)
    
    print("\n" + "=" * 60)
    print("Top 10 Feature Importances:")
    print("=" * 60)
    importance = model.get_feature_importance(top_k=10)
    for idx, imp in importance.items():
        print(f"  Feature {idx}: {imp:.4f}")
