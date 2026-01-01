"""
Rigorous Evaluation: Multi-Seed Statistical Validation

This script runs FedGraph-VASP and baselines with multiple random seeds
to provide statistically valid results with mean ± std.

Usage:
    python experiments/rigorous_evaluation.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.elliptic_loader import EllipticDataset
from data.partitioner import create_federated_data
from models.gnn import create_model
from federated.client import create_clients
from federated.server import FederatedServer
from federated.boundary_exchange import BoundaryEmbeddingBuffer


# Configuration for rigorous evaluation
SEEDS = [42, 123, 456, 789, 2024]
NUM_ROUNDS = 100  # Extended training for publication results (Paper Setting)
LOCAL_EPOCHS = 3
NUM_CLIENTS = 3


@dataclass
class ExperimentResult:
    """Container for a single experiment run."""
    seed: int
    method: str
    partition_strategy: str
    best_f1: float
    best_precision: float
    best_recall: float
    final_f1: float
    training_time: float


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_experiment(
    data,
    seed: int,
    method: str,  # "local", "fedavg", "fedgraph"
    partition_strategy: str,
    config,
    device: torch.device,
    cached_partition: tuple = None  # Optional cached (client_data, silos, stats)
) -> ExperimentResult:
    """Run a single experiment with given seed and method."""
    
    set_seed(seed)
    start_time = time.time()
    
    # Use cached partition if provided, otherwise compute
    if cached_partition is not None:
        client_data, silos, partition_stats = cached_partition
    else:
        client_data, silos, partition_stats = create_federated_data(
            data,
            num_clients=NUM_CLIENTS,
            strategy=partition_strategy
        )
    
    if method == "local":
        # Local baseline: train only on first client, evaluate on its test set
        # This represents a siloed VASP
        model = create_model(
            config.model.model_type,
            in_channels=data.num_features,
            hidden_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.federated.learning_rate,
            weight_decay=5e-4
        )
        
        # Train on all silos independently, report average
        all_f1s = []
        all_precisions = []
        all_recalls = []
        
        for cd in client_data:
            cd = cd.to(device)
            model_copy = create_model(
                config.model.model_type,
                in_channels=data.num_features,
                hidden_channels=config.model.hidden_channels,
                num_layers=config.model.num_layers
            ).to(device)
            
            opt = torch.optim.Adam(model_copy.parameters(), lr=0.01)
            
            for epoch in range(NUM_ROUNDS * LOCAL_EPOCHS):
                model_copy.train()
                opt.zero_grad()
                out, _ = model_copy(cd.x, cd.edge_index)  # Unpack tuple
                
                # Create proper boolean mask for labeled training data
                labeled_mask = (cd.y >= 0).bool()  # Explicit bool
                train_mask = cd.train_mask.bool() & labeled_mask
                
                if train_mask.sum() > 0:
                    # Get indices where mask is True
                    train_idx = torch.where(train_mask)[0]
                    labels = cd.y[train_idx].float()  # Binary labels as float
                    # Binary cross entropy (model outputs 1D logits)
                    loss = F.binary_cross_entropy_with_logits(out[train_idx], labels)
                    loss.backward()
                    opt.step()
            
            # Evaluate
            model_copy.eval()
            with torch.no_grad():
                out, _ = model_copy(cd.x, cd.edge_index)  # Unpack tuple
                pred = (out > 0).long()  # Binary prediction (threshold at 0 for logits)
                
                # Create proper boolean mask for labeled test data
                labeled_mask = (cd.y >= 0).bool()
                test_mask = cd.test_mask.bool() & labeled_mask
                
                if test_mask.sum() > 0:
                    test_idx = torch.where(test_mask)[0]
                    y_true = cd.y[test_idx].cpu().numpy()
                    y_pred = pred[test_idx].cpu().numpy()
                    
                    # Compute metrics
                    from sklearn.metrics import f1_score, precision_score, recall_score
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    prec = precision_score(y_true, y_pred, zero_division=0)
                    rec = recall_score(y_true, y_pred, zero_division=0)
                    
                    all_f1s.append(f1)
                    all_precisions.append(prec)
                    all_recalls.append(rec)
        
        best_f1 = np.mean(all_f1s) if all_f1s else 0
        best_precision = np.mean(all_precisions) if all_precisions else 0
        best_recall = np.mean(all_recalls) if all_recalls else 0
        final_f1 = best_f1
        
    else:
        # Federated methods (fedavg or fedgraph)
        use_boundary = (method == "fedgraph")
        
        template_model = create_model(
            config.model.model_type,
            in_channels=data.num_features,
            hidden_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers
        )
        
        clients = create_clients(
            model=template_model,
            client_data_list=client_data,
            learning_rate=config.federated.learning_rate,
            boundary_loss_weight=config.federated.boundary_loss_weight,
            pseudo_label_weight=config.federated.pseudo_label_weight,
            device=str(device)
        )
        
        server = FederatedServer(template_model)
        
        boundary_buffer = BoundaryEmbeddingBuffer(
            embedding_dim=config.model.hidden_channels,
            use_pqc=False  # Default: False for speed. Set to True to verify PQC (add ~1000x overhead in Python)
        ) if use_boundary else None
        
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        history = []
        
        for round_num in range(NUM_ROUNDS):
            # Update boundary buffer if using FedGraph
            if use_boundary and round_num > 0:
                for client in clients:
                    result = client.get_boundary_embeddings()
                    if result is not None:
                        indices, embeddings = result
                        boundary_buffer.update(client.client_id, indices, embeddings)
            
            # Run training round
            round_metrics = server.run_round(
                clients=clients,
                local_epochs=LOCAL_EPOCHS,
                use_boundary_exchange=use_boundary,
                boundary_buffer=boundary_buffer.buffer if boundary_buffer else None,
                use_unlabeled=False,
                verbose=False
            )
            
            # Progress indicator every 5 rounds
            if round_num % 5 == 4:
                print(f"    Round {round_num+1}/{NUM_ROUNDS}, F1={round_metrics.get('avg_test_f1', 0):.4f}")
            test_f1 = round_metrics.get('avg_test_f1', 0)
            test_prec = round_metrics.get('avg_test_precision', 0)
            test_rec = round_metrics.get('avg_test_recall', 0)
            
            history.append(test_f1)
            
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_precision = test_prec
                best_recall = test_rec
        
        final_f1 = history[-1] if history else 0
    
    training_time = time.time() - start_time
    
    return ExperimentResult(
        seed=seed,
        method=method,
        partition_strategy=partition_strategy,
        best_f1=best_f1,
        best_precision=best_precision,
        best_recall=best_recall,
        final_f1=final_f1,
        training_time=training_time
    )


def compute_statistics(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Compute mean, std, and confidence intervals from multiple runs."""
    f1_scores = [r.best_f1 for r in results]
    precisions = [r.best_precision for r in results]
    recalls = [r.best_recall for r in results]
    times = [r.training_time for r in results]
    
    n = len(f1_scores)
    
    # 95% confidence interval
    ci_multiplier = stats.t.ppf(0.975, n - 1) if n > 1 else 0
    
    return {
        "n_runs": n,
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores, ddof=1) if n > 1 else 0,
        "f1_ci95": ci_multiplier * np.std(f1_scores, ddof=1) / np.sqrt(n) if n > 1 else 0,
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions, ddof=1) if n > 1 else 0,
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls, ddof=1) if n > 1 else 0,
        "time_mean": np.mean(times),
        "time_std": np.std(times, ddof=1) if n > 1 else 0,
        "raw_f1_scores": f1_scores
    }


def perform_ttest(results_a: List[ExperimentResult], results_b: List[ExperimentResult]) -> Dict[str, float]:
    """Perform paired t-test between two methods."""
    f1_a = [r.best_f1 for r in results_a]
    f1_b = [r.best_f1 for r in results_b]
    
    # Paired t-test (same seeds used)
    t_stat, p_value = stats.ttest_rel(f1_a, f1_b)
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01
    }


def main():
    """Run the complete rigorous evaluation."""
    print("=" * 70)
    print("RIGOROUS EVALUATION: Multi-Seed Statistical Validation")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}, Clients: {NUM_CLIENTS}")
    
    config = get_config()
    device = torch.device(config.experiment.device)
    print(f"Device: {device}")
    
    # Load data once
    print("\nLoading Elliptic dataset...")
    dataset = EllipticDataset(root=str(config.data.data_root))
    data = dataset.load()
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    
    # Methods to evaluate
    methods = ["local", "fedavg", "fedgraph"]
    # Realistic partitioning - ~10% cross-edges like real VASPs
    partition_strategy = "metis"
    
    all_results: Dict[str, List[ExperimentResult]] = {m: [] for m in methods}
    
    # Pre-compute partitions for each seed (spectral clustering is slow)
    print("\nPre-computing partitions for all seeds (this takes a while)...")
    cached_partitions = {}
    for seed in SEEDS:
        set_seed(seed)
        print(f"  Partitioning for seed {seed}...", end=" ", flush=True)
        partition = create_federated_data(
            data,
            num_clients=NUM_CLIENTS,
            strategy=partition_strategy
        )
        cached_partitions[seed] = partition
        print("done.")
    
    # Run experiments using cached partitions
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        
        for method in methods:
            print(f"  Running {method}...", end=" ", flush=True)
            result = run_single_experiment(
                data=data,
                seed=seed,
                method=method,
                partition_strategy=partition_strategy,
                config=config,
                device=device,
                cached_partition=cached_partitions[seed]
            )
            all_results[method].append(result)
            print(f"F1={result.best_f1:.4f} ({result.training_time:.1f}s)")
    
    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<15} {'F1 (mean±std)':<20} {'Precision':<15} {'Recall':<15}")
    print("-" * 70)
    
    stats_dict = {}
    for method in methods:
        method_stats = compute_statistics(all_results[method])
        stats_dict[method] = method_stats
        
        f1_str = f"{method_stats['f1_mean']:.4f} ± {method_stats['f1_std']:.4f}"
        prec_str = f"{method_stats['precision_mean']:.4f}"
        rec_str = f"{method_stats['recall_mean']:.4f}"
        
        print(f"{method:<15} {f1_str:<20} {prec_str:<15} {rec_str:<15}")
    
    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    
    # FedGraph vs FedAvg
    ttest_fg_fa = perform_ttest(all_results["fedgraph"], all_results["fedavg"])
    print(f"\nFedGraph-VASP vs FedAvg:")
    print(f"  t-statistic: {ttest_fg_fa['t_statistic']:.4f}")
    print(f"  p-value: {ttest_fg_fa['p_value']:.4f}")
    print(f"  Significant at α=0.05: {ttest_fg_fa['significant_at_005']}")
    
    # FedGraph vs Local
    ttest_fg_local = perform_ttest(all_results["fedgraph"], all_results["local"])
    print(f"\nFedGraph-VASP vs Local:")
    print(f"  t-statistic: {ttest_fg_local['t_statistic']:.4f}")
    print(f"  p-value: {ttest_fg_local['p_value']:.4f}")
    print(f"  Significant at α=0.05: {ttest_fg_local['significant_at_005']}")
    
    # Improvement percentages
    fg_mean = stats_dict["fedgraph"]["f1_mean"]
    fa_mean = stats_dict["fedavg"]["f1_mean"]
    local_mean = stats_dict["local"]["f1_mean"]
    
    print(f"\n📊 Improvement vs Local: {((fg_mean - local_mean) / local_mean * 100):+.1f}%")
    print(f"📊 Improvement vs FedAvg: {((fg_mean - fa_mean) / fa_mean * 100):+.1f}%")
    
    # Save results
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"rigorous_evaluation_{timestamp}.json"
    
    save_data = {
        "config": {
            "seeds": SEEDS,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "num_clients": NUM_CLIENTS,
            "partition_strategy": partition_strategy
        },
        "statistics": stats_dict,
        "ttests": {
            "fedgraph_vs_fedavg": ttest_fg_fa,
            "fedgraph_vs_local": ttest_fg_local
        },
        "raw_results": {method: [asdict(r) for r in results] for method, results in all_results.items()}
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return save_data


if __name__ == "__main__":
    main()
