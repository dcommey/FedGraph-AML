"""
FedGraph-AML: Federated Graph Learning for Cross-VASP AML Detection

Main entry point for running experiments.

Usage:
    # Run all experiments
    python main.py --all
    
    # Run baselines only
    python main.py --baselines
    
    # Run federated only
    python main.py --federated
    
    # Quick test (minimal epochs)
    python main.py --quick
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="FedGraph-AML: Federated Graph Learning for AML Detection"
    )
    
    # Experiment selection
    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--baselines", action="store_true",
                       help="Run baseline experiments only")
    parser.add_argument("--federated", action="store_true",
                       help="Run federated experiments only")
    
    # Configuration
    parser.add_argument("--strategy", type=str, default="temporal",
                       choices=["temporal", "random", "metis"],
                       help="Graph partitioning strategy")
    parser.add_argument("--clients", type=int, default=3,
                       help="Number of federated clients")
    parser.add_argument("--rounds", type=int, default=20,
                       help="Number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=3,
                       help="Local epochs per round")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Quick test mode
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with minimal epochs")
    
    # Semi-supervised learning
    parser.add_argument("--semi-supervised", action="store_true",
                       help="Use semi-supervised learning")
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.rounds = 3
        args.local_epochs = 1
        print("Quick mode: 3 rounds, 1 local epoch\n")
    
    # Default to running all if nothing specified
    if not (args.all or args.baselines or args.federated):
        args.all = True
    
    print("=" * 60)
    print("FedGraph-AML: Federated Graph Learning for AML Detection")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Clients: {args.clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Seed: {args.seed}")
    print(f"  Semi-supervised: {args.semi_supervised}")
    
    # Run experiments
    if args.all or args.baselines:
        print("\n" + "=" * 60)
        print("RUNNING BASELINE EXPERIMENTS")
        print("=" * 60)
        
        from experiments.run_baseline import run_baselines
        
        class BaselineArgs:
            strategy = args.strategy
            clients = args.clients
            seed = args.seed
        
        baseline_args = BaselineArgs()
        run_baselines(baseline_args)
    
    if args.all or args.federated:
        print("\n" + "=" * 60)
        print("RUNNING FEDERATED EXPERIMENTS")
        print("=" * 60)
        
        from experiments.run_federated import run_all_experiments
        
        class FedArgs:
            strategy = args.strategy
            clients = args.clients
            rounds = args.rounds
            local_epochs = args.local_epochs
            seed = args.seed
            semi_supervised = args.semi_supervised
            quick = args.quick
        
        fed_args = FedArgs()
        run_all_experiments(fed_args)
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
