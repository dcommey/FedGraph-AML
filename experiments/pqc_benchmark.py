"""
PQC Benchmark: Kyber-512 + AES-256-GCM Performance Analysis

This script measures the actual overhead of the Post-Quantum
cryptographic tunnel used in FedGraph-VASP.

Metrics:
- Encryption latency (ms)
- Decryption latency (ms)
- Throughput (embeddings/second)
- Ciphertext expansion ratio
- Memory footprint

Usage:
    python experiments/pqc_benchmark.py
"""

import sys
import json
import time
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config

# Try to import the PQC module
try:
    from federated.security import PostQuantumTunnel, HAS_CRYPTO
except ImportError:
    HAS_CRYPTO = False
    PostQuantumTunnel = None


# Benchmark parameters
EMBEDDING_DIMS = [64, 128, 256]
BATCH_SIZES = [1, 10, 50, 100, 200]
NUM_ITERATIONS = 100  # Per configuration


def get_hardware_info() -> Dict[str, str]:
    """Collect hardware information for reproducibility."""
    import os
    
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_device"] = torch.cuda.get_device_name(0)
    else:
        info["cuda_available"] = False
    
    # Try to get CPU info
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count()
        info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        info["cpu_count"] = "unknown"
        info["memory_gb"] = "unknown"
    
    return info


def benchmark_single_embedding(tunnel: 'PostQuantumTunnel', embedding_dim: int) -> Dict[str, float]:
    """Benchmark encrypt/decrypt for a single embedding."""
    
    embedding = torch.randn(embedding_dim)
    
    # Encryption
    enc_times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        encrypted = tunnel.encrypt_embedding(embedding)
        enc_times.append((time.perf_counter() - start) * 1000)  # ms
    
    # Decryption
    dec_times = []
    for _ in range(NUM_ITERATIONS):
        encrypted = tunnel.encrypt_embedding(embedding)
        start = time.perf_counter()
        decrypted = tunnel.decrypt_embedding(encrypted)
        dec_times.append((time.perf_counter() - start) * 1000)  # ms
    
    # Ciphertext expansion
    original_size = embedding.numel() * 4  # float32 = 4 bytes
    encrypted_size = len(encrypted) if isinstance(encrypted, bytes) else sys.getsizeof(encrypted)
    
    return {
        "enc_mean_ms": np.mean(enc_times),
        "enc_std_ms": np.std(enc_times, ddof=1),
        "dec_mean_ms": np.mean(dec_times),
        "dec_std_ms": np.std(dec_times, ddof=1),
        "original_bytes": original_size,
        "encrypted_bytes": encrypted_size,
        "expansion_ratio": encrypted_size / original_size if original_size > 0 else 0
    }


def benchmark_batch(tunnel: 'PostQuantumTunnel', embedding_dim: int, batch_size: int) -> Dict[str, float]:
    """Benchmark encrypt/decrypt for a batch of embeddings."""
    
    embeddings = [torch.randn(embedding_dim) for _ in range(batch_size)]
    
    # Batch encryption
    batch_enc_times = []
    for _ in range(NUM_ITERATIONS // 10):  # Fewer iterations for large batches
        start = time.perf_counter()
        for emb in embeddings:
            tunnel.encrypt_embedding(emb)
        batch_enc_times.append((time.perf_counter() - start) * 1000)  # ms
    
    throughput = batch_size / (np.mean(batch_enc_times) / 1000)  # embeddings/sec
    
    return {
        "batch_size": batch_size,
        "total_enc_time_ms": np.mean(batch_enc_times),
        "per_embedding_ms": np.mean(batch_enc_times) / batch_size,
        "throughput_per_sec": throughput
    }


def get_kyber_security_params() -> Dict[str, any]:
    """Document Kyber-512 security parameters."""
    return {
        "algorithm": "CRYSTALS-Kyber-512",
        "nist_level": 1,
        "classical_security_bits": 128,
        "quantum_security_bits": 64,
        "public_key_bytes": 800,
        "secret_key_bytes": 1632,
        "ciphertext_bytes": 768,
        "shared_secret_bytes": 32,
        "failure_probability": "2^-139",
        "standardization": "NIST FIPS 203 (ML-KEM)",
        "reference": "Bos et al., 'CRYSTALS-Kyber', EuroS&P 2018"
    }


def main():
    """Run complete PQC benchmark suite."""
    print("=" * 70)
    print("PQC BENCHMARK: Kyber-512 + AES-256-GCM Performance Analysis")
    print("=" * 70)
    
    # Hardware info
    hw_info = get_hardware_info()
    print("\nHardware Configuration:")
    for k, v in hw_info.items():
        print(f"  {k}: {v}")
    
    # Check if PQC is available
    if not HAS_CRYPTO:
        print("\n⚠️  WARNING: kyber-py or pycryptodome not installed.")
        print("   Running in SIMULATION mode (no actual encryption).")
        print("   Install with: pip install kyber-py pycryptodome")
    
    # Initialize tunnel
    if PostQuantumTunnel is not None:
        tunnel = PostQuantumTunnel()
    else:
        print("\n❌ Cannot proceed without PostQuantumTunnel")
        return
    
    # Kyber security parameters
    print("\n" + "=" * 70)
    print("KYBER-512 SECURITY PARAMETERS")
    print("=" * 70)
    security_params = get_kyber_security_params()
    for k, v in security_params.items():
        print(f"  {k}: {v}")
    
    all_results = {
        "hardware": hw_info,
        "security_parameters": security_params,
        "has_real_crypto": HAS_CRYPTO,
        "single_embedding_benchmarks": [],
        "batch_benchmarks": []
    }
    
    # Benchmark single embeddings
    print("\n" + "=" * 70)
    print("SINGLE EMBEDDING BENCHMARKS")
    print("=" * 70)
    print(f"{'Dim':<10} {'Enc (ms)':<15} {'Dec (ms)':<15} {'Expansion':<12}")
    print("-" * 52)
    
    for dim in EMBEDDING_DIMS:
        result = benchmark_single_embedding(tunnel, dim)
        result["embedding_dim"] = dim
        all_results["single_embedding_benchmarks"].append(result)
        
        print(f"{dim:<10} {result['enc_mean_ms']:.3f}±{result['enc_std_ms']:.3f}    "
              f"{result['dec_mean_ms']:.3f}±{result['dec_std_ms']:.3f}    "
              f"{result['expansion_ratio']:.2f}x")
    
    # Benchmark batches
    print("\n" + "=" * 70)
    print("BATCH BENCHMARKS (Embedding Dim = 128)")
    print("=" * 70)
    print(f"{'Batch Size':<12} {'Total (ms)':<15} {'Per Emb (ms)':<15} {'Throughput':<15}")
    print("-" * 57)
    
    for batch_size in BATCH_SIZES:
        result = benchmark_batch(tunnel, 128, batch_size)
        all_results["batch_benchmarks"].append(result)
        
        print(f"{batch_size:<12} {result['total_enc_time_ms']:.2f}          "
              f"{result['per_embedding_ms']:.4f}          "
              f"{result['throughput_per_sec']:.0f}/s")
    
    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    
    # Find typical case: 128-dim, 200-node batch
    batch_200 = next((b for b in all_results["batch_benchmarks"] if b["batch_size"] == 200), None)
    if batch_200:
        print(f"For a typical boundary batch (200 nodes, 128-dim embeddings):")
        print(f"  Total encryption time: {batch_200['total_enc_time_ms']:.1f}ms")
        print(f"  Per-embedding overhead: {batch_200['per_embedding_ms']:.2f}ms")
        print(f"  Throughput: {batch_200['throughput_per_sec']:.0f} embeddings/second")
    
    single_128 = next((s for s in all_results["single_embedding_benchmarks"] if s["embedding_dim"] == 128), None)
    if single_128:
        print(f"  Ciphertext expansion: {single_128['expansion_ratio']:.2f}x")
    
    # Save results
    config = get_config()
    results_dir = Path(config.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"pqc_benchmark_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    main()
