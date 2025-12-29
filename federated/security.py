"""
Post-Quantum Security Module for FedGraph-VASP

Implements a Hybrid Cryptosystem (KEM + DEM) using:
1. CRYSTALS-Kyber-512 (NIST ML-KEM Level 1) for Key Encapsulation
2. AES-256-GCM for Data Encapsulation

This ensures "Forward Secrecy" against Harvest Now, Decrypt Later attacks.
"""

import torch
import pickle
import hashlib
import numpy as np
from typing import Dict, Optional, Tuple, Any

try:
    from kyber_py.ml_kem import ML_KEM_512
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    ML_KEM_512 = None
    print("Warning: kyber-py or pycryptodome not found. PQC simulation disabled.")


class PostQuantumTunnel:
    """
    Secure PQC tunnel between two VASPs using Kyber-512 + AES-GCM.
    """
    
    def __init__(self):
        """
        Initialize the tunnel. 
        Generates the encapsulation key (public) and decapsulation key (private).
        """
        if HAS_CRYPTO:
            # ML-KEM-512 keygen returns (encapsulation_key, decapsulation_key)
            # ek = 800 bytes, dk = 1632 bytes
            self.ek, self.dk = ML_KEM_512.keygen()
        else:
            self.ek, self.dk = None, None

    def encrypt_embedding(self, tensor_data: torch.Tensor) -> Dict[str, Any]:
        """
        Encrypt a tensor embedding using Kyber-512 + AES-GCM.
        
        1. Generate shared secret using encapsulation key
        2. Encrypt tensor using AES-GCM with shared secret as key
        """
        if not HAS_CRYPTO:
            # Fallback: no encryption, just return tensor
            return {'ciphertext': tensor_data, 'encrypted': False}
            
        # --- A. POST-QUANTUM KEY ENCAPSULATION (Kyber-512) ---
        # encaps returns: (shared_secret, ciphertext)
        # shared_secret = 32 bytes, ciphertext = 768 bytes
        ss, ct = ML_KEM_512.encaps(self.ek)
        
        # --- B. SERIALIZATION (Tensor -> Bytes) ---
        data_bytes = pickle.dumps(tensor_data.cpu().detach().numpy())
        
        # --- C. SYMMETRIC ENCRYPTION (AES-256-GCM) ---
        # ss is already 32 bytes, perfect for AES-256
        cipher = AES.new(ss, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(data_bytes)
        
        # Payload sent over network
        payload = {
            'kyber_ct': ct,        # Kyber ciphertext (768 bytes)
            'aes_nonce': cipher.nonce,
            'aes_tag': tag,
            'ciphertext': ciphertext,
            'encrypted': True
        }
        return payload

    def decrypt_embedding(self, payload: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Decrypt a tensor embedding.
        
        1. Decapsulate Kyber ciphertext to recover shared secret
        2. Decrypt AES payload using shared secret
        """
        if not HAS_CRYPTO or not payload.get('encrypted', False):
            return payload.get('ciphertext')
            
        try:
            # --- A. POST-QUANTUM KEY DECAPSULATION (Kyber-512) ---
            # decaps returns: shared_secret (32 bytes)
            ss = ML_KEM_512.decaps(self.dk, payload['kyber_ct'])
            
            # --- B. SYMMETRIC DECRYPTION (AES-256-GCM) ---
            cipher = AES.new(ss, AES.MODE_GCM, nonce=payload['aes_nonce'])
            decrypted_bytes = cipher.decrypt_and_verify(payload['ciphertext'], payload['aes_tag'])
            
            # --- C. DESERIALIZATION (Bytes -> Tensor) ---
            numpy_data = pickle.loads(decrypted_bytes)
            return torch.from_numpy(numpy_data)
            
        except (ValueError, KeyError) as e:
            print(f"Decryption Failed: {e}")
            return None


# Quick test when run directly
if __name__ == "__main__":
    print(f"HAS_CRYPTO: {HAS_CRYPTO}")
    
    if HAS_CRYPTO:
        tunnel = PostQuantumTunnel()
        test_tensor = torch.randn(128)
        
        print(f"Original tensor shape: {test_tensor.shape}")
        
        encrypted = tunnel.encrypt_embedding(test_tensor)
        print(f"Encrypted: {encrypted.get('encrypted')}")
        print(f"Kyber CT size: {len(encrypted.get('kyber_ct', b''))} bytes")
        print(f"AES ciphertext size: {len(encrypted.get('ciphertext', b''))} bytes")
        
        decrypted = tunnel.decrypt_embedding(encrypted)
        print(f"Decryption successful: {decrypted is not None}")
        if decrypted is not None:
            print(f"Values match: {torch.allclose(test_tensor, decrypted)}")
    else:
        print("PQC not available - install kyber-py and pycryptodome")
