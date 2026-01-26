
import time
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from remember_me.core.csnp import CSNPManager
from remember_me.core.embedder import LocalEmbedder

class MockEmbedder:
    def __init__(self):
        self.dim = 384
    def __call__(self, text):
        if isinstance(text, list):
            return torch.randn(len(text), self.dim)
        return torch.randn(1, self.dim)

def benchmark():
    print("⚡ Starting CSNP Benchmark (High Load)...")

    # Increase context limit to 5000
    manager = CSNPManager(embedding_dim=384, context_limit=5000, embedder=MockEmbedder())

    # Fill Phase
    print("Filling buffer (5000 items)...")
    start = time.time()
    for i in range(5000):
        manager.update_state(f"User {i}", f"AI {i}")
    print(f"Fill (5000 items) took: {time.time() - start:.4f}s")

    # Force Invalidation & Retrieve Loop
    print("Benchmarking retrieve_context with invalidation (100 iter on 5000 items)...")
    start = time.time()
    for i in range(100):
        manager._context_cache = None
        _ = manager.retrieve_context()
    duration = time.time() - start
    print(f"Retrieve (Uncached, 5000 items) took: {duration:.4f}s ({duration/100*1000:.2f}ms per call)")

if __name__ == "__main__":
    benchmark()
