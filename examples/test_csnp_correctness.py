
import time
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from remember_me.core.csnp import CSNPManager

class MockEmbedder:
    def __init__(self):
        self.dim = 384
    def __call__(self, text):
        if isinstance(text, list):
            return torch.randn(len(text), self.dim)
        return torch.randn(1, self.dim)

def test_correctness():
    print("⚡ Testing Correctness...")
    manager = CSNPManager(embedding_dim=384, context_limit=10, embedder=MockEmbedder())

    # 1. Fill (Incremental Cache should work)
    for i in range(5):
        manager.update_state(f"U{i}", f"A{i}")

    # Check cache
    ctx = manager.retrieve_context()
    lines = ctx.split('\n')
    assert len(lines) == 5
    print("   [OK] Incremental Cache (Fill Phase)")

    # 2. Add one more (Append)
    manager.update_state("U5", "A5")
    ctx = manager.retrieve_context()
    lines = ctx.split('\n')
    assert len(lines) == 6
    print("   [OK] Append working")

    # 3. Trigger Compression (Invalidate)
    # limit is 10. Add 5 more to reach 11 (trigger compress)
    for i in range(6, 12):
        manager.update_state(f"U{i}", f"A{i}")

    # Size should be 10 now (limit)
    assert manager.size == 10
    ctx = manager.retrieve_context()
    lines = ctx.split('\n')
    assert len(lines) == 10
    print("   [OK] Compression Cache Invalidation")

    # 4. Verify Uncached Retrieve (Correctness of list comp)
    manager._context_cache = None
    ctx2 = manager.retrieve_context()
    assert ctx == ctx2
    print("   [OK] Uncached Retrieve Integrity")

if __name__ == "__main__":
    test_correctness()
