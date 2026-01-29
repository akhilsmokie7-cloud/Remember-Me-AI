
import torch
from remember_me.core.csnp import CSNPManager
from remember_me.core.embedder import LocalEmbedder

class MockEmbedder(LocalEmbedder):
    def __init__(self, device):
        self.device = device
        self._dim = 384

    @property
    def dim(self):
        return self._dim

def verify_device():
    print("--- Verifying CSNPManager Device Initialization ---")

    # Case 1: Default (should be CPU)
    csnp = CSNPManager(embedder=MockEmbedder("cpu"))
    print(f"Embedder Device: cpu")
    print(f"Memory Bank Device: {csnp.memory_bank.device}")

    # Case 2: Simulate 'cuda' (even if not available, we check if logic respects it)
    # Note: torch.zeros(..., device='cuda') will fail if no cuda.
    # So we can't fully test this if no GPU.
    # But we can check if CSNPManager *checks* the device.

    # If we inspect the code, we know it doesn't.
    # Let's just check if memory_bank is always cpu.

    if torch.cuda.is_available():
        csnp_gpu = CSNPManager(embedder=MockEmbedder("cuda"))
        print(f"Embedder Device: cuda")
        print(f"Memory Bank Device: {csnp_gpu.memory_bank.device}")
        if csnp_gpu.memory_bank.device.type != 'cuda':
             print("❌ BUG CONFIRMED: Memory Bank is on CPU while Embedder is on CUDA!")
        else:
             print("✓ Correct Device.")
    else:
        print("Skipping CUDA verification (no GPU).")

if __name__ == "__main__":
    verify_device()
