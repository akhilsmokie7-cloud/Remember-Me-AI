import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

print("[TEST] 1. Importing Sovereign OS Stack...")
try:
    from q_os_ultimate import Q_OS_Trinity
    print("   [OK] Import successful.")
except Exception as e:
    print(f"   [FAIL] Import failed: {e}")
    sys.exit(1)

print("[TEST] 2. Initializing Trinity Kernel (QDMA + CSNP + Yggdrasil)...")
try:
    # Initialize the OS
    os_kernel = Q_OS_Trinity()
    print("   [OK] Kernel initialized.")
except Exception as e:
    print(f"   [FAIL] Kernel init failed: {e}")
    sys.exit(1)

print("[TEST] 3. Verifying Local Brain Engine...")
server_path = os.path.join("brain_engine", "llama-server.exe")
model_path = os.path.join("brain_engine", "model.gguf")

if os.path.exists(server_path):
    print(f"   [OK] Engine found: {server_path}")
else:
    print(f"   [FAIL] Engine missing: {server_path}")
    sys.exit(1)

if os.path.exists(model_path):
    print(f"   [OK] Model found: {model_path}")
    # Get model size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"        Size: {size_mb:.2f} MB")
else:
    print(f"   [WARN] Model missing: {model_path} (This is expected if not downloaded yet)")

print("\n[SUCCESS] SYSTEM INTEGRITY VERIFIED.")
