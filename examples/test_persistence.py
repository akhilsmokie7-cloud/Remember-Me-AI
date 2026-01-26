import os
import torch
import shutil
from remember_me.core.csnp import CSNPManager

def test_persistence():
    print(">>> TESTING PERSISTENCE AND HASH BUFFER...")
    filename = "test_state.pth"

    # 1. Create State
    print("1. Creating initial state...")
    csnp = CSNPManager(context_limit=3)
    csnp.update_state("User: Hi", "AI: Hello")
    csnp.update_state("User: Fact", "AI: Sky is blue")

    # Verify hash buffer exists and matches
    assert len(csnp.hash_buffer) == 2
    assert len(csnp.text_buffer) == 2
    print(f"   Hash Buffer Size: {len(csnp.hash_buffer)} (Correct)")

    # 2. Save State
    print(f"2. Saving state to {filename}...")
    csnp.save_state(filename)

    # 3. Load State into new instance
    print("3. Loading state into new instance...")
    csnp2 = CSNPManager(context_limit=3)
    csnp2.load_state(filename)

    # 4. Verify loaded state
    print("4. Verifying loaded state...")
    assert len(csnp2.hash_buffer) == 2, f"Hash buffer size mismatch: {len(csnp2.hash_buffer)}"
    assert csnp2.hash_buffer == csnp.hash_buffer, "Hash buffer content mismatch"
    assert csnp2.text_buffer == csnp.text_buffer, "Text buffer content mismatch"

    # Verify retrieval works
    context = csnp2.retrieve_context()
    assert "Sky is blue" in context
    print("   Context retrieval verified.")

    # 5. Test Legacy Load (Simulated)
    print("5. Testing legacy load (Simulating missing hash_buffer)...")
    # Manually corrupt the file to remove hash_buffer
    state_dict = torch.load(filename)
    del state_dict["hash_buffer"]
    torch.save(state_dict, "test_legacy.pth")

    csnp3 = CSNPManager(context_limit=3)
    csnp3.load_state("test_legacy.pth")

    assert len(csnp3.hash_buffer) == 2, "Failed to regenerate hash buffer"
    assert csnp3.hash_buffer == csnp.hash_buffer, "Regenerated hash buffer mismatch"
    print("   Legacy regeneration verified.")

    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists("test_legacy.pth"):
        os.remove("test_legacy.pth")

    print("\n✓ PERSISTENCE VERIFIED")

if __name__ == "__main__":
    test_persistence()
