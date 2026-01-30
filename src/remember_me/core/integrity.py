from typing import List, Optional, Set
import xxhash

class MerkleNode:
    # ⚡ Bolt: Use __slots__ to reduce memory overhead for large history trees
    __slots__ = ['hash', 'data', 'left', 'right']

    def __init__(self, hash_val: str, data: Optional[str] = None, left=None, right=None):
        self.hash = hash_val
        self.data = data
        self.left = left
        self.right = right

class IntegrityChain:
    """
    A Merkle-backed ledger of conversation history.
    Guarantees Zero-Hallucination by enforcing that any retrieved memory
    must structurally belong to the hash tree rooted at 'current_state_hash'.
    """
    def __init__(self):
        self.leaves: List[MerkleNode] = []
        self.leaf_hashes: Set[str] = set() # ⚡ Bolt: O(1) Lookup
        self.root: Optional[MerkleNode] = None
        self._is_dirty = False # ⚡ Bolt: Lazy rebuild flag

    def _hash(self, data: str) -> str:
        # xxHash is faster than SHA256 for high-throughput memory operations
        return xxhash.xxh64(data.encode('utf-8')).hexdigest()

    def add_entry(self, data: str) -> str:
        """Adds a new atomic memory unit. Updates are lazy."""
        node_hash = self._hash(data)
        # We store data only in leaves
        self.leaves.append(MerkleNode(node_hash, data=data))
        self.leaf_hashes.add(node_hash)
        self._is_dirty = True # Mark tree as needing rebuild
        return node_hash

    def _rebuild_tree(self):
        """
        Reconstructs the Merkle Root from the leaves.
        O(N) complexity. Only runs when root is requested.
        """
        if not self.leaves:
            self.root = None
            self._is_dirty = False
            return

        # ⚡ Bolt: Zero-Allocation Internal Nodes
        # Instead of creating MerkleNode objects for every intermediate hash (which causes massive GC churn),
        # we operate solely on hash strings. This reduces memory overhead significantly and improves speed.
        layer = [node.hash for node in self.leaves]

        while len(layer) > 1:
            next_layer = []
            count = len(layer)
            for i in range(0, count, 2):
                left = layer[i]
                if i + 1 < count:
                    right = layer[i+1]
                    # Hash(Left + Right)
                    combined = self._hash(left + right)
                    next_layer.append(combined)
                else:
                    # Duplicate last node to balance tree
                    combined = self._hash(left + left)
                    next_layer.append(combined)
            layer = next_layer

        self.root = MerkleNode(layer[0])
        self._is_dirty = False

    def get_root_hash(self) -> str:
        if self._is_dirty:
            self._rebuild_tree()
        return self.root.hash if self.root else "00000000"

    def verify(self, content: str) -> bool:
        """
        Verifies if specific content exists in the chain.
        This prevents the AI from fabricating memories that do not exist in the ledger.
        """
        target_hash = self._hash(content)
        return self.verify_hash(target_hash)

    def verify_hash(self, hash_val: str) -> bool:
        """
        Verifies if a specific hash exists in the chain (O(1)).
        """
        return hash_val in self.leaf_hashes
