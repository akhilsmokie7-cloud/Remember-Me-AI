import torch
from typing import List, Union

class LocalEmbedder:
    """
    Provides local, cost-free embeddings using HuggingFace's Sentence Transformers.
    Default model: 'all-MiniLM-L6-v2' (Small, fast, effective).
    """

    # ⚡ Bolt: Hardcoded dimensions for common models to allow lazy loading
    # This prevents loading the full model just to check .dim
    KNOWN_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        # ⚡ Bolt: Expanded known dimensions to prevent load triggers
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "intfloat/multilingual-e5-large": 1024
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.model = None

        # ⚡ Bolt: Use known dimension if available
        self._dim = self.KNOWN_DIMS.get(model_name, None)

    @property
    def dim(self) -> int:
        """
        Returns the embedding dimension. Triggers model load if not known.
        """
        if self._dim is None:
             # If we don't know the dimension, we MUST load the model to find out.
             self._ensure_model_loaded()
        return self._dim

    @dim.setter
    def dim(self, value: int):
        self._dim = value

    def _ensure_model_loaded(self):
        if self.model is None:
            print(f"⚡ Bolt: Lazy loading embedding model: {self.model_name} on {self.device}...")
            # ⚡ Bolt: Import here to prevent blocking startup time
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Update dim only if not set (or to confirm)
            loaded_dim = self.model.get_sentence_embedding_dimension()
            if self._dim is not None and self._dim != loaded_dim:
                print(f"⚠️ Warning: Model dimension mismatch. Expected {self._dim}, got {loaded_dim}. Updating.")
            self._dim = loaded_dim

    def __call__(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Embeds text into a torch tensor [N, D].
        """
        self._ensure_model_loaded()

        if isinstance(text, str):
            text = [text]

        embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device)

        # Ensure we return [N, D]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return embeddings
