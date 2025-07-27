from typing import Dict, Any, Text
import numpy as np
from scipy.sparse import dok_matrix

from yakakrasa.core.nlu.pipeline import Component
from yakakrasa.core.nlu.tokenizer import Token

class Featurizer(Component):
    """
    A simple featurizer that creates a bag-of-words representation.
    """
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def fit(self, texts: list[list[Token]]):
        """Create vocabulary from a list of tokenized texts."""
        all_tokens = [token.text for text in texts for token in text]
        unique_tokens = sorted(list(set(all_tokens)))
        self.vocab = {token: i for i, token in enumerate(unique_tokens)}
        self.vocab_size = len(self.vocab)

    def process(self, message: Dict[Text, Any]) -> None:
        """Create a bag-of-words vector for a message."""
        tokens = message.get("tokens", [])
        if not self.vocab:
            # In a real scenario, you'd load a pre-trained vocab
            # or handle this case more gracefully.
            raise RuntimeError("Featurizer has not been fitted yet.")

        # Use dok_matrix for efficient sparse matrix creation
        vec = dok_matrix((1, self.vocab_size), dtype=np.float32)
        for token in tokens:
            if token.text in self.vocab:
                vec[0, self.vocab[token.text]] = 1.0
        
        message["features"] = vec.tocsr() 