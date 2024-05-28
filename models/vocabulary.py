"""
Based on the implementation from https://github.com/MolecularAI/reinvent-models
"""
from typing import List
import re
import numpy as np


class Vocabulary:
    """Stores the tokens and their conversion to Vocabulary indices."""

    def __init__(self, tokens=None, starting_id=0):
        self.tokens = {}
        self.current_id = starting_id

        # add tokens if they are provided - useful when wanting to incorporate specific chemistry (e.g., atoms, or ring syntax)
        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self.current_id = max(self.current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self.tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self.current_id)
        self.current_id += 1
        return self.current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self.tokens[token_or_id]
        del self.tokens[other_val]
        del self.tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self.tokens

    def __eq__(self, other_vocabulary):
        return self.tokens == other_vocabulary.tokens  # pylint: disable=W0212

    def __len__(self) -> int:
        return len(self.tokens) // 2  # since self.tokens stores a bidirectional mapping

    def encode(self, tokens):
        """Encodes a list of tokens as Vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for idx, token in enumerate(tokens):
            vocab_index[idx] = self.tokens[token]
        return vocab_index

    def decode(self, vocab_index):
        """Decodes a Vocabulary index matrix to a list of tokens."""
        tokens = []
        for idx in vocab_index:
            tokens.append(self[idx])
        return tokens

    def _add(self, token, idx):
        if idx not in self.tokens:
            self.tokens[token] = idx
            self.tokens[idx] = token
        else:
            raise ValueError("Index already present in Vocabulary.")

    def get_tokens(self) -> List[str]:
        """Returns the tokens from the Vocabulary."""
        return [t for t in self.tokens if isinstance(t, str)]
    
class SMILESTokenizer:
    """
    SMILES tokenizer based on REINVENT's implementation.
    """

    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for idx, split in enumerate(splitted):
                if idx % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi


def create_vocabulary(smiles: np.ndarray[str], tokenizer) -> Vocabulary:
    """Creates a Vocabulary given a dataset of SMILES."""
    tokens = set()
    for smi in smiles:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))

    vocabulary = Vocabulary()
    vocabulary.update(["$", "^"] + sorted(tokens))  # end token is 0 (also counts as padding)
    return vocabulary
