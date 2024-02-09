from dataclasses import dataclass

from typing import List


@dataclass
class InceptionConfiguration:
    smiles: List[str]
    memory_size: int
    sample_size: int
    augmented_memory_mode_collapse_guard: bool
    # self.update_size = update_size
