from dataclasses import dataclass, field
from typing import List

@dataclass
class ExperienceReplayParameters:
    memory_size: int = 100
    sample_size: int = 10
    smiles: List[str] = field(default_factory=list)
    augmented_memory_mode_collapse_guard: bool = False
