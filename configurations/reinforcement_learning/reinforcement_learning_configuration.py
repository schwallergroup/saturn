from typing import Dict
from dataclasses import dataclass, field


@dataclass
class ReinforcementLearningConfiguration:
    prior: str
    agent: str
    n_steps: int
    oracle_limit: int
    sigma: int = 128
    learning_rate: float = 0.0001
    batch_size: int = 128
    margin_threshold: int = 50
    optimization_algorithm: str = "augmented_memory"
    specific_algorithm_parameters: Dict[str, float] = field(default_factory=lambda: {"top_k": 0.5,
                                                                                     "alpha": 0.5,
                                                                                     "update_frequency": 5
                                                                                     })
    augmented_memory: bool = True
    augmentation_rounds: int = 2
    selective_memory_purge: bool = True
