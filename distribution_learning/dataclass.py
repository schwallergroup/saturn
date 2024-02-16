from dataclasses import dataclass

@dataclass
class DistributionLearningConfiguration:
    seed: int
    training_steps: int
    batch_size: int
    learning_rate: float
    training_dataset: str
    validation_dataset: str
    train_with_randomization: bool = True
