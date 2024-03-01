from dataclasses import dataclass

@dataclass
class DistributionLearningConfiguration:
    seed: int
    model_architecture: str
    agent: str
    training_steps: int
    batch_size: int
    learning_rate: float
    training_dataset_path: str
    train_with_randomization: bool = True,
    transfer_learning: bool = False
