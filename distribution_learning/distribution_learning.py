"""
Adapted from https://github.com/MolecularAI/Reinvent.
"""
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils.utils import to_tensor

from models.model import Model
from distribution_learning.dataclass import DistributionLearningConfiguration
from distribution_learning.dataset.smiles_dataset import SMILESDataset


class DistributionLearningTrainer:
    """
    Distribution Learning by Maximum Likelihood Estimation (MLE) and using Teacher Forcing.
    Training objective is to maximize the likelihood of reproducing the dataset.

    Used to either:
        1. Pre-train a Prior 
        2. Fine-tune (same as transfer learning) an Agent
    """
    def __init__(
        self, 
        configuration: DistributionLearningConfiguration
    ):
        self.seed = configuration.seed
        self.learning_rate = configuration.learning_rate
        self.training_steps = configuration.training_steps
        self.batch_size = configuration.batch_size
        self.train_with_randomization = configuration.train_with_randomization

        self.train_dataset = SMILESDataset(
            agent = configuration.agent,
            dataset_path=configuration.training_dataset_path,
            batch_size=configuration.batch_size,
            transfer_learning=configuration.transfer_learning,
            randomize=configuration.train_with_randomization
        )

        self.val_dataset = SMILESDataset(
            agent = configuration.agent,
            dataset_path=configuration.validation_dataset_path,
            batch_size=configuration.batch_size,
            transfer_learning=configuration.transfer_learning,
            randomize=configuration.train_with_randomization
        )

        self.agent = Model.load_from_file(configuration.agent)
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=self.learning_rate)
  
    def run(self):
        for epoch in self.training_steps:
            # NOTE: Each epoch loops through the entire dataset
            train_dataloader, val_dataloader = self.setup_dataloaders()
            for _, batch in tqdm(enumerate(train_dataloader)):
                # 1. Compute the Negative Log-Likelihood (NLL) from Teacher Forcing
                loss = self.agent.likelihood(batch)
                # 2. Backpropagate
                self.backpropagate(loss)

    def backpropagate(self, loss: torch.Tensor) -> None:
        """Agent update via backpropagation."""
        loss = loss.mean()
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

    def setup_dataloaders(self):
        """
        Initialize the DataLoader for the training and validation datasets.
        """
        train_dataloader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=self.train_dataset.collate_fn
            )
        
        val_dataloader = DataLoader(
                dataset=self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=self.val_dataset.collate_fn
            )
        
        return train_dataloader, val_dataloader
