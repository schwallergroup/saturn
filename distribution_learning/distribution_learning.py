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
from models.generator import Generator
from distribution_learning.dataclass import DistributionLearningConfiguration
from distribution_learning.dataset.smiles_dataset import SMILESDataset


class DistributionLearningTrainer:
    """
    Distribution Learning by Maximum Likelihood Estimation (MLE) and using Teacher Forcing.
    Training objective is to maximize the likelihood of reproducing the SMILES dataset.

    Used to either:
        1. Pre-train a Prior 
        2. Fine-tune (same as transfer learning) an Agent
    """
    def __init__(
        self, 
        configuration: DistributionLearningConfiguration
    ):
        # Training parameters
        self.seed = configuration.seed
        # TODO: Adaptive learning rate
        self.learning_rate = configuration.learning_rate
        self.training_steps = configuration.training_steps
        self.batch_size = configuration.batch_size
        self.transfer_learning = configuration.transfer_learning
        self.train_with_randomization = configuration.train_with_randomization

        self.train_dataset = SMILESDataset(
            agent=configuration.agent,
            dataset_path=configuration.training_dataset_path,
            batch_size=configuration.batch_size,
            transfer_learning=configuration.transfer_learning,
            randomize=configuration.train_with_randomization
        )

        self.val_dataset = SMILESDataset(
            agent=configuration.agent,
            dataset_path=configuration.validation_dataset_path,
            batch_size=configuration.batch_size,
            transfer_learning=configuration.transfer_learning,
            randomize=configuration.train_with_randomization
        )

        # Initialize model
        if self.transfer_learning:
            # Load the pre-trained Agent
            self.agent = Generator.load_from_file(configuration.agent)
        else:
            # Otherwise, train the Agent from scratch
            self.agent = Generator(
                model_architecture=configuration.model_architecture,
                vocabulary=self.train_dataset.vocabulary,
                tokenizer=self.train_dataset.tokenizer,
                network_params=None
            )

        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=self.learning_rate)
  
    def run(self):
        for epoch in range(1, self.training_steps + 1, 1):
            self.agent.network.train()
            # NOTE: Each epoch loops through the entire dataset
            train_dataloader, val_dataloader = self.setup_dataloaders()
            losses = []
            for _, batch in tqdm(enumerate(train_dataloader)):
                # 1. Compute the Negative Log-Likelihood (NLL) from Teacher Forcing
                batch = batch.to("cuda")
                loss = self.agent.likelihood(batch)
                losses.append(loss.mean().item())
                # 2. Backpropagate
                self.backpropagate(loss)

            self.agent.network.eval()
            sampled = []
            while len(sampled) < 1e4:
                sampled_batch, sampled_nlls = self.agent.sample_smiles(num=self.batch_size, batch_size=self.batch_size)
                sampled.extend(sampled_batch)
            valid = 0
            for s in sampled:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    if len(s) > 5:
                        print(s)
                    valid += 1
            # TODO: Compute success by sampling 10k SMILES and checking validity and distribution overlap (how to measure this?)
            print(f"Epoch {epoch} | NLL: {np.mean(losses)} | Valid: {round(valid / len(sampled)*100, 2)}%")
            # Save the trained Agent
            self.agent.save(f"{self.agent.model_architecture}.prior")

        # Save the trained Agent
        self.agent.save(f"{self.agent.model_architecture}.prior")

    def backpropagate(
        self, 
        loss: torch.Tensor
    ) -> None:
        """Policy update via backpropagation."""
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
