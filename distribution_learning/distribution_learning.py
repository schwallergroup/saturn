import os
import logging
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from rdkit import Chem

from models.generator import Generator
from distribution_learning.dataclass import DistributionLearningConfiguration
from distribution_learning.dataset.smiles_dataset import SMILESDataset

from utils.utils import setup_logging
from utils.chemistry_utils import canonicalize_smiles_batch


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
        logging_path: str,
        model_checkpoints_dir: str,
        configuration: DistributionLearningConfiguration
    ):
        # Training parameters
        self.configuration = configuration
        self.seed = configuration.seed
        # TODO: Adaptive learning rate
        self.learning_rate = configuration.learning_rate
        self.training_steps = configuration.training_steps
        self.batch_size = configuration.batch_size
        self.transfer_learning = configuration.transfer_learning
        self.train_with_randomization = configuration.train_with_randomization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the Agent
        self.agent = self._initialize_agent(configuration)
        self.optimizer = torch.optim.AdamW(self.agent.network.parameters(), lr=self.learning_rate)

        # Set up logging
        self.model_checkpoints_dir = model_checkpoints_dir
        os.makedirs(self.model_checkpoints_dir, exist_ok=True)
        setup_logging(logging_path)
  
    def run(self):
        start_time = time.perf_counter()
        for epoch in range(1, self.training_steps + 1, 1):
            self.agent.network.train()

            # --- Train Loop ---
            train_dataloader = self.get_train_dataloader()
            losses = np.array([])
            for _, batch in tqdm(enumerate(train_dataloader)):

                # 1. Compute the Negative Log-Likelihood (NLL) from Teacher Forcing
                batch = batch.to(self.device)
                loss = self.agent.likelihood(batch)
                losses = np.concatenate([losses, loss.detach().cpu().numpy()])

                # 2. Backpropagate
                self.backpropagate(loss)

            # --- Compute Base Metrics ---
            self.agent.network.eval()
            sampled = np.array([])
            
            # 1. Sample 10,000 SMILES
            while len(sampled) < 1e4:
                sampled_batch, _ = self.agent.sample_smiles(num=self.batch_size, batch_size=self.batch_size)
                sampled = np.concatenate([sampled, sampled_batch])

            # 2. Compute Validity
            valid = [smiles for smiles in sampled if Chem.MolFromSmiles(smiles) is not None]
            validity = len(valid) / len(sampled) * 100

            # 3. Compute Uniqueness
            valid = canonicalize_smiles_batch(valid)
            unique = len(set(valid)) / len(sampled) * 100

            # --- Log Results ---
            logging.info(f"Epoch {epoch} | NLL: {round(np.mean(losses), 3)} | Validity (10k): {round(validity, 2)}% | Uniqueness (10k): {round(unique, 2)}%")
            # Save current Agent
            self.agent.save(os.path.join(self.model_checkpoints_dir, f"{self.agent.model_architecture}_{epoch}.prior"))

        end_time = time.perf_counter()
        logging.info(f"Total wall time: {end_time - start_time} seconds.")

    def backpropagate(
        self, 
        loss: torch.Tensor
    ) -> None:
        """Policy update via backpropagation."""
        loss = loss.mean()
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

    def _initialize_agent(self, configuration: DistributionLearningConfiguration) -> Generator:
        # Initialize model
        if self.transfer_learning:
            # TODO: Construct the Transfer Learning DataLoader
            # Load the pre-trained Agent
            agent = Generator.load_from_file(configuration.agent, self.device)
        else:
            # Otherwise, train the Agent from scratch
            train_dataloader = self.get_train_dataloader()
            agent = Generator(
                model_architecture=configuration.model_architecture,
                vocabulary=train_dataloader.dataset.vocabulary,
                tokenizer=train_dataloader.dataset.tokenizer,
                network_params=None,
                device=self.device
            )
            vocabulary = train_dataloader.dataset.vocabulary
            # GEAM benchmarking: ZINC 250k randomization requires extra tokens
            # extra_zinc_tokens = ["[P@H]", "[S@+]"]
            # vocabulary.update(extra_zinc_tokens)
            agent = Generator(
                model_architecture=configuration.model_architecture,
                vocabulary=vocabulary,
                tokenizer=train_dataloader.dataset.tokenizer,
                network_params=None,
                device=self.device
            )
        return agent

    def get_train_dataloader(self):
        """
        Initialize the DataLoader for the training dataset.
        """
        train_dataset = SMILESDataset(
            agent=self.configuration.agent,
            dataset_path=self.configuration.training_dataset_path,
            batch_size=self.configuration.batch_size,
            transfer_learning=self.configuration.transfer_learning,
            randomize=self.configuration.train_with_randomization
        )
        train_dataloader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.configuration.batch_size, 
                shuffle=True,
                collate_fn=train_dataset.collate_fn
            )
        return train_dataloader
