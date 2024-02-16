"""
Adapted from https://github.com/MolecularAI/Reinvent.
"""
from typing import Tuple
import torch
import numpy as np

import utils.chemistry_utils as chemistry_utils
from utils.utils import to_tensor

from distribution_learning.dataclass import DistributionLearningConfiguration
from models.model import Model


class DistributionLearningTrainer:
    """
    Distribution Learning by Maximum Likelihood Estimation (MLE) and using Teacher Forcing.
    Training objective is to maximize the likelihood of reproducing the SMILES dataset.

    Used to pre-train the Prior or for fine-tuning (transfer learning) the Agent.
    """
    def __init__(
        self, 
        configuration: DistributionLearningConfiguration
    ):
        # TODO: "create" empty model with vocabulary
        # TODO: this "empty" model parameter should also allow for an already pre-trained Agent to be loaded so transfer learning can be applied
        self.seed = configuration.seed
        self.learning_rate = configuration.learning_rate
        self.training_steps = configuration.training_steps
        self.batch_size = configuration.batch_size
        self.training_dataset = configuration.training_dataset
        self.validation_dataset = configuration.validation_dataset
        self.train_with_randomization = configuration.train_with_randomization

        # only the Agent is updated
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=self.learning_rate)
  
    def train(self):
        for epoch in self.training_steps:
            # TODO: train
            if self.train_with_randomization:
                # TODO: randomize training batch
                pass
            pass


        self.write_out_results()

    def compute_loss(
        self, 
        smiles: np.ndarray[str],
        rewards: np.ndarray[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the RL agent.
        Based on REINVENT's original loss function: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x
        """
        if len(smiles) != 0:
            prior_likelihoods = -self.prior.likelihood_smiles(smiles)
            agent_likelihoods = -self.agent.likelihood_smiles(smiles)
            augmented_likelihoods = prior_likelihoods + self.sigma * to_tensor(rewards)
            loss = torch.pow((augmented_likelihoods - agent_likelihoods), 2)
            return loss, prior_likelihoods, agent_likelihoods
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

    def backpropagate(self, loss: torch.Tensor) -> None:
        """Agent update via backpropagation."""
        loss = loss.mean()
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

    def write_out_results(self):
        """
        Writes out the following results:
            1. Oracle History
            2. Beam Enumeration History
            3. Hallucination History
            4. Number of Oracle repeats
        """
        self.oracle.write_out_oracle_history()
        self.oracle.write_out_repeat_history()

        if self.execute_beam_enumeration:
            self.beam_enumeration.end_actions(self.oracle.calls)

        if self.execute_hallucinated_memory:
            self.hallucinator.write_out_hallucination_history()
