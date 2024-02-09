"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent
Implements Diversity Filter as described in the paper: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0
"""

from typing import List
import numpy as np
from utils import chemistry_utils

class DiversityFilter:
    def __init__(
            self, 
            diversity_filter_configuration
        ):
        self.diversity_filter_configuration = diversity_filter_configuration
        # track the number of times a given Bemis-Murcko scaffold has been generated
        self.bucket_history = dict()
        self.bucket_size = diversity_filter_configuration.bucket_size
        self.min_similarity = diversity_filter_configuration.min_similarity

    def update(
            self,
            smiles: np.array
        ) -> None:
        """
        Update the bucket history based on the sampled (or hallucinated) batch of SMILES.
        """
        # get the Bemis-Murcko scaffold for each SMILES
        scaffolds = [chemistry_utils.get_bemis_murcko_scaffold(smiles) for smiles in smiles]
        for scaf in scaffolds:
            if scaf in self.bucket_history:
                self.bucket_history[scaf] += 1
            else:
                self.bucket_history[scaf] = 1

    def penalize_reward(
            self,
            smiles: np.array,
            rewards: np.array
        ) -> List[float]:
        """
        Penalize sampled (or hallucinated) SMILES based on the bucket history.
        """
        # if a given scaffold has been generated more than the bucket size, truncate the reward to 0.0
        scaffolds = [chemistry_utils.get_bemis_murcko_scaffold(smiles) for smiles in smiles]
        penalized_rewards = []
        for idx, scaf in enumerate(scaffolds):
            if scaf in self.bucket_history and self.bucket_history[scaf] > self.bucket_size:
                penalized_rewards.append(0.0)
            else:
                penalized_rewards.append(rewards[idx])
        
        return penalized_rewards
