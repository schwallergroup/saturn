"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent
"""
import numpy as np
from utils import chemistry_utils
from diversity_filter.dataclass import DiversityFilterParameters

class DiversityFilter:
    """
    Implements Diversity Filter as described in the paper: 
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0
    """
    def __init__(
        self, 
        parameters: DiversityFilterParameters
    ):
        self.parameters = parameters
        # Track the number of times a given Bemis-Murcko scaffold has been generated
        self.bucket_history = dict()
        self.bucket_size = parameters.bucket_size

    def update(
        self,
        smiles: np.ndarray[str]
    ) -> None:
        """
        Update the bucket history based on the sampled (or hallucinated) batch of SMILES.
        """
        # Get the Bemis-Murcko scaffold for each SMILES
        scaffolds = [chemistry_utils.get_bemis_murcko_scaffold(smiles) for smiles in smiles]
        for scaf in scaffolds:
            if scaf in self.bucket_history:
                self.bucket_history[scaf] += 1
            else:
                self.bucket_history[scaf] = 1

    def penalize_reward(
        self,
        smiles: np.ndarray[str],
        rewards: np.ndarray[float]
    ) -> np.ndarray[float]:
        """
        Penalize sampled (or hallucinated) SMILES based on the bucket history.
        """
        # If a given scaffold has been generated more than the bucket size, truncate the reward to 0.0
        if len(smiles) > 0:
            scaffolds = [chemistry_utils.get_bemis_murcko_scaffold(s) for s in smiles]
            penalized_rewards = []
            for idx, scaf in enumerate(scaffolds):
                if scaf in self.bucket_history and self.bucket_history[scaf] > self.bucket_size:
                    penalized_rewards.append(0.0)
                else:
                    penalized_rewards.append(rewards[idx])
            
            return np.array(penalized_rewards)
        else:
            return np.array([])
