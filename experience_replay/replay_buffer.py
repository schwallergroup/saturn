"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent.
"""
from typing import Tuple, List
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from utils import chemistry_utils
from utils.utils import to_tensor

from experience_replay.dataclass import ExperienceReplayParameters



class ReplayBuffer:
    """
    Replay buffer class which stores the top N highest reward molecules generated so far. Specific information stored includes:
        1. Canonicalized SMILES string
        2. Reward
        3. Likelihood under the Prior
        4. Likelihood under the Agent

    The SMILES stored in the replay buffer are manipulated and used by Augmented Memory and Hallucinated Memory to enhance sample efficiency.
    """
    def __init__(
        self, 
        parameters: ExperienceReplayParameters, 
        # TODO: keep this for incepting purposes?
        scoring_function=None
        ):
        self.parameters = parameters
        self.memory_size = parameters.memory_size
        self.sample_size = parameters.sample_size
        # stores the top N highest reward molecules generated so far
        self.memory = pd.DataFrame(
            columns=[
                "smiles", 
                "reward", 
                "prior_likelihood",
                "agent_likelihood"
                ]
            )
        
        # TODO: allow incepting
        # self.smiles = parameters.smiles

    def add(
        self, 
        smiles: np.array, 
        rewards: np.array, 
        prior_likelihoods: np.array, 
        agent_likelihoods: np.array
        ) -> None:
        # NOTE: likelihood should be already negative
        df = pd.DataFrame({
            "smiles": smiles, 
            "reward": rewards, 
            "prior_likelihood": prior_likelihoods.detach().cpu().numpy(),
            "agent_likelihood": agent_likelihoods.detach().cpu().numpy()}
            )
        self.memory = pd.concat([self.memory, df])
        # keep only the top N (by reward)
        self.purge_memory()

    def sample_memory(self) -> Tuple[np.ndarray[str], np.ndarray[float], torch.Tensor]:
        sample_size = min(len(self.memory), self.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            rewards = sampled["reward"].values
            prior_likelihoods = sampled["prior_likelihood"].values
            return np.array(smiles), np.array(rewards), to_tensor(np.array(prior_likelihoods))
        else:
            return [], [], torch.tensor([])

    def augmented_memory_replay(self, prior) -> Tuple[List[str], np.array, np.array]:
        """
        Augmented Memory's key operation for sample efficiency:
        Randomizes all SMILES in the memory and returns the randomized SMILES, reward, and prior likelihood.
        """
        if len(self.memory) != 0:
            smiles = self.memory["smiles"].values
            # randomize the smiles
            randomized_smiles = chemistry_utils.randomize_smiles_batch(smiles, prior)
            rewards = self.memory["reward"].values
            prior_likelihoods = -prior.likelihood_smiles(randomized_smiles).cpu()
            return randomized_smiles, rewards, to_tensor(np.array(prior_likelihoods))
        else:
            return [], [], torch.tensor([])
        
    def purge_memory(self):
        """
        Removes duplicate SMILES in the memory and keeps the top N by reward.
        Keep only non-zero rewards SMILES.
        """
        # NOTE: want to keep randomized versions?
        unique_df = self.memory.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values("reward", ascending=False)
        self.memory = sorted_df.head(self.memory_size)
        self.memory = self.memory.loc[self.memory["reward"] != 0.0]

    def selective_memory_purge(
            self, 
            smiles: np.array, 
            reward: np.array
        ) -> None:
        """
        Augmented Memory's key operation to prevent mode collapse and promote diversity:
        Purges the memory of SMILES that have penalized rewards (0.0) *before* executing Augmented Memory updates.
        Intuitively, this operation prevents penalized SMILES from directing the Agent's chemical space navigation.

        # NOTE: Consider a MPO objective task using a product aggregator. If one of the OracleComponent's reward is 0, 
        #       then the aggregated reward may be 0. But other OracleComponents may have a non-zero reward. We do not
        #       want to purge the memory of these scaffolds. This is already handled because 0 reward SMILES are not
        #       added to the memory in the first place. Selective Memory Purge *only* removes scaffolds that are 
        #       penalized by the Diversity Filter.
        """
        zero_reward_indices = np.where(reward == 0.)[0]
        if len(zero_reward_indices) > 0:
            smiles_to_purge = smiles[zero_reward_indices]
            scaffolds_to_purge = [chemistry_utils.get_bemis_murcko_scaffold(smiles) for smiles in smiles_to_purge]
            purged_memory = deepcopy(self.memory)
            purged_memory["scaffolds"] = purged_memory["smiles"].apply(chemistry_utils.get_bemis_murcko_scaffold)
            purged_memory = purged_memory.loc[~purged_memory["scaffolds"].isin(scaffolds_to_purge)]
            purged_memory.drop("scaffolds", axis=1, inplace=True)
            self.memory = purged_memory
        else:
            # if no scaffolds are penalized, do nothing
            return
