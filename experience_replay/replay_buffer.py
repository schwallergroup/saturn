"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from copy import deepcopy
from utils import chemistry_utils

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
        # TODO: keep this for Inception purposes?
        scoring_function=None
        ):
        print(type(parameters))
        exit()
        self.parameters = parameters
        # stores the top N highest reward molecules generated so far
        self.memory = pd.DataFrame(
            columns=[
                "smiles", 
                "reward", 
                "prior_likelihood",
                "agent_likelihood"
                ]
            )

    def add(
        self, 
        smiles: np.array, 
        reward: np.array, 
        prior_likelihood: np.array, 
        agent_likelihood: np.array
        ) -> None:
        # NOTE: likelihood should be already negative
        df = pd.DataFrame({
            "smiles": smiles, 
            "reward": reward, 
            "prior_likelihood": prior_likelihood.detach().cpu().numpy(),
            "agent_likelihood": agent_likelihood.detach().cpu().numpy()}
            )
        self.memory = pd.concat([self.memory, df])
        # keep only the top N (by reward)
        self.purge_memory()

    def sample_memory(self) -> Tuple[np.array, np.array, np.array]:
        sample_size = min(len(self.memory), self.configuration.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            reward = sampled["reward"].values
            prior_likelihood = sampled["prior_likelihood"].values
            return smiles, reward, prior_likelihood
        else:
            return [], [], []

    def augmented_memory_replay(self, prior) -> Tuple[List[str], np.array, np.array]:
        """
        Augmented Memory's key operation for sample efficiency:
        Randomizes all SMILES in the memory and returns the randomized SMILES, reward, and prior likelihood.
        """
        if len(self.memory) != 0:
            smiles = self.memory["smiles"].values
            # randomize the smiles
            randomized_smiles = chemistry_utils.get_randomized_smiles(smiles, prior)
            reward = self.memory["reward"].values
            prior_likelihood = -prior.likelihood_smiles(randomized_smiles).cpu()
            return randomized_smiles, reward, prior_likelihood
        else:
            return [], [], []
        
    def purge_memory(self):
        """
        Removes duplicate SMILES in the memory and keeps the top N by reward.
        """
        # NOTE: want to keep randomized versions?
        unique_df = self.memory.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values("reward", ascending=False)
        self.memory = sorted_df.head(self.configuration.memory_size)
        self.memory = self.memory.loc[self.memory["reward"] != 0.0]

    def selective_memory_purge(
            self, 
            smiles: np.array, 
            reward: np.array
        ) -> None:
        """
        Augmented Memory's key operation to prevent mode collapse and promite diversity:
        Purges the memory of SMILES that have penalized rewards (0.0) *before* executing Augmented Memory updates.
        Intuitively, this operation prevents penalized SMILES from directing the chemical space exploration.
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
            return

    def mode_collapse_guard(self):
        """
        In *pure* exploitation scenarios (*not recommended*) where Selective Memory Purge is not used, the following heuristic
        pre-emptively guards against rare cases of mode collapse at sub-optimal rewards.
        """
        sliced_memory = self.memory.head(int(self.configuration.memory_size*0.5))
        if (sliced_memory["reward"].nunique() == 1) and (int(sliced_memory["reward"].iloc[0]) != 1):
            print("---- Pre-emptively guarding against mode collapse: purging buffer -----")
            self.memory = pd.DataFrame(columns=["smiles", "reward", "prior_likelihood", "agent_likelihood"])
