"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from copy import deepcopy
from rdkit import Chem
from utils import chemistry_utils
from utils.utils import to_tensor

from experience_replay.dataclass import ExperienceReplayParameters

# Oracle is called if seeding molecules into the Replay Buffer at the start of the generative experiment
from oracles.oracle import Oracle


class ReplayBuffer:
    """
    Replay buffer class which stores the top N highest reward molecules generated so far. Specific information stored includes:
        1. Canonicalized SMILES string
        2. Reward

    The SMILES stored in the replay buffer are manipulated and used by Augmented Memory and Hallucinated Memory to enhance sample efficiency.
    """
    def __init__(
        self, 
        parameters: ExperienceReplayParameters
        ):
        self.parameters = parameters
        self.memory_size = parameters.memory_size
        self.sample_size = parameters.sample_size
        # Stores the top N highest reward molecules generated so far
        self.memory = pd.DataFrame(
            columns=[
                "smiles", 
                "reward"
                ]
            )

    def add(
        self,
        smiles: np.ndarray[str], 
        rewards: np.ndarray[float]
        ) -> None:
        df = pd.DataFrame({
            "smiles": smiles, 
            "reward": rewards
            }
        )
        self.memory = pd.concat([self.memory, df])
        # Keep only the top N (by reward)
        self.purge_memory()

    def sample_memory(self) -> Tuple[np.ndarray[str], np.ndarray[float]]:
        sample_size = min(len(self.memory), self.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            rewards = sampled["reward"].values
            return np.array(smiles), np.array(rewards)
        else:
            return [], []

    def augmented_memory_replay(self, prior) -> Tuple[List[str], np.array]:
        """
        Augmented Memory's key operation for sample efficiency:
        Randomizes all SMILES in the memory and returns the randomized SMILES and their corresponding rewards.
        """
        if len(self.memory) != 0:
            smiles = self.memory["smiles"].values
            # Randomize the smiles
            randomized_smiles = chemistry_utils.randomize_smiles_batch(smiles, prior)
            rewards = self.memory["reward"].values
            return randomized_smiles, rewards
        else:
            return [], []
        
    def purge_memory(self):
        """
        Removes duplicate SMILES in the memory and keeps the top N by reward.
        Keep only non-zero rewards SMILES.
        """
        # NOTE: Want to keep randomized versions?
        unique_df = self.memory.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values("reward", ascending=False)
        self.memory = sorted_df.head(self.memory_size)
        self.memory = self.memory.loc[self.memory["reward"] != 0.0]

    def selective_memory_purge(
            self, 
            smiles: np.ndarray[str], 
            rewards: np.ndarray[float]
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
        zero_reward_indices = np.where(rewards == 0.)[0]
        if len(zero_reward_indices) > 0:
            smiles_to_purge = smiles[zero_reward_indices]
            scaffolds_to_purge = [chemistry_utils.get_bemis_murcko_scaffold(smiles) for smiles in smiles_to_purge]
            purged_memory = deepcopy(self.memory)
            purged_memory["scaffolds"] = purged_memory["smiles"].apply(chemistry_utils.get_bemis_murcko_scaffold)
            purged_memory = purged_memory.loc[~purged_memory["scaffolds"].isin(scaffolds_to_purge)]
            purged_memory.drop("scaffolds", axis=1, inplace=True)
            self.memory = purged_memory
        else:
            # If no scaffolds are penalized, do nothing
            return
        
    def prepopulate_buffer(self, oracle: Oracle) -> Oracle:
        """
        Seeds the replay buffer with a set of SMILES.
        Useful if there are known high-reward molecules to pre-populate the Replay Buffer with.

        Oracle is returned here because seeding updates the Oracle's history with the seeded SMILES.

        NOTE: With more SMILES to seed with, the generative experiment will become more like
              transfer learning rather than reinforcement learning (at the start). Continuing
              the run will more and more leverage reinforcement learning to find other diverse
              solutions. Therefore, while seeding will quick-start the Agent's learning, there
              are implications on the diversity of the solutions found.
        """
        if len(self.parameters.smiles) > 0:
            canonical_smiles = chemistry_utils.canonicalize_smiles_batch(self.parameters.smiles)
            mols = [Chem.MolFromSmiles(s) for s in canonical_smiles]
            mols = [mol for mol in mols if mol is not None]

            oracle_components_df = pd.DataFrame()
            rewards = np.empty((len(oracle), len(mols)))
            for idx, oracle in enumerate(oracle):
                raw_property_values, component_rewards = oracle.calculate_reward(mols, oracle_calls=0)
                oracle_components_df[f"{oracle.name}_raw_values"] = raw_property_values
                oracle_components_df[f"{oracle.name}_reward"] = component_rewards
                rewards[idx] = component_rewards

            aggregated_rewards = oracle.aggregator(rewards, oracle.oracle_weights)

            # Add the SMILES to the Replay Buffer
            self.add(
                smiles=self.parameters.smiles,  # add the original SMILES
                rewards=aggregated_rewards
            )

            oracle.update_oracle_history(
                smiles=self.parameters.smiles,
                rewards=aggregated_rewards,
                penalized_rewards=aggregated_rewards,
                oracle_components_df=oracle_components_df
            )
            # Update the Oracle Cache with the canonical SMILES
            oracle.update_oracle_cache(canonical_smiles, rewards)

        return oracle
