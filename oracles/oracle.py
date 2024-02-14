from typing import List, Tuple
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from utils.chemistry_utils import canonicalize_smiles_batch

from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters, OracleConfiguration
from oracles.reward_aggregator.reward_aggregator import RewardAggregator
from diversity_filter.diversity_filter import DiversityFilter

from oracles.utils import construct_oracle_component


class Oracle:
    def __init__(
        self, 
        oracle_configuration: OracleConfiguration
    ):
        self.oracle_configuration = oracle_configuration
        
        # construct the oracle function which can be composed of >1 individual oracles (multi-parameter optimization)
        self.oracle = self.construct_oracle(oracle_configuration.components)
        # preliminary oracles can be executed as a first pass to filter out poor candidates
        self.preliminary_oracles = [oracle for oracle in self.oracle if oracle.preliminary_check]
        self.oracle_weights = [oracle.weight for oracle in self.oracle]
        self.aggregator = RewardAggregator(oracle_configuration.aggregator)

        # track oracle budget
        self.budget = oracle_configuration.budget
        self.allow_oracle_repeats = oracle_configuration.allow_oracle_repeats
        self.calls = 0

        # cache dictionary to store the results of previous oracle calls
        self.cache = dict()

        # oracle history to assess sample efficiency via Generative Yield and Oracle Burden metrics
        self.oracle_history = pd.DataFrame({
            "oracle_calls": [],
            "smiles": [],
            "reward": [],
            "penalized_reward": []
        })
        # add each oracle component's raw value and reward to the oracle history DataFrame
        for oracle in self.oracle:
            self.oracle_history[f"{oracle.name}_raw_values"] = []
            self.oracle_history[f"{oracle.name}_reward"] = []

        # track how many times the same SMILES is sampled
        self.num_repeated_smiles = []

    def __call__(
        self, 
        smiles: np.ndarray[str],
        diversity_filter: DiversityFilter
    ) -> Tuple[np.ndarray[str], np.ndarray[float]]:
        """
        The Oracle is called at every generation epoch and performs the following:
            1. Calls each oracle component in the Oracle 
            2. Aggregates the oracle feedback into a single scalar reward
            3. Penalizes the reward based on the Diversity Filter
            4. Updates the Diversity Filter
            5. Updates the Oracle History which tracks oracle calls, rewards, and penalized rewards
            6. Updates the Oracle Cache to store the results of previous oracle calls

        Returns the SMILES and the penalized rewards. 
        SMILES need to be returned because they are filtered here (based on RDKit validity) and preliminary check.
        Likelihoods of the SMILES are calculated in the Reinforcement Learning module.
        """
        # 1. Only keep the valid SMILES (RDKit parsable into Mols)
        smiles = np.array([s for s in smiles if Chem.MolFromSmiles(s) is not None])
        
        # 2. Rewards can be obtained directly for SMILES in the Oracle Cache 
        repeat_smiles, cached_rewards, new_smiles = self.rewards_from_oracle_cache(smiles)

        # 3. Get the Mols for the new SMILES
        new_mols = np.vectorize(Chem.MolFromSmiles)(new_smiles)

        # 4. Execute preliminary check (if applicable) which removes Mols that do not satisfy the (relatively) cheaper oracle components
        #    e.g., molecular weight is too high (> 500 Da), so discard without wasting computational resources on a docking oracle
        new_smiles, new_mols = self.execute_preliminary_check(new_smiles, new_mols)

        # 5. Call each oracle component and aggregate the rewards
        #    Initialize a DataFrame to store the raw values and rewards of each oracle component for tracking purposes
        oracle_components_df = pd.DataFrame()
        rewards = np.empty((len(self.oracle), len(new_mols)))
        for idx, oracle in enumerate(self.oracle):
            raw_property_values, component_rewards = oracle.calculate_reward(new_mols)
            oracle_components_df[f"{oracle.name}_raw_values"] = raw_property_values
            oracle_components_df[f"{oracle.name}_reward"] = component_rewards
            rewards[idx] = component_rewards

        # 6. Aggregate the rewards
        aggregated_rewards = self.aggregator(rewards, self.oracle_weights)

        # 7. Penalize the rewards based on the Diversity Filter
        penalized_rewards = diversity_filter.penalize_reward(new_smiles, aggregated_rewards)

        # 8. Update the Diversity Filter
        diversity_filter.update(new_smiles)

        # 9. Update the Oracle History
        self.update_oracle_history(
            smiles=smiles,
            rewards=aggregated_rewards,
            penalized_rewards=penalized_rewards,
            oracle_components_df=oracle_components_df
        )

        # 10. Update the Oracle Cache - important to cache the penalized rewards
        self.update_oracle_cache(new_smiles, penalized_rewards)
                           
        return np.concatenate([repeat_smiles, new_smiles]), np.concatenate([cached_rewards, penalized_rewards])
        
        
    def construct_oracle(self, oracle_components: List[OracleComponentParameters]) -> List[OracleComponent]:
        """
        Construct the oracle function.
        """
        oracle = []
        for component in oracle_components:
            # construct the OracleComponent
            oracle_component = construct_oracle_component(OracleComponentParameters(**component))
            oracle.append(oracle_component)

        return oracle
    
    def rewards_from_oracle_cache(self, smiles: np.ndarray[str]) -> Tuple[np.ndarray[str], np.ndarray[float], np.ndarray[str]]:
        """
        Checks if there are any Cached rewards in a sampled batch of SMILES. If Oracle repeats are permitted, directly return.
        """
        if not self.allow_oracle_repeats:
            # canonicalize the SMILES before checking Cache
            canonical_smiles = canonicalize_smiles_batch(smiles)
            repeat_indices = []
            cached_rewards = []
            for idx, s in enumerate(canonical_smiles):
                if s in self.cache:
                    repeat_indices.append(idx)
                    cached_rewards.append(self.cache[s])

            # track number of repeated SMILES
            self.num_repeated_smiles.append(len(repeat_indices))

            if len(repeat_indices) != 0:
                return smiles[repeat_indices], np.array(cached_rewards), np.delete(smiles, repeat_indices)
            else:
                return np.array([]), np.array([]), smiles
        
        else:
            return np.array([]), np.array([]), smiles
    
    def update_oracle_cache(self, smiles: np.ndarray[str], rewards: np.ndarray[float]) -> None:
        """
        Updates the Oracle Cache to store the results of previous oracle calls.
        """
        # canonicalize the SMILES before adding to Cache
        canonical_smiles = canonicalize_smiles_batch(smiles)
        for s, r in zip(canonical_smiles, rewards):
            self.cache[s] = r
    
    def execute_preliminary_check(self, smiles: np.ndarray[str], mols: np.ndarray[Mol]) -> np.ndarray[Mol]:
        """
        Executes a preliminary check (if applicable). Each oracle component has a preliminary_check flag that can be set to True.
        Components set to True will be executed first to check that the molecule satisfies that component based on a reward threshold.
        If the molecule does not satisfy the threshold, it is removed from the sampled batch.
        """
        # FIXME: set a threshold for each component. If not using Step transformation, rewards are not necessarily 0
        THRESHOLD = 0.05

        if len(self.preliminary_oracles) != 0:
            filtered_indices = []
            for mol in mols:
                for idx, oracle in enumerate(self.preliminary_oracles):
                    _, rewards = oracle.calculate_reward(mol)
                    if (rewards > THRESHOLD) and (idx == len(self.preliminary_oracles) - 1):
                        filtered_indices.append(idx)

            return smiles[filtered_indices], mols[filtered_indices]
            
        else:
            return smiles, mols

    def update_oracle_history(
        self, 
        smiles: np.ndarray[str],
        rewards: np.ndarray[float],
        penalized_rewards: np.ndarray[float],
        oracle_components_df: pd.DataFrame
    ) -> None:
        """
        This method performs the following on every generation epoch:
        1. Increments the number of oracle calls so far
        2. Updates the Oracle History that tracks the generative sampling as a function of oracle calls
        """
        self.calls += len(smiles)
        # track generated SMILES + reward as a function of oracle calls
        df = pd.DataFrame({
                "oracle_calls": np.full_like(smiles, self.calls),
                "smiles": smiles,
                "reward": rewards, 
                "penalized_reward": penalized_rewards, 
            })
        df = pd.concat([df, oracle_components_df], axis=1)  

        self.oracle_history = pd.concat([self.oracle_history, df])

    def budget_exceeded(self) -> bool:
        """
        Check if the oracle budget has been exceeded.
        """
        return self.calls >= self.budget

    def write_out_oracle_history(self):
        """
        Write out the oracle history as a CSV.
        """
        self.oracle_history.to_csv("oracle_history.csv")

    def write_out_repeat_history(self):
        """
        Write out the repeated SMILES history as a CSV.
        """
        pd.DataFrame(self.repeated_smiles).to_csv("repeated_smiles_history.csv")
