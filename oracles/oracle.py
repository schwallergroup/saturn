from typing import List, Tuple
import os
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from utils.chemistry_utils import canonicalize_smiles_batch, get_bemis_murcko_scaffold

from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters, OracleConfiguration
from oracles.reward_aggregator.reward_aggregator import RewardAggregator
from diversity_filter.diversity_filter import DiversityFilter

from oracles.utils import construct_oracle_component


class Oracle:
    """
    The Oracle function to optimize for in the generative experiment.
    Can be composed of multiple OracleComponents, each of which handles a specific property.
    Aggregating the rewards from each OracleComponent returns a scalar reward (return in RL terminology) for Agent update.
    """
    def __init__(
        self, 
        oracle_configuration: OracleConfiguration
    ):
        self.oracle_configuration = oracle_configuration
        
        # Construct the oracle function which can be composed of >1 individual oracles (multi-parameter optimization)
        self.oracle = self.construct_oracle(oracle_configuration.components)
        # Preliminary oracles can be executed as a first pass to filter out poor candidates
        self.preliminary_oracles = [oracle for oracle in self.oracle if oracle.preliminary_check]
        self.oracle_weights = [oracle.weight for oracle in self.oracle]
        self.aggregator = RewardAggregator(oracle_configuration.aggregator)

        # Track oracle budget
        self.budget = oracle_configuration.budget
        self.allow_oracle_repeats = oracle_configuration.allow_oracle_repeats
        self.calls = 0

        # Cache dictionary to store the results of previous oracle calls
        self.cache = dict()

        # Oracle history to assess sample efficiency via Generative Yield and Oracle Burden metrics
        self.oracle_history = pd.DataFrame({
            "oracle_calls": [],
            "scaffold": [],
            "smiles": [],
            "reward": [],
            "penalized_reward": []
        })
        # Add oracle components' raw value and reward to the oracle history DataFrame
        for oracle in self.oracle:
            if oracle.name == "geam":
                self.oracle_history["raw_vina"] = []
                self.oracle_history["qed"] = []
                self.oracle_history["raw_sa"] = []
                self.oracle_history["aggregated_reward"] = []
            else:
                self.oracle_history[f"{oracle.name}_raw_values"] = []
                self.oracle_history[f"{oracle.name}_reward"] = []

        # Track how many times the same SMILES is sampled
        self.repeated_sampled_smiles = {}
        self.repeated_hallucinated_smiles = {}

    def __call__(
        self, 
        smiles: np.ndarray[str],
        diversity_filter: DiversityFilter,
        is_hallucinated_batch: bool = False
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
        repeat_smiles, cached_rewards, new_smiles = self.rewards_from_oracle_cache(smiles, is_hallucinated_batch)

        # In case all SMILES are repeats
        if len(new_smiles) > 0:
            # 3. Get the Mols for the new SMILES
            new_mols = np.vectorize(Chem.MolFromSmiles)(new_smiles)

            # 4. Execute preliminary check (if applicable) which removes Mols that do not satisfy the (relatively) cheaper oracle components
            #    e.g., molecular weight is too high (> 500 Da), so discard without wasting computational resources on a docking oracle
            new_smiles, new_mols = self.execute_preliminary_check(new_smiles, new_mols)

            # In case no SMILES pass the preliminary check
            if len(new_smiles) > 0:
                # 5. Call each oracle component and aggregate the rewards
                #    Initialize a DataFrame to store the raw values and rewards of each oracle component for tracking purposes
                oracle_components_df = pd.DataFrame()
                rewards = np.empty((len(self.oracle), len(new_mols)))
                for idx, oracle in enumerate(self.oracle):
                    if oracle.name == "geam":
                        raw_vina, qed_rewards, raw_sa, aggregated_rewards = oracle(new_mols)
                        oracle_components_df["raw_vina"] = raw_vina
                        oracle_components_df["qed"] = qed_rewards
                        oracle_components_df["raw_sa"] = raw_sa
                        oracle_components_df["aggregated_reward"] = aggregated_rewards
                    else:
                        raw_property_values, component_rewards = oracle.calculate_reward(new_mols, self.calls)
                        oracle_components_df[f"{oracle.name}_raw_values"] = raw_property_values
                        oracle_components_df[f"{oracle.name}_reward"] = component_rewards
                        rewards[idx] = component_rewards
                
                # 6. Aggregate the rewards
                if oracle.name == "geam":
                    rewards = np.array([aggregated_rewards])
                else:
                    aggregated_rewards = self.aggregator(rewards, self.oracle_weights)
            else:
                aggregated_rewards = np.array([0.0])

        # FIXME: probably another way to do this
        else:
            aggregated_rewards = np.array([0.0])

        # 7. At this point, rewards have been obtained for the new SMILES
        #    Increment the number of oracle calls
        self.calls += len(new_smiles)

        # 8. Concatenate the repeated and new SMILES and their corresponding rewards
        #    The Diversity Filter operations are performed on the concatenated set
        all_smiles = np.concatenate([repeat_smiles, new_smiles])
        all_rewards = np.concatenate([cached_rewards, aggregated_rewards])

        # 9. Penalize the rewards based on the Diversity Filter
        penalized_new_rewards = diversity_filter.penalize_reward(new_smiles, aggregated_rewards)
        penalized_all_rewards = diversity_filter.penalize_reward(all_smiles, all_rewards)

        # 10. Update the Oracle History
        if len(new_smiles) > 0:
            # Only update with the new SMILES
            if not self.allow_oracle_repeats:
                oracle_history_smiles = new_smiles
                oracle_history_rewards = aggregated_rewards
                oracle_history_penalized_rewards = penalized_new_rewards
            # Update with all SMILES
            elif self.allow_oracle_repeats:
                oracle_history_smiles = all_smiles
                oracle_history_rewards = all_rewards
                oracle_history_penalized_rewards = penalized_all_rewards

            self.update_oracle_history(
                smiles=oracle_history_smiles,
                scaffolds=np.vectorize(get_bemis_murcko_scaffold)(oracle_history_smiles),
                rewards=oracle_history_rewards,
                penalized_rewards=oracle_history_penalized_rewards,
                oracle_components_df=oracle_components_df
            )

        # 11. Update the Diversity Filter
        diversity_filter.update(all_smiles)
        
        # 12. Update the Oracle Cache - important to cache the penalized rewards
        self.update_oracle_cache(all_smiles, penalized_all_rewards)
                           
        return all_smiles, penalized_all_rewards
        
    def construct_oracle(self, oracle_components: List[OracleComponentParameters]) -> List[OracleComponent]:
        """
        Construct the oracle function which can be composed of multiple inidividual oracle components.
        """
        oracle = []
        for component in oracle_components:
            # Construct the OracleComponent
            oracle_component = construct_oracle_component(OracleComponentParameters(**component))
            oracle.append(oracle_component)

        return oracle
    
    def rewards_from_oracle_cache(
        self,
        smiles: np.ndarray[str],
        is_hallucinated_batch: bool
    ) -> Tuple[np.ndarray[str], np.ndarray[float], np.ndarray[str]]:
        """
        Checks if there are any Cached rewards in a sampled batch of SMILES. 
        Also updates trackers for repeated SMILES. If Oracle repeats are permitted, directly return.
        """
        if not self.allow_oracle_repeats:
            # Canonicalize the SMILES before checking Cache
            canonical_smiles = canonicalize_smiles_batch(smiles)
            repeat_indices = []
            cached_rewards = []
            for idx, s in enumerate(canonical_smiles):
                if s in self.cache:
                    repeat_indices.append(idx)
                    # Take the mean of the cached rewards in case of repeats
                    cached_rewards.append(np.mean(self.cache[s]))

            if len(repeat_indices) != 0:
            # Track the repeated SMILES and their rewards
                repeated_smiles = smiles[repeat_indices]
                for idx, s in enumerate(repeated_smiles):
                    if is_hallucinated_batch:
                        if s not in self.repeated_hallucinated_smiles:
                            self.repeated_hallucinated_smiles[s] = (1, cached_rewards[idx])
                        else:
                            self.repeated_hallucinated_smiles[s] = (self.repeated_hallucinated_smiles[s][0] + 1, cached_rewards[idx])
                    else:
                        if s not in self.repeated_sampled_smiles:
                            self.repeated_sampled_smiles[s] = (1, cached_rewards[idx])
                        else:
                            self.repeated_sampled_smiles[s] = (self.repeated_sampled_smiles[s][0] + 1, cached_rewards[idx])

                return repeated_smiles, np.array(cached_rewards), np.delete(smiles, repeat_indices)
            else:
                return np.array([]), np.array([]), smiles
        
        else:
            return np.array([]), np.array([]), smiles
    
    def update_oracle_cache(self, smiles: np.ndarray[str], rewards: np.ndarray[float]) -> None:
        """
        Updates the Oracle Cache to store the results of previous oracle calls.
        """
        # Canonicalize the SMILES before adding to Cache
        canonical_smiles = canonicalize_smiles_batch(smiles)
        for s, r in zip(canonical_smiles, rewards):
            # If the same SMILES is sampled, all rewards are tracked for two reasons:
            #   1. Potential stocasticity in the oracle feedback
            #   2. Penalized rewards (by the Diversity Filter) should be reflected so the Agent is steered away from these scaffolds
            if s not in self.cache:
                self.cache[s] = [r]
            elif r == 0.0:
                self.cache[s] = [r]
            else:
                self.cache[s].append(r)
    
    def execute_preliminary_check(self, smiles: np.ndarray[str], mols: np.ndarray[Mol]) -> Tuple[np.ndarray[str], np.ndarray[Mol]]:
        """
        Executes a preliminary check (if applicable). Each oracle component has a preliminary_check flag that can be set to True.
        Components set to True will be executed first to check that the molecule satisfies that component based on a reward threshold.
        If the molecule does not satisfy the threshold, it is removed from the sampled batch.
        """
        # FIXME: Set a threshold for each component. If not using Step transformation, rewards are not necessarily 0
        THRESHOLD = 0.05

        if len(self.preliminary_oracles) > 0:
            filtered_indices = []
            # TODO: Vectorize the batch of Mols
            for idx, mol in enumerate(mols):
                for oracle in self.preliminary_oracles:
                    _, reward = oracle.calculate_reward(np.array([mol]), self.calls)
                    if reward < THRESHOLD:
                        # If the reward is below the threshold for at least one oracle component, add the index to the filtered indices
                        filtered_indices.append(idx)
                        break

            return np.delete(smiles, filtered_indices), np.delete(mols, filtered_indices)
            
        else:
            return smiles, mols

    def update_oracle_history(
        self, 
        scaffolds: np.ndarray[str],
        smiles: np.ndarray[str],
        rewards: np.ndarray[float],
        penalized_rewards: np.ndarray[float],
        oracle_components_df: pd.DataFrame
    ) -> None:
        """
        This method performs the following on every generation epoch:
        1. Increments the number of oracle calls so far
        2. Updates the Oracle History that tracks the generative sampling as a function of oracle calls

        # NOTE: If self.allow_oracle_repeats = True, the Oracle History tracks every single SMILES generated and not just the unique set.
        #       This can be useful to interrogate the stochasticity of the oracle to inform downstream molecule prioritization.
        """
        # Track generated SMILES + reward as a function of oracle calls
        df = pd.DataFrame({
                "oracle_calls": np.full_like(smiles, self.calls),
                "scaffold": scaffolds,
                "smiles": smiles,
                "reward": rewards, 
                "penalized_reward": penalized_rewards 
            })
        df = pd.concat([df, oracle_components_df], axis=1)  

        self.oracle_history = pd.concat([self.oracle_history, df])

    def budget_exceeded(self) -> bool:
        """Check if the oracle budget has been exceeded."""
        return self.calls >= self.budget

    def write_out_oracle_history(self, path: str):
        """Write out the oracle history as a CSV."""
        self.oracle_history.to_csv(os.path.join(path, "oracle_history.csv"))

    def write_out_repeat_history(self, path: str):
        """Write out the repeated SMILES histories as JSON."""
        with open(os.path.join(path, "repeated_sampled_smiles_history.json"), "w") as f:
            json.dump(self.repeated_sampled_smiles, f, indent=2)
        with open(os.path.join(path, "repeated_hallucinated_smiles_history.json"), "w") as f:
            json.dump(self.repeated_hallucinated_smiles, f, indent=2)
