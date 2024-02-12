from typing import List
import pandas as pd
import numpy as np

from oracles.oracle_component import OracleComponent
from oracles.oracle_dataclass import OracleComponentParameters, OracleConfiguration
from diversity_filter.diversity_filter import DiversityFilter

from oracles.utils import construct_oracle_component


class Oracle:
    def __init__(
        self, 
        oracle_configuration: OracleConfiguration
    ):
        self.oracle_configuration = oracle_configuration
        # construct the oracle function which can be composed of >1 individual oracles (multi-parameter optimization)
        self.oracle = self.construct_oracle(oracle_configuration["components"])
        self.aggregator = oracle_configuration["aggregator"]

        # track oracle budget
        self.budget = oracle_configuration["budget"]
        self.calls = 0

        # cache dictionary to store the results of previous oracle calls
        self.cache = dict()
        self.calls = 0

        # oracle history to assess sample efficiency via Generative Yield and Oracle Burden metrics
        # TODO: need to also track reward of each component
        self.oracle_history = pd.DataFrame({
            "oracle_calls": [],
            "smiles": [],
            "reward": [],
            "penalized_reward": []
        })

        # NOTE: assume no repeated oracle calls are allowed


    def __call__(
        self, 
        smiles_batch: np.ndarray[str],
        diversity_filter: DiversityFilter) -> np.ndarray[float]:
        """
        Args:
            smiles_batch: np.array of strings of smiles
        Returns:a
            np.array of rewards (float) of the same length as smiles_batch
        """
        pass
        # TODO: have option to flag which component of the oracle is "expensive" - 
        #       in this scenario, check the other cheap components are satisfied before calling the expensive component 
        #       (e.g., MW < 500 and QED > 0.4 before docking)
    
        # TODO: construct oracle should return a list of OracleComponent objects. __call__ should iterate through each component and call it
        #       each return value should already be subjected to a transformation function in the OracleComponent class
        #       the __call__ method should just aggregate the rewards either by weighted sum or weighted mean
     
        # NOTE: DO NOT forget to assign the OracleComponent weight 
    
        # each OracleComponent has a calculate reward method --> run through all of them in the list and then aggregate
    
        # NOTE: check the preliminary_check flag!!!!
    
        # TODO: only increment the NEW SMILES - check this by canonicalization BUT make sure not to return the canonicalization for backpropagation
        #       because augmented SMILES can be useful for likelihood learning
    

        # TODO: apply diversity filter!!
        reward = np.ones((30, 10000))
        penalized_reward = diversity_filter.penalize_reward(smiles_batch, reward)

        # TODO: add raw rewards to this
        self.update_oracle_history(
            num_valid_smiles=len(reward),
            smiles=smiles_batch,
            reward=reward,
            penalized_reward=penalized_reward
        )

        # TODO: update the cache!!! 
        # ******
        # cache the penalized rewards!!!!!
        # ******
                                   
        
        
    def construct_oracle(self, oracle_components: List[OracleComponentParameters]) -> List[OracleComponent]:
        """
        Construct the oracle function.
        """
        oracle = []
        for component in oracle_components:
            # construct the OracleComponent
            oracle_component = construct_oracle_component(component)
            oracle.append(oracle_component)

        return oracle
    
    def update_oracle_history(
        self, 
        num_valid_smiles: int,
        smiles: np.ndarray[str],
        reward: np.ndarray[float],
        penalized_reward: np.ndarray[float]
    ) -> None:
        """
        This method performs 2 updates on every epoch:
        1. Increments the number of oracle calls so far
        2. Updates the Oracle History that tracks the generative sampling as a function of oracle calls
        """
        self.calls += num_valid_smiles
        # track generated SMILES + reward as a function of oracle calls
        df = pd.DataFrame({
                "oracle_calls": np.full_like(smiles, self.calls),
                "smiles": smiles,
                "reward": reward, 
                "penalized_reward": penalized_reward, 
            })
        
        self.oracle_history = pd.concat([self.oracle_history, df])

    def update_generative_history(
        self, 
        num_valid_smiles: int,
        smiles: np.ndarray[str],
        reward: np.ndarray[float]
    ) -> None:
        """
        This method performs 2 updates on every epoch:
        1. Increments the number of oracle calls so far
        2. Updates the Oracle History that tracks the generative sampling as a function of oracle calls
        """
        self.calls += num_valid_smiles
        # track generated SMILES + reward as a function of oracle calls
        df = pd.DataFrame({
                "oracle_calls": np.full_like(smiles, self.calls),
                "reward": reward, 
                "smiles": smiles
            })
        
        self.oracle_history = pd.concat([self.oracle_history, df])

    def oracle_budget_exceeded(self) -> bool:
        """
        Check if the oracle budget has been exceeded.
        """
        return self.calls >= self.budget

    def write_out_oracle_history(self):
        """
        Write out the oracle history as a CSV.
        """
        self.oracle_history.to_csv("oracle_history.csv")
