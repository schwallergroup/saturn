from typing import List
import os
import pandas as pd
import numpy as np
from oracles.oracle_component import OracleComponent


class Oracle:
    def __init__(
            self, 
            oracle_configuration
        ):
        self.oracle_configuration = oracle_configuration
        # construct the oracle function which can be composed of >1 individual oracles (multi-parameter optimization)
        self.oracle = self.construct_oracle()
        # cache dictionary to store the results of previous oracle calls
        self.cache = dict()
        self.calls = 0
        self.budget = oracle_configuration.budget
        # NOTE: assume no repeated oracle calls are allowed


    def __call__(self, smiles_batch: np.array) -> np.array:
        """
        Args:
            smiles_batch: np.array of strings of smiles
        Returns:a
            np.array of rewards (float) of the same length as smiles_batch
        """
        pass
        # increment the number of calls by the unique (canonicalized) SMILES
        # TODO: have option to flag which component of the oracle is "expensive" - 
        #       in this scenario, check the other cheap components are satisfied before calling the expensive component 
        #       (e.g., MW < 500 and QED > 0.4 before docking)
    
        # TODO: construct oracle should return a list of OracleComponent objects. __call__ should iterate through each component and call it
        #       each return value should already be subjected to a transformation function in the OracleComponent class
        #       the __call__ method should just aggregate the rewards either by weighted sum or weighted mean 
        # NOTE: DO NOT forget to assign the OracleComponent weight 
    
        
        
    def construct_oracle(self) -> List[OracleComponent]:
        """
        Construct the oracle function.
        """
        # TODO: aggregate the individual oracles into a single oracle function
        raise NotImplementedError
    

    def oracle_budget_exceeded(self) -> bool:
        """
        Check if the oracle budget has been exceeded.
        """
        return self.calls >= self.budget
