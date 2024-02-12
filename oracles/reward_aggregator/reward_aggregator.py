"""
Adapted from https://github.com/MolecularAI/reinvent-scoring.
Implements the following reward aggregators:
    1. Sum
    2. Product
Both can assign different weights to individual OracleComponents
"""

import numpy as np
import math

class RewardAggregator:

    def __init__(self, aggregator: str):
        self.aggregator = aggregator.lower()

    def __call__(
        self, 
        rewards: np.array[float], 
        weights: np.array[float]
    ) -> np.array[float]:
        """
        Takes as input the list of transformed rewards based on the OracleComponent and aggregates them into a single scalar.
        """
        if self.aggregator == "sum":
            return self.sum(rewards, weights)
        elif self.aggregator == "product":
            return self.product(rewards, weights)
        else:
            raise NotImplementedError(f"{self.aggregator} reward aggregator is not implemented.")

    def sum(
        self, 
        rewards: np.ndarray,  # (number of OracleComponents, number of SMILES)
        weights: np.ndarray  # (number of OracleComponents, 1)
    ) -> np.array[float]:
        """
        Weighted Sum aggregation.
        """
        total_sum = np.sum(rewards * weights, axis=0)
        total_weight = np.sum(weights)
        return total_sum / total_weight  # (number of SMILES,)
        
    def product(
        self, 
        rewards: np.ndarray,  # (number of OracleComponents, number of SMILES, 1)
        weights: np.ndarray  # (number of OracleComponents, 1)
    ) -> np.array[float]:
        """
        Weighted Product aggregation.
        """
        product = np.ones(rewards.shape[1], dtype=np.float32)  # (number of SMILES,)

        def weighted_power(values: np.array[float], weight: float) -> np.array[float]:
            """
            Helper function to calculate the weighted power of individual rewards.
            """
            return np.power(values, weight)
        
        total_weight = np.sum(weights)

        for r, w in zip(rewards, weights):
            product *= weighted_power(r, w / total_weight)

        return product # (number of SMILES,)
