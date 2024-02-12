"""
This module contains the OracleComponent class, which is the base class for all oracles.
"""

from abc import ABC, abstractmethod
from rdkit.Chem import Mol
import numpy as np
from oracles.oracle_dataclass import OracleComponentParameters
from oracles.reward_shaping.reward_shaping_function import RewardShapingFunction


class OracleComponent(ABC):
    def __init__(self, parameters: OracleComponentParameters):
        self.parameters = parameters
        self.reward_shaping_function = RewardShapingFunction(
            oracle_name=parameters["name"],
            parameters=parameters["reward_shaping_function_parameters"]
        )

    @abstractmethod
    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        """
        Each OracleComponent implements this method with specifics depending on the property being calculated.
        """
        raise NotImplementedError("__call__ method is not implemented")

    def calculate_reward(self, mols: np.ndarray[Mol]) -> np.ndarray[Mol]:
        """
        All OracleComponents execute __call__ and then apply the reward shaping function to get normalized rewards [0, 1]. 
        Errors are assigned a reward of 0.0.
        """
        # calculate the raw property values
        # FIXME: np.vectorize may not handle computation errors
        raw_property_values = self(mols)
        # apply reward shaping
        return self.reward_shaping_function(raw_property_values)
