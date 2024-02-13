"""
This module contains the OracleComponent class, which is the base class for all oracles.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from rdkit.Chem import Mol
import numpy as np
from oracles.dataclass import OracleComponentParameters
from oracles.reward_shaping.reward_shaping_function import RewardShapingFunction
from oracles.reward_shaping.function_parameters import RewardShapingFunctionParameters


class OracleComponent(ABC):
    def __init__(self, parameters: OracleComponentParameters):
        self.parameters = parameters
        self.reward_shaping_function = RewardShapingFunction(
            RewardShapingFunctionParameters(**parameters.reward_shaping_function_parameters)
        )
        self.name = parameters.name
        self.weight = parameters.weight
        self.preliminary_check = parameters.preliminary_check
        self.specific_parameters = parameters.specific_parameters

    @abstractmethod
    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        """
        Each OracleComponent implements this method with specifics depending on the property being calculated.
        """
        raise NotImplementedError("__call__ method is not implemented")

    def calculate_reward(self, mols: np.ndarray[Mol]) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """
        All OracleComponents execute __call__ and then apply the reward shaping function to get normalized rewards [0, 1]. 
        Errors are assigned a reward of 0.0.
        """
        # calculate the raw property values
        raw_property_values = self(mols)
        # apply reward shaping
        # FIXME: in case raw_property_values of 0.0 are good, then there will be a problem when reward shaping
        rewards = self.reward_shaping_function(raw_property_values)
        return raw_property_values, rewards
