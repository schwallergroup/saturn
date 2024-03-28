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

# physics-based oracle (e.g., docking) output extra information (e.g., docking poses)
# flag these so that the number of oracle calls can be used for prefixing the output files
PHYSICS_ORACLES = ["dockstream", "rocs", "shapelinker", "gromacs", "orca"]

class OracleComponent(ABC):
    """
    Base class for all OracleComponents. 
    OracleComponents are used to calculate the properties of a set of molecules.
    """
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

    def calculate_reward(self, mols: np.ndarray[Mol], oracle_calls: int) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """
        All OracleComponents execute __call__ and then apply the reward shaping function to get normalized rewards [0, 1]. 
        Errors are assigned a reward of 0.0.

        Oracle calls is only used for the physics-based oracles which use it as a prefix for the output files.
        """
        # FIXME: hard-coded GEAM oracle to make it run out-of-the-box without changes to Saturn's logic flow
        if self.name == "geam":
            vina_reward, qed_reward, sa_reward, aggregated_reward = self(mols)
            return vina_reward, qed_reward, sa_reward, aggregated_reward
        else:
            # calculate the raw property values
            raw_property_values = self(mols, oracle_calls) if self.name in PHYSICS_ORACLES else self(mols) 
            # apply reward shaping
            # FIXME: in case raw_property_values of 0.0 are good, then there will be a problem when reward shaping
            rewards = self.reward_shaping_function(raw_property_values)
            return raw_property_values, rewards
