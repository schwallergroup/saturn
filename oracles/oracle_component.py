"""
This module contains the OracleComponent class, which is the base class for all oracles.
Many individual OracleComponents are based on code from https://github.com/MolecularAI/reinvent-scoring/tree/main/reinvent_scoring/scoring/score_components.
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
# TODO: Pharmacophore and Shape Matching, DFT, and MD oracles are not implemented yet
PHYSICS_ORACLES = ["dockstream"]

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
        # FIXME: Hard-coded GEAM oracle to make it run out-of-the-box without changes to Saturn's logic flow
        if self.name == "geam":
            raw_vina, qed_rewards, raw_sa, aggregated_rewards = self(mols)
            return raw_vina, qed_rewards, raw_sa, aggregated_rewards
        else:
            # Calculate the raw property values
            raw_property_values = self(mols, oracle_calls) if self.name in PHYSICS_ORACLES else self(mols) 
            # Apply reward shaping
            # FIXME: in case raw_property_values of 0.0 are good, then there will be a problem when reward shaping
            rewards = self.reward_shaping_function(raw_property_values)
            return raw_property_values, rewards
