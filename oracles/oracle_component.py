"""
This module contains the OracleComponent class, which is a base class for all oracles.
"""

from abc import ABC, abstractmethod
from rdkit.Chem import Mol
import numpy as np


class OracleComponent(ABC):
    def __init__(self, parameters: ComponentParameters):
        self.parameters = parameters

    @abstractmethod
    def __call__(self, mols: np.array[Mol]) -> np.array[float]:
        """
        Each OracleComponent implements this method with specifics depending on the property being calculated.
        """
        raise NotImplementedError("__call__ method is not implemented")

    def calculate_reward(self, mols: np.array[Mol]) -> np.array[Mol]:
        """
        All OracleComponents execute __call__ and then apply the reward shaping function to get normalized rewards [0, 1]. 
        Errors are assigned a reward of 0.0.
        """
        # calculate the raw property values
        raw_property_values = self(mols)
        # apply reward shaping
        return self.apply_reward_shaping_function(raw_property_values)

    def apply_reward_shaping_function(self, query_mols) -> np.array:
        scores = []
        for mol in query_mols:
            try:
                score = self._calculate_phys_chem_property(mol)
            except ValueError:
                score = 0.0
            scores.append(score)
        transform_params = self.parameters.specific_parameters.get(
            self.component_specific_parameters.TRANSFORMATION, {}
        )
        transformed_scores = self._transformation_function(scores, transform_params)
        return np.array(transformed_scores, dtype=np.float32), np.array(scores, dtype=np.float32)
