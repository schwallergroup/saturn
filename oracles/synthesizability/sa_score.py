import numpy as np
from oracles.oracle_component import OracleComponent
from rdkit.Chem import Mol
from oracles.synthesizability.sascorer import calculateScore

class SAScore(OracleComponent):
    """
    Synthetic Accessibility score based on: https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8.
    """
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.array[Mol]) -> np.array[float]:
        return np.vectorize(calculateScore)(mols)
