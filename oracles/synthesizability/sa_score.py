import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.oracle_dataclass import OracleComponentParameters
from rdkit.Chem import Mol
from oracles.synthesizability.sascorer import calculateScore

class SAScore(OracleComponent):
    """
    Synthetic Accessibility score based on: https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        return np.vectorize(calculateScore)(mols)
