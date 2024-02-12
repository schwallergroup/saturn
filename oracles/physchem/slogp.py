import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.oracle_dataclass import OracleComponentParameters
from rdkit.Chem import Mol
from rdkit.Chem.Crippen import MolLogP

class SlogP(OracleComponent):
    """
    Wildman-Crippen LogP based on: https://pubs.acs.org/doi/full/10.1021/ci990307l.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        return np.vectorize(MolLogP)(mols)
