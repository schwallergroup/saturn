from oracles.oracle_component import OracleComponent
from rdkit.Chem import Mol
import numpy as np
from rdkit.Chem.Crippen import MolLogP

class SlogP(OracleComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.array[Mol]) -> np.array[float]:
        """Based on: https://pubs.acs.org/doi/full/10.1021/ci990307l"""
        return np.vectorize(MolLogP)(mols)
