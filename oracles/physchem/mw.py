from oracles.oracle_component import OracleComponent
from rdkit.Chem import Mol
import numpy as np
from rdkit.Chem.Descriptors import MolWt

class MolecularWeight(OracleComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.array[Mol]) -> np.array[float]:
        return np.vectorize(MolWt)(mols)
