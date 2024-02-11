import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.oracle_component_parameters import OracleComponentParameters
from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import MolWt

class MolecularWeight(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.array[Mol]) -> np.array[float]:
        return np.vectorize(MolWt)(mols)
