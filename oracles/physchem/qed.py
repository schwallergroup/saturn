import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.oracle_dataclass import OracleComponentParameters
from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import qed

class QED(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        return np.vectorize(qed)(mols)
