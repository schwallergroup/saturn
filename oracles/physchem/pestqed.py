import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit.Chem import Mol
from oracles.physchem.utils.pestqed_calc import pestqed



class PestQED(OracleComponent):
    """
    Modified QED class to compute insecticide QED from this paper: 
    https://doi.org/10.1186/s13321-014-0042-6
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        return np.vectorize(self._compute_property)(mols)
    
    def _compute_property(self, mol: Mol) -> float:
        """
        Wrapper function in case of exceptions.
        """
        try:
            return pestqed(mol)
        except Exception:
            return 0.0
