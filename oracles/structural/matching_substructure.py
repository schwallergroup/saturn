import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol



class MatchingSubstructure(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.smarts = parameters.specific_parameters.get("smarts", [])

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        # In case of empty list of SMARTS
        if len(self.smarts) == 0:
            return np.ones(len(mols), dtype=np.float32)
        else:
            match = []
            for mol in mols:
                match.append(any(
                    [mol.HasSubstructMatch(Chem.MolFromSmarts(substructure)) for substructure in self.smarts if Chem.MolFromSmarts(substructure)])
                )
            return 0.5 * (1 + np.array(match))
