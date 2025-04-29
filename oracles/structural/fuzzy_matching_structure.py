import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdFMCS



class FuzzyMatchingSubstructure(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.smarts = parameters.specific_parameters.get("smarts", [])
        self.smarts_mol = Chem.MolFromSmarts(self.smarts)
        self.smarts_mol_num_atoms = self.smarts_mol.GetNumAtoms()
        self.tolerance = parameters.specific_parameters.get("tolerance", 0.5)
        assert self.tolerance > 0 and self.tolerance <= 1, "Fuzzy Matching Structure Tolerance must be between 0 and 1"


    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        # In case of empty list of SMARTS
        if len(self.smarts) == 0:
            return np.ones(len(mols), dtype=np.float32)
        else:
            out = []
            for mol in mols:
                # Perform MCS (find Maximum Common Substructure)
                mcs_result = rdFMCS.FindMCS(
                        mols=[mol, self.smarts_mol], 
                        matchChiralTag=True, 
                        bondCompare=rdFMCS.BondCompare.CompareOrderExact, 
                        ringCompare=rdFMCS.RingCompare.StrictRingFusion, 
                        completeRingsOnly=True
                    ) 
                mcs_num_atoms = mcs_result.numAtoms
                matched = True if (mcs_num_atoms / self.smarts_mol_num_atoms > self.tolerance) else False
                out.append(matched)
            return np.array(out, dtype=np.float32)
