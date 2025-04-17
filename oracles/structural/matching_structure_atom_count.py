import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdFMCS



class MatchingSubstructureAtomCount(OracleComponent):
    """
    Count the maximum number of atoms in the Maximum Common Substructure (MCS) between generated molecules and enforced structures.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.enforced_structures = parameters.specific_parameters.get("enforced_structures")
        assert self.enforced_structures is not None, "Enforced Structures must be provided"
        assert type(self.enforced_structures) in [list, str], "Enforced Structures must be a list of SMILES or a path to a file containing SMILES"

        if type(self.enforced_structures) == list:
            self.enforced_structures_mols = [Chem.MolFromSmiles(smiles) for smiles in self.enforced_structures]
        else:
            with open(self.enforced_structures, "r") as f:
                self.enforced_structures_mols = [Chem.MolFromSmiles(line.strip()) for line in f.readlines()]

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        out = []
        for mol in mols:
            # Initialize the maximum MCS atoms to 0
            max_mcs_atoms = 0
            for enforced_structure in self.enforced_structures_mols:
                # Perform MCS (find Maximum Common Substructure)
                mcs_result = rdFMCS.FindMCS(
                        mols=[mol, enforced_structure], 
                        matchChiralTag=True, 
                        bondCompare=rdFMCS.BondCompare.CompareOrderExact, 
                        ringCompare=rdFMCS.RingCompare.StrictRingFusion, 
                        completeRingsOnly=True
                    ) 
                max_mcs_atoms = max(max_mcs_atoms, (mcs_result.numAtoms / enforced_structure.GetNumAtoms()))
            # Store the maximum MCS atoms amongst all enforced structures
            out.append(max_mcs_atoms)
        return np.array(out, dtype=np.float32)
