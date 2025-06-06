import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.DataStructs import BulkTanimotoSimilarity



class JaccardDistance(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        self.radius = self.parameters.specific_parameters.get("radius", 3)
        self.use_counts = self.parameters.specific_parameters.get("use_counts", True)
        self.use_features = self.parameters.specific_parameters.get("use_features", True)
        self.reference_smiles = self.parameters.specific_parameters.get("reference_smiles", [])
        assert len(self.reference_smiles) > 0, "Please provide the reference SMILES for the Jaccard Distance calculation."
        self.reference_mols = [Chem.MolFromSmiles(smiles) for smiles in self.reference_smiles]
        self.reference_fingerprints = [
            GetMorganFingerprint(mol=mol, radius=self.radius, useCounts=self.use_counts, useFeatures=self.use_features) for mol in self.reference_mols
        ]

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        sims = []
        fps = [GetMorganFingerprint(mol=mol, radius=self.radius, useCounts=self.use_counts, useFeatures=self.use_features) for mol in mols]

        for fp in fps:
            sims.append(np.max(BulkTanimotoSimilarity(fp, self.reference_fingerprints)))

        return 1 - np.array(sims, dtype=np.float32)
