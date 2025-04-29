import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.DataStructs import BulkTanimotoSimilarity



class JaccardDistanceDataset(OracleComponent):
    """
    Jaccard Distance to a dataset of reference molecules.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        self.radius = self.parameters.specific_parameters.get("radius", 3)
        self.use_counts = self.parameters.specific_parameters.get("use_counts", True)
        self.use_features = self.parameters.specific_parameters.get("use_features", True)

        # Load the dataset and construct the fingerprints
        dataset_path = self.parameters.specific_parameters.get("dataset_path", None)
        assert dataset_path.endswith(".smi"), "Dataset must be a SMILES file."
        self.construct_fingerprints(dataset_path)

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        sims = []
        query_fps = [
            GetMorganFingerprint(
                mol=mol, 
                radius=self.radius, 
                useCounts=self.use_counts, 
                useFeatures=self.use_features
            ) for mol in mols
        ]

        for fp in query_fps:
            # Compute the maximum similarity to any reference fingerprint
            sims.append(np.max(
                [BulkTanimotoSimilarity(fp, self.reference_fingerprints)]
                )
            )

        return 1 - np.array(sims, dtype=np.float32)
    
    def construct_fingerprints(self, dataset_path: str) -> None:
        smiles = []
        with open(dataset_path, "r") as f:
            for line in f:
                smiles.append(line.replace("\n", "").strip())

        mols = [Chem.MolFromSmiles(s) for s in smiles]

        self.reference_fingerprints = [
            GetMorganFingerprint(
                mol=mol, 
                radius=self.radius, 
                useCounts=self.use_counts, 
                useFeatures=self.use_features
            ) for mol in mols
        ]
