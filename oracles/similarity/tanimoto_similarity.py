import numpy as np
from oracles.oracle_component import OracleComponent
from rdkit.Chem import Mol
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import BulkTanimotoSimilarity

class TanimotoSimilarity(OracleComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def __call__(self, reference_mols: np.array[Mol], mols: np.array[Mol]) -> np.array[float]:
        sims = []
        reference_fps = np.vectorize(GetMorganFingerprintAsBitVect(radius=3, nBits=2048))(reference_mols)
        fps = np.vectorize(GetMorganFingerprintAsBitVect(radius=3, nBits=2048))(mols)
        for fp in fps:
            sims.append(np.max(
                [BulkTanimotoSimilarity(fp, ref_fp) for ref_fp in reference_fps]
                ))

        return np.array(sims, dtype=np.float32)
