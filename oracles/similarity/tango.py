import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol

from utils.chemistry_utils import construct_morgan_fingerprints_batch, construct_morgan_fingerprints_batch_from_file
from oracles.synthesizability.utils.utils import extract_functional_groups, tango_reward

class Tango(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        self.reward_type = parameters.specific_parameters.get("reward_type", None)
        assert self.reward_type in ["tango_fg", "tango_fms", "tango_all"], "Reward Type must be one of tango_fg, tango_fms, or tango_all"

        self.tango_weights = self.parameters.specific_parameters.get("tango_weights", None)
        assert self.tango_weights is not None, "Please provide TANGO weights."

        self.enforced_structures = parameters.specific_parameters.get("enforced_structures")
        assert self.enforced_structures is not None, "Enforced Structures must be provided"
        assert type(self.enforced_structures) in [list, str], "Enforced Structures must be a list of SMILES or a path to a file containing SMILES"

        if type(self.enforced_structures) == list:
            self.enforced_structures_smiles = self.enforced_structures
            self.enforced_structures_fps = construct_morgan_fingerprints_batch(self.enforced_structures_smiles)
        else:
            self.enforced_structures_smiles = [smiles for smiles in open(self.enforced_structures, "r").readlines()]
            self.enforced_structures_fps = construct_morgan_fingerprints_batch_from_file(self.enforced_structures)

        self.enforced_structures_functional_groups = extract_functional_groups(self.enforced_structures_smiles)

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        rewards = []
        for query_mol in mols:
            query_smiles = Chem.MolToSmiles(query_mol, canonical=True)
            tango = tango_reward(
                query_smiles=query_smiles,
                enforce_blocks_fps=self.enforced_structures_fps,
                enforced_blocks_functional_groups=self.enforced_structures_functional_groups,
                reward_type=self.reward_type,
                tango_weights=self.tango_weights
            )
            rewards.append(tango)

        return np.array(rewards, dtype=np.float32)
        