from utils.chemistry_utils import canonicalize_smiles, canonicalize_smiles_batch
from oracles.synthesizability.utils.enforced_blocks_df.dataclass import EnforcedBlocksDiversityFilterParameters


class EnforcedBlocksDiversityFilter:
    """
    Implements Diversity Filter as described in the paper: 
    https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0

    But for use in constrained synthesizability to penalize the same enforced block from being used multiple times.
    """
    def __init__(
        self, 
        parameters: EnforcedBlocksDiversityFilterParameters
    ):
        self.parameters = parameters
        # Track the number of times a given enforced block has been incoporated in a generated molecule
        self.enforced_blocks = [smiles.strip() for smiles in open(parameters.enforced_building_blocks_file, "r").readlines()]
        self.enforced_blocks = canonicalize_smiles_batch(self.enforced_blocks)
        self.bucket_history = dict()
        self.bucket_size = parameters.bucket_size

    def update(
        self,
        smiles: str
    ) -> None:
        """
        Update the bucket history based on the enforced blocks.
        """
        smiles = canonicalize_smiles(smiles)
        if smiles in self.enforced_blocks:
            if smiles in self.bucket_history:
                self.bucket_history[smiles] += 1
            else:
                self.bucket_history[smiles] = 1

    def penalize_reward(
        self,
        smiles: str,
        reward: float
    ) -> float:
        """
        Truncate the reward to 0.0 if the enforced block has been used too many times.
        """
        if smiles in self.bucket_history and self.bucket_history[smiles] > self.bucket_size:
            return 0.0
        else:
            return reward
