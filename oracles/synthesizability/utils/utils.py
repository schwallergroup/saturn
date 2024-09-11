from typing import List, Set
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdFMCS
from rdkit.DataStructs import BulkTanimotoSimilarity
from utils.chemistry_utils import canonicalize_smiles, construct_morgan_fingerprint
from oracles.synthesizability.utils.CONSTANTS import FUNCTIONAL_GROUPS


def match_stock(
    query_smiles: str, 
    enforced_building_blocks_file: str
) -> bool:
    """
    Check if the query SMILES is in the building blocks stock.
    """
    with open(enforced_building_blocks_file, "r") as f:
        for smiles in f.readlines():
            if query_smiles == canonicalize_smiles(smiles.strip()):
                return True
    return False

def get_max_stock_similarity(
    query_smiles: str, 
    enforced_building_blocks_fps: List[np.ndarray[int]]
) -> float:
    """
    Get the max Tanimoto similarity of the query SMILES to the enforced building blocks stock.
    """
    query_fp = construct_morgan_fingerprint(query_smiles)
    return np.max([BulkTanimotoSimilarity(query_fp, enforced_building_blocks_fps)])      

def matched_fuzzy_substructure(
    generated_smiles: str, 
    enforced_blocks: List[Mol],
    threshold: float = 0.50
) -> bool:
    """
    Check if the generated SMILES matches threshold % of the reference blocks' structures.
    """
    generated_mol = Chem.MolFromSmiles(generated_smiles)
    for eb in enforced_blocks:
        # Number of atoms in the enforced block
        ref_block_num_atoms = eb.GetNumAtoms()

        # Find Maximum Common Substructure (MCS)
        mcs_result = rdFMCS.FindMCS(
            mols=[generated_mol, eb], 
            matchChiralTag=True,  # Chirality has to match
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,  # Bond order has to match, i.e., single and aromatic are different
            ringCompare=rdFMCS.RingCompare.StrictRingFusion,  # Rings have to match
            completeRingsOnly=True  # Only consider complete rings
        ) 

        # Number of atoms in the MCS
        mcs_num_atoms = mcs_result.numAtoms

        # Check if the generated molecule has a MCS of at least threshold % to the enforced block
        if (mcs_num_atoms / ref_block_num_atoms) > threshold:
            return True
    return False

def extract_functional_groups(enforced_building_blocks_file: str) -> Set[str]:
    """
    Extract the functional groups from the reference blocks' SMILES.
    """
    matched_groups = set()

    # Read the building blocks file
    with open(enforced_building_blocks_file, "r") as f:
        for smiles in f.readlines():
            for _, smarts in FUNCTIONAL_GROUPS.items():
                pattern = Chem.MolFromSmarts(smarts)
                if Chem.MolFromSmiles(smiles).HasSubstructMatch(pattern):
                    matched_groups.add(smarts)
    return matched_groups

def matched_functional_groups(
    query_smiles: str, 
    enforced_blocks_functional_groups: Set[str], 
    threshold: float = 0.75
) -> bool:
    """
    Check if the query SMILES matches threshold %of the functional groups.
    """
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_functional_groups = set()
    for _, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if query_mol.HasSubstructMatch(pattern):
            query_functional_groups.add(smarts)

    # Check that at least threshold % of the query functional groups are present in the enforced blocks functional groups
    count = 0
    for fg in query_functional_groups:
        if fg in enforced_blocks_functional_groups:
            count += 1
    return count > int(len(query_functional_groups) * threshold)