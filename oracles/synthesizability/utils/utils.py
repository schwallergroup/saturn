from typing import List, Union, Dict, Set, Tuple
import numpy as np
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdFMCS
from rdkit.DataStructs import BulkTanimotoSimilarity
from utils.chemistry_utils import canonicalize_smiles, construct_morgan_fingerprint
from oracles.synthesizability.utils.CONSTANTS import FUNCTIONAL_GROUPS


def match_stock(
    query_smiles: str, 
    enforced_building_blocks_file: str
) -> Tuple[bool, str]:
    """
    Check if the query SMILES is in the building blocks stock.
    """
    with open(enforced_building_blocks_file, "r") as f:
        for smiles in f.readlines():
            canonicalized_block_smiles = canonicalize_smiles(smiles.strip())
            if query_smiles == canonicalized_block_smiles:
                return True, canonicalized_block_smiles
    return False, None

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

def extract_functional_groups(smiles: Union[str, List[str]]) -> Union[List[str], Dict[str, List[str]]]:
    """
    Extract the functional groups present in the given SMILES or SMILES list.
    """
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        functional_groups_present = []
        for fg_name, fg_smarts in FUNCTIONAL_GROUPS.items():
            pattern = Chem.MolFromSmarts(fg_smarts)
            if mol.HasSubstructMatch(pattern):
                functional_groups_present.append(fg_name)
        return functional_groups_present
    elif isinstance(smiles, list):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        functional_groups_present = {}
        for smiles, mol in zip(smiles, mols):
            current_fg_present = []
            for fg_name, fg_smarts in FUNCTIONAL_GROUPS.items():
                pattern = Chem.MolFromSmarts(fg_smarts)
                if mol.HasSubstructMatch(pattern):
                    current_fg_present.append(fg_name)
            functional_groups_present[smiles] = current_fg_present
        return functional_groups_present
    else:
        raise ValueError("Invalid input type for smiles. Please provide a single SMILES or a list of SMILES.")

def matched_functional_groups(
    query_smiles: str, 
    enforced_blocks_functional_groups: Set[str], 
    threshold: float = 0.75
) -> bool:
    """
    Check if the query SMILES matches threshold % of the functional groups.
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

def functional_groups_overlap(
    query_smiles: str, 
    enforced_blocks_functional_groups: Dict[str, List[str]], 
) -> float:
    """
    Calculate the *mean* of the fraction of overlap between the query SMILES and each enforced blocks' functional groups.
    """
    fraction_overlaps = []
    query_functional_groups = set(extract_functional_groups(query_smiles))
    for _, fgs in enforced_blocks_functional_groups.items():
        fgs_set = set(fgs)
        overlap = len(query_functional_groups.intersection(fgs_set)) / len(fgs_set)
        fraction_overlaps.append(overlap)

    return sum(fraction_overlaps) / len(fraction_overlaps)

def fuzzy_matching_substructure(
    query_smiles: str,
    enforced_blocks_functional_groups: Dict[str, List[str]],
) -> float:
    """
    Calculate the *max* substructure overlap between the query SMILES and each enforced block.
    """

    def _query_is_in_bbs(query_mol: str, 
                         enforced_blocks_mols: List[Mol]
    ) -> bool:
        """
        Return True if query mol is in enforced building blocks.
        """
        canon_query = Chem.MolToSmiles(query_mol)

        # Edge case when score != 1 but the block is enforced
        canonicalized_bbs_smiles = [Chem.MolToSmiles(mol) 
                                    for mol in enforced_blocks_mols]
        
        is_in_bbs = any([
            (canon_query == smiles) for smiles in canonicalized_bbs_smiles
        ])
        
        return True if is_in_bbs else False

    query_mol = Chem.MolFromSmiles(query_smiles)
    enforced_blocks_mols = [Chem.MolFromSmiles(smiles) for smiles in enforced_blocks_functional_groups.keys()]
    max_mcs_atoms = 0

    # Edge case if query mol is in enforced building blocks
    if _query_is_in_bbs(query_mol, enforced_blocks_mols):
        return 1.0

    # FMS computation
    for block_mol in enforced_blocks_mols:

        # Perform MCS (find Maximum Common Substructure)
        mcs_result = rdFMCS.FindMCS(
                mols=[query_mol, block_mol],
                matchChiralTag=True,
                bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                ringCompare=rdFMCS.RingCompare.StrictRingFusion,
                completeRingsOnly=True
            )
        overlap = mcs_result.numAtoms / block_mol.GetNumAtoms()
        if int(overlap) == 1:

            # Remove stereochemistry from molecules to guard against stereo problems
            query_mol_copy = deepcopy(query_mol)
            block_mol_copy = deepcopy(block_mol)

            Chem.RemoveStereochemistry(query_mol_copy)
            Chem.RemoveStereochemistry(block_mol_copy)

            if (
                canonicalize_smiles(query_smiles) == canonicalize_smiles(Chem.MolToSmiles(block_mol)) 
                or Chem.MolToSmiles(query_mol_copy, canonical=True) == Chem.MolToSmiles(block_mol_copy, canonical=True)
            ):
                return 1.0

            # Edge case
            else:
                asymmetric_overlap = mcs_result.numAtoms / query_mol.GetNumAtoms()
                assert int(asymmetric_overlap) != 1, "Asymmetric FMS error"
                return asymmetric_overlap
        else:
            max_mcs_atoms = max(max_mcs_atoms, overlap)
    return max_mcs_atoms
    
def tango_reward(
    query_smiles: str, 
    enforce_blocks_fps: List[np.ndarray[int]],
    enforced_blocks_functional_groups: Dict[str, List[str]],
    reward_type: str,
    tango_weights: Dict[str, float]
) -> float:
    """
    Calculate all TANGO rewards.
    """
    tanimoto_weight = tango_weights["tanimoto"]
    fg_weight = tango_weights["fg"]
    fms_weight = tango_weights["fms"]

    tanimoto_similarity = get_max_stock_similarity(
        query_smiles=query_smiles, 
        enforced_building_blocks_fps=enforce_blocks_fps
    )
    
    # Compute FG overlap depending on reward type
    if "fg" in reward_type or "all" in reward_type:
        fg_overlap = functional_groups_overlap(
            query_smiles=query_smiles, 
            enforced_blocks_functional_groups=enforced_blocks_functional_groups
        )

    fms_overlap = fuzzy_matching_substructure(
        query_smiles=query_smiles, 
        enforced_blocks_functional_groups=enforced_blocks_functional_groups
    )
    if reward_type == "tango_fg":
        assert tanimoto_weight + fg_weight == 1, "TANGO-FG weights must sum to 1."
        return (tanimoto_similarity * tanimoto_weight) + (fg_overlap * fg_weight)
    elif reward_type == "tango_fms":
        assert tanimoto_weight + fms_weight == 1, "TANGO-FMS weights must sum to 1."
        return (tanimoto_similarity * tanimoto_weight) + (fms_overlap * fms_weight)
    elif reward_type == "tango_all":
        print(abs((tanimoto_weight + fg_weight + fms_weight) - 1))
        assert abs((tanimoto_weight + fg_weight + fms_weight) - 1) <= 1.1e-2, "TANGO-All weights must sum to 1 within a few decimal points."
        return (tanimoto_similarity * tanimoto_weight) + (fg_overlap * fg_weight) + (fms_overlap * fms_weight)

def get_node_reward(
    reward_type: str,
    query_smiles: str,
    enforce_blocks_fps: List[np.ndarray[int]],
    enforced_blocks_functional_groups: Dict[str, List[str]],
    tango_weights: Dict[str, float]
) -> float:
    """
    Calculate the reward for a given node:

        1. *Max* Tanimoto similarity to the enforced building blocks
        2. *Mean* Functional Groups overlap to the enforced building blocks
        3. *Max* Fuzzy Matching Substructure to the enforced building blocks
        4. TANGO-FG: *Max* Tanimoto similarity + *Mean* Functional Groups overlap
        5. TANGO-FMS: *Max* Tanimoto similarity + *Max* Fuzzy Matching Substructure
        6. TANGO-All: *Max* Tanimoto similarity + *Mean* Functional Groups overlap + *Max* Fuzzy Matching Substructure

    """
    if reward_type in ["tanimoto", "tanimoto_similarity", "tan_sim", "tansim"]:
        reward = get_max_stock_similarity(
            query_smiles=query_smiles,
            enforced_building_blocks_fps=enforce_blocks_fps
        )
    elif reward_type in ["fg", "functional_groups"]:
        reward = functional_groups_overlap(
            query_smiles=query_smiles,
            enforced_blocks_functional_groups=enforced_blocks_functional_groups
        )
    elif reward_type in ["fms", "fuzzy_ms", "fuzzy_matching_substructure"]:
        reward = fuzzy_matching_substructure(
            query_smiles=query_smiles,
            enforced_blocks_functional_groups=enforced_blocks_functional_groups
        )
    elif "tango" in reward_type:
        reward = tango_reward(
            query_smiles=query_smiles,
            enforce_blocks_fps=enforce_blocks_fps,
            enforced_blocks_functional_groups=enforced_blocks_functional_groups,
            reward_type=reward_type,
            tango_weights=tango_weights
        )
    else:
        raise ValueError(f"Invalid reward type: {reward_type}")
    return reward

def get_percentage_of_carbon(
    smiles_bb: str, 
    smiles_target: str
) -> float:
    """
    Get percentage of carbon atoms in structure based on reference molecule.
    """

    bb = Chem.MolFromSmiles(smiles_bb)
    target = Chem.MolFromSmiles(smiles_target)

    # Find MCS. We use CompareAny
    mcs = rdFMCS.FindMCS(
        mols = [bb, target],
        matchChiralTag=True,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringCompare=rdFMCS.RingCompare.StrictRingFusion,
        completeRingsOnly=True
    )

    # Get match
    matched_atoms = Chem.MolFromSmarts(mcs.smartsString).GetAtoms()

    # Get number of matched carbons 
    matched_C = len([atom for atom in matched_atoms if atom.GetSymbol() == "C"])
    
    # Get total number of carbons
    total_C = len([atom for atom in target.GetAtoms() if atom.GetSymbol() == "C"])
    assert total_C > 0, "Total number of carbons must be greater than 0."
    
    return matched_C/total_C

def shape_path_length_reward(
    length: int,
    low: float = 1.0,
    high: float = 8.0,
    k: float = 0.25
) -> float:
    """
    Hard-coded reverse sigmoid reward tranformation for the path length of a synthetic route.
    Output is in the range [0, 1].
    Used if minimizing path length while also enforcing block and/or reaction constraints.
    """
    return 1 / (1 + 10 ** (k * (length - (high + low) / 2) * 10 / (high - low)))