"""
Enumerate molecules according to specified reactions to seed the Replay Buffer.
"""
from typing import List, Dict, Union
import os
import json
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
import pandas as pd

import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)
from enumeration.utils import within_molecular_weight_range, within_small_molecule_size, is_charged, longest_aliphatic_c_chain, passes_ring_filter


def passes_property_filter(mol: Mol) -> bool:
    """
    Check if the building block passess all the property filters.
    """
    # TODO: Add more filters or remove some
    if (within_molecular_weight_range(mol) and 
        not is_charged(mol) and
        longest_aliphatic_c_chain(mol) < 3 and 
        passes_ring_filter(mol)):
        return True
    
    return False

def get_smirks_from_list(rxn_list: list[str]) -> Dict[str, List[str]]:
    """
    Returns dict of the form:

    {
        "rxn_class": [smirks_1, smirks_2, ...]
    }

    Where for each enforced reaction class specified by the user, the associated SMIRKS are extracted.
    """
    # TODO: path to smirks, this may break or maybe we could have it inside
    with open(os.path.join(BASE_PATH, "smirks.json"), "r") as f:
        data = [json.loads(line) for line in f]
    
    smirks_names = {}

    for rxn in rxn_list:

        rxn_smirks = []
        for entry in data:
            if rxn in entry["name"].lower():
                rxn_smirks.append(entry["smirks"])
        
        smirks_names[rxn] = rxn_smirks

    return smirks_names
    

def get_product_from_building_blocks(
    smirks: Dict[str, List[str]], 
    building_blocks: List[str]
) -> Union[str, None]:
    
    """
    Get a seeding molecule from a dictionary with reaction names and SMIRKS and a list of building blocks. 
    Randomly sample blocks and reactions and return the product.

    # FIXME: Currently purposely disallowing for > 2 reactants - returns None
    """
    # Randomly sample reaction
    rxn_name = random.choice(list(smirks.keys()))
    rxn_smirks = random.choice(smirks[rxn_name])
    
    # Get SMARTS for reactants
    smarts = rxn_smirks.split(">>")[0].split(".")
    # FIXME: Do not allow > 2 reactants at the moment
    if len(smarts) > 2:
        return None

    mol0 = Chem.MolFromSmarts(smarts[0])

    # Sample compatible building block from list
    while True:
        idx = random.randint(0, len(building_blocks) - 1)
        mol = Chem.MolFromSmiles(building_blocks[idx])

        if passes_property_filter(mol):
            if mol.HasSubstructMatch(mol0):
                r0 = mol
                break
        else:
            continue

    # If bimol
    if len(smarts) > 1:
        mol1 = Chem.MolFromSmarts(smarts[1])

        while True:
            idx = random.randint(0, len(building_blocks) - 1)
            mol = Chem.MolFromSmiles(building_blocks[idx])

            if passes_property_filter(mol):
                if mol.HasSubstructMatch(mol1):
                    r1 = mol
                    break

            else:
                continue
    
    # React unimol
    if len(smarts) == 1:
        product = AllChem.ReactionFromSmarts(rxn_smirks).RunReactants((r0,))
    
    # Else react bimol
    else:
        product = AllChem.ReactionFromSmarts(rxn_smirks).RunReactants((r0, r1))
    
    product = product[0][0]

    return Chem.MolToSmiles(product) if within_small_molecule_size(product) else None

def rxn_based_enumeration(
    rxn_list: List[str], 
    building_blocks_path: str,
    n_seeds: int = 10
) -> List[str]:
    """
    Enumerate molecules using specified reactions and building blocks.
    """
    assert os.path.exists(building_blocks_path), f"Seed (by reaction) building blocks file {building_blocks_path} does not exist."
    # Read building blocks
    bbs = pd.read_csv(
        building_blocks_path, 
        header=None
    )[0].values

    # Get smirks
    names_smirks = get_smirks_from_list(rxn_list)

    # Create set to store seed molecules
    seed_smiles = set()

    while len(seed_smiles) < n_seeds:

        try:
            smiles = get_product_from_building_blocks(
                smirks=names_smirks, 
                building_blocks=bbs
            )
            if smiles is not None:
                seed_smiles.add(smiles)

        except Exception: 
            pass

    return list(seed_smiles)
