"""
Enumerate molecules according to specified reactions to seed the Replay Buffer.
"""
from typing import List, Dict
import os
import json
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from rdkit.Chem import AllChem
import pandas as pd


def passes_property_filter(mol: Mol) -> bool:
    """
    Check if the building block passess all the property filters.
    """
    # TODO: Add more filters
    if CalcExactMolWt(mol) > 150 and CalcExactMolWt(mol) < 200:
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
    base_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_path, "smirks.json"), "r") as f:
        data = [json.loads(line) for line in f]
    
    smirks_names = {}

    for rxn in rxn_list:

        rxn_smirks = []
        for entry in data:
            if rxn in entry["name"].lower():
                rxn_smirks.append(entry["smirks"])
        
        smirks_names[rxn] = rxn_smirks

    return smirks_names
    

def get_product_from_bbs(
    smirks: Dict[str, List[str]], 
    building_blocks: List[str]
) -> str:
    
    """
    Get a seeding molecule from a dictionary with reaction names and SMIRKS and a list of building blocks. 
    Randomly sample blocks and reactions and return the product.
    """
    # Randomly sample reaction
    rxn_name = random.choice(list(smirks.keys()))

    rxn_smirks = random.choice(smirks[rxn_name])
    
    # Get SMARTS for reactants
    smarts = rxn_smirks.split(">>")[0].split(".")

    mol0 = Chem.MolFromSmarts(smarts[0])

    # Sample compatible building block from list
    while True:
        idx = random.randint(0, len(building_blocks) - 1)
        bb = building_blocks[idx]
        mol = Chem.MolFromSmiles(bb)

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
            bb = building_blocks[idx]
            mol = Chem.MolFromSmiles(bb)

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
    
    product = Chem.MolToSmiles(product[0][0])

    return product

def seed_enumeration(
    rxn_list: List[str], 
    building_blocks_path: str,
    n_seeds: int = 10
) -> List[str]:
    """
    Enumerate molecules using target rxns and specified bbs.
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
            smiles = get_product_from_bbs(
                smirks=names_smirks, 
                building_blocks=bbs
            )
            print(smiles)
            exit()

            seed_smiles.add(smiles)

        except Exception: 
            pass

    return list(seed_smiles)
