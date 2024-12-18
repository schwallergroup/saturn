"""Enumerate molecules to load replay buffer with molecules from selected rxns.
"""

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from rdkit.Chem import AllChem
import pandas as pd
import json
import random


def get_smirks_from_list(rxn_list: list[str]) -> dict:
    """Get SMIRKS dict from a list of reaction names.
    """

    # TO DO?: path to smirks, this may break or maybe we could have it inside
    with open("enumeration/smirks.json", "r") as f:
        data = [json.loads(line) for line in f]
    
    smirks_names = {}

    for rxn in rxn_list:

        rxn_smirks = []
        for entry in data:
            if rxn in entry["name"].lower():
                rxn_smirks.append(entry["smirks"])
        
        smirks_names[rxn] = rxn_smirks

    return smirks_names
    

def get_candidates_from_bbs(smirks: str, 
                            bbs: list) -> tuple[str, str]:
    
    """Get a seeding molecule from a dictionary with reaction names and
    SMIRKS and a list of building blocks. Randomly sample and then """

    # Shuffle bbs
    random.shuffle(bbs)

    # Randomly sample reaction
    rxn_name = random.choice(list(smirks.keys()))

    rxn_smirks = random.choice(smirks[rxn_name])
    
    # Get SMARTS for reactants
    smarts = rxn_smirks.split(">>")[0].split(".")

    mol0 = Chem.MolFromSmarts(smarts[0])

    # Sample compatible bb from list
    for bb in bbs:
        mol = Chem.MolFromSmiles(bb)
        if mol.HasSubstructMatch(mol0):
            r0 = mol
            break

    # If bimol
    if len(smarts) > 1:
        mol1 = Chem.MolFromSmarts(smarts[1])

        for bb in bbs:
            mol = Chem.MolFromSmiles(bb)
            if mol.HasSubstructMatch(mol1):
                r1 = mol
                break
    
    # React unimol
    if len(smarts) == 1:
        product = AllChem.ReactionFromSmarts(rxn_smirks).RunReactants((r0,))
    
    # Else react bimol
    else:
        product = AllChem.ReactionFromSmarts(rxn_smirks).RunReactants((r0, r1))
    
    product = Chem.MolToSmiles(product[0][0])

    return product


def get_bbs(bbs_path: str) -> list:
    """Get building blocks from .smi file"""

    bbs = pd.read_csv(bbs_path, 
                      header=None)[0].values
    
    mols = [Chem.MolFromSmiles(smi) for smi in bbs]

    filter_mols = [mol for mol in mols if CalcExactMolWt(mol) > 175]

    filter_mols = [Chem.MolToSmiles(mol) for mol in filter_mols]

    return filter_mols


def seed_enumeration(rxn_list: list[str], 
                     bbs_path: str,
                     n_seeds: int = 10) -> list:
    """Enumerate molecules using target rxns and specified bbs.
    """

    # Open bbs
    bbs = get_bbs(bbs_path)

    # Get smirks
    names_smirks = get_smirks_from_list(rxn_list)

    # Create set to store seed molecules
    seed_smiles = set()

    # Get seeds until list is full
    while len(seed_smiles) < n_seeds:

        try:
            smile = get_candidates_from_bbs(names_smirks, 
                                        bbs)

            seed_smiles.add(smile)
        except: 
            pass

    return list(seed_smiles)
