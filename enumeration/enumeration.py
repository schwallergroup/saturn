"""
Enumerate molecules according to specified reactions to seed the Replay Buffer.
"""
from typing import List, Dict, Union
import os
import json
import random
import gzip
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
import pandas as pd

from models.generator import Generator
from utils.chemistry_utils import is_encodable

import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)

from enumeration.utils import (within_molecular_weight_range, 
                               within_small_molecule_size, 
                               is_charged, 
                               longest_aliphatic_c_chain, 
                               passes_ring_filter,
                               are_solvable_by_retro)

from enumeration.preprocessing import match_bbs


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
    # Cap the maximum number of tries in sampling compatible building blocks
    MAX_TRIES = 1000
    tries = 0

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
    while tries < MAX_TRIES:
        idx = random.randint(0, len(building_blocks) - 1)
        mol = Chem.MolFromSmiles(building_blocks[idx])

        if passes_property_filter(mol):
            if mol.HasSubstructMatch(mol0):
                r0 = mol
                break
        else:
            tries += 1
            continue

    # If bimol
    if len(smarts) > 1:
        mol1 = Chem.MolFromSmarts(smarts[1])

        while tries < MAX_TRIES:
            idx = random.randint(0, len(building_blocks) - 1)
            mol = Chem.MolFromSmiles(building_blocks[idx])

            if passes_property_filter(mol):
                if mol.HasSubstructMatch(mol1):
                    r1 = mol
                    break

            else:
                tries += 1
                continue
    
    # React unimol
    if len(smarts) == 1:
        product = AllChem.ReactionFromSmarts(rxn_smirks).RunReactants((r0,))
    
    # Else react bimol
    else:
        product = AllChem.ReactionFromSmarts(rxn_smirks).RunReactants((r0, r1))
    
    product = product[0][0]

    return Chem.MolToSmiles(product) if within_small_molecule_size(product) else None

def sample_react(rxn: Dict[str, Union[List, str]]) -> str:
    """Sample and react from a preloaded reaction"""

    reaction = AllChem.ReactionFromSmarts(rxn["smirks"])

    # sample 1
    r1 = Chem.MolFromSmiles(random.choice(rxn["available_reactants"][0]))

    if rxn["num_reactant"] == 2:
        r2 = Chem.MolFromSmiles(random.choice(rxn["available_reactants"][1]))

        product = reaction.RunReactants((r1, r2))

    # Do not allow more than 2 reactants
    elif rxn["num_reactant"] > 2:
        return None

    else:
        product = reaction.RunReactants((r1,))
    
    product = Chem.MolToSmiles(product[0][0])

    return product 

def sample_products(rxns: Dict[str, List],
                    n_seeds: int,
                    prior: Generator) -> List[str]:
    """Sample products from a preloaded reaction and the corresponding seed
    """
    
    # Take reactions from preloaded file
    reactions = rxns["reactions"]

    # We take a high enough number of samples per reaction (this was done based on tests)
    samples_per_reaction = n_seeds*150//len(reactions)

    seed_smiles = []

    # For each reaction, we randomly generate samples_per_reaction products 
    # FIXME: this is just a way of making sure we will have a large number of solved molecules
    for reaction in reactions:

        reaction_seed = []

        for _ in range(samples_per_reaction):
            product = sample_react(reaction)

            if product:
                reaction_seed.append(product)
        
        seed_smiles.extend(reaction_seed)

    final_smiles = []

    # Load SMILES that can be canonicalized and are in our range
    for smiles in seed_smiles:
        try:
            if within_small_molecule_size(Chem.MolFromSmiles(smiles)) and is_encodable(smiles, prior):
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                final_smiles.append(smiles)
        except Exception:
            pass

    return final_smiles


def rxn_based_enumeration(
    prior_path: str,
    device: str,
    syntheseus_params: Dict[str, str],
    n_seeds: int = 5,
) -> List[str]:
    """
    Enumerate molecules using specified reactions and building blocks.
    """
    
    # Things that will be used from run config
    rxn_list = syntheseus_params["enforced_reactions"]["enforced_rxn_classes"]
    building_blocks_path = syntheseus_params["enforced_reactions"]["seed_building_blocks_file"]
    prefiltered_rxn_folder = syntheseus_params["enforced_reactions"]["seed_reactions_file_folder"]
    smirks_file = os.path.join(BASE_PATH, "smirks.json")

    # Load Prior to check that enumerated SMILES are tokenizable
    prior = Generator.load_from_file(prior_path, device)

    assert os.path.exists(building_blocks_path), f"Seed (by reaction) building blocks file {building_blocks_path} does not exist."
    
    # Load prefiltered file if it exists, otherwise generate it
    if not os.path.exists(os.path.join(prefiltered_rxn_folder, "enumeration_rxns.json.gz")):

        os.makedirs(prefiltered_rxn_folder, exist_ok=True)
        
        print("Generating file")
        # Generate prefiltered reactions file
        match_bbs(building_blocks_path,
                  smirks_file,
                  prefiltered_rxn_folder,
                  rxn_list=rxn_list)

    with gzip.open(os.path.join(prefiltered_rxn_folder, "enumeration_rxns.json.gz"), "r") as f:
        rxns = json.load(f)
    
    # Function that generates candidate molecules with the preloaded reactions and filters them
    candidate_seeds = sample_products(rxns, 
                                      n_seeds,
                                      prior)
        
    # We double check that our retro model 
    final_smiles = are_solvable_by_retro(candidate_seeds,
                                         syntheseus_params)
    
    # If we have more SMILES than n_seeds, randomly sample n_seeds
    if len(final_smiles) > n_seeds:
        final_smiles = random.sample(final_smiles, n_seeds)
    
    else:
        print(f"Loading {len(final_smiles)} in replay buffer")

    return final_smiles
