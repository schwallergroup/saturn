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

from enumeration.utils import are_solvable_by_retro
from enumeration.preprocessing import match_bbs


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
            if is_encodable(smiles, prior): #and within_small_molecule_size(Chem.MolFromSmiles(smiles)):
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
    rxn_names = "_".join(sorted(rxn_list))
    building_blocks_path = syntheseus_params["building_blocks_file"]
    prefiltered_rxn_folder = syntheseus_params["enforced_reactions"]["seed_reactions_file_folder"]
    prefiltered_file_name = f"enumeration_rxns_{rxn_names}.json.gz"
    smirks_file = os.path.join(BASE_PATH, "smirks.json")

    # Load Prior to check that enumerated SMILES are tokenizable
    prior = Generator.load_from_file(prior_path, device)

    assert os.path.exists(building_blocks_path), f"Seed (by reaction) building blocks file {building_blocks_path} does not exist."
    
    # Load prefiltered file if it exists, otherwise generate it
    if not os.path.exists(os.path.join(prefiltered_rxn_folder, prefiltered_file_name)):

        os.makedirs(prefiltered_rxn_folder, exist_ok=True)
        
        print("Generating file")
        
        # Generate prefiltered reactions file
        match_bbs(building_blocks_path,
                  smirks_file,
                  prefiltered_rxn_folder,
                  prefiltered_file_name,
                  rxn_list=rxn_list)

    with gzip.open(os.path.join(prefiltered_rxn_folder, prefiltered_file_name), "r") as f:
        rxns = json.load(f)
    
    # Function that generates candidate molecules with the preloaded reactions and filters them
    candidate_seeds = sample_products(rxns, 
                                      n_seeds,
                                      prior)
    
    # Limit candidate seeds to n_seeds*10, otherwise retro model will take long
    if len(candidate_seeds) > n_seeds*10:
        candidate_seeds = random.sample(candidate_seeds, n_seeds*10)

    # We double check that our retro model 
    final_smiles = are_solvable_by_retro(candidate_seeds,
                                         syntheseus_params)
    
    # If we have more SMILES than n_seeds, randomly sample n_seeds
    if len(final_smiles) > n_seeds:
        final_smiles = random.sample(final_smiles, n_seeds)
    
    else:
        print(f"Loading {len(final_smiles)} in replay buffer")

    return final_smiles
