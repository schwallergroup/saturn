"""
Enumerate molecules according to specified reactions to seed the Replay Buffer.
"""
from typing import List, Dict, Union
import os
import json
import random
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem

from oracles.synthesizability.syntheseus import Syntheseus
from models.generator import Generator
from utils.chemistry_utils import is_encodable
from enumeration.utils import enumerated_mol_passes_property_filter

import sys
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)

from enumeration.utils import are_solvable_by_retro
from enumeration.preprocessing import match_bbs


def sample_react(rxn: Dict[str, Union[List, str]]) -> str:
    """Sample and react from a pre-loaded reaction."""

    reaction = AllChem.ReactionFromSmarts(rxn["smirks"])

    # Sample 1
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

def sample_products(
    rxns: Dict[str, List],
    n_seeds: int,
    prior: Generator,
    syntheseus_oracle: Syntheseus
) -> List[str]:
    """
    Sample products from a pre-loaded reaction and building blocks set. 
    Ensure syntheseus_reward = 1 for all enumerated molecules.
    """
    import time
    start_time = time.perf_counter()
    # Take reactions from pre-loaded file
    reactions = rxns["reactions"]

    # Final enumerated SMILES
    enumerated_smiles = set()

    # Enumerate and check until n_seeds
    while len(enumerated_smiles) < n_seeds:

        # List of SMILES batch to verify with Syntheseus
        smiles_batch = []

        # Sample batch of reactions
        sampled_reactions = random.choices(reactions, k=(n_seeds - len(enumerated_smiles) + 20))

        # Sample SMILES
        for reaction in sampled_reactions:

            tries = 0

            while tries < 1000:
                product = sample_react(reaction)

                try:
                    if is_encodable(product, prior):
                        mol = Chem.MolFromSmiles(product)
                        if mol:
                            if enumerated_mol_passes_property_filter(mol):
                                smiles_batch.append(product)
                                break
                except Exception:
                    pass

                tries += 1

        if len(smiles_batch) > 0:
            # Compute the reward with the oracle and append valid SMILES
            syntheseus_rewards = syntheseus_oracle(
                mols=[Chem.MolFromSmiles(smiles) for smiles in smiles_batch],  # Redundant 
                oracle_calls=-1  # Tagging the tracker with -1
            )

            # TODO: Implement enumeration with enforced blocks in the future
            # If only enforcing reactions, successfully meeting the reaction constraints yields a reward of 1.
            # If enforcing blocks, a dense reward should be returned. Additionally adding on a reaction constraints should therefore also yield a dense reward.
            # Therefore, checking for reward != 0 covers both cases
            solved_smiles = [smiles for smiles, reward in 
                             zip(smiles_batch, syntheseus_rewards) if reward != 0]

            enumerated_smiles.update(solved_smiles)

    end_time = time.perf_counter()
    print(f"Enumeration seeding time: {end_time - start_time} seconds")
    
    return list(enumerated_smiles)[:n_seeds]

def rxn_based_enumeration(
    prior_path: str,
    device: str,
    syntheseus_params: Dict[str, str],
    syntheseus_oracle: Syntheseus,
    n_seeds: int
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
    
    # Load pre-filtered file if it exists, otherwise generate it
    if not os.path.exists(os.path.join(prefiltered_rxn_folder, prefiltered_file_name)):

        os.makedirs(prefiltered_rxn_folder, exist_ok=True)
        
        print("Pre-processing reactions and building blocks for replay buffer seeding via enumeration")
        
        # Generate pre-filtered reactions file
        match_bbs(
            bbs_file=building_blocks_path,
            rxn_templates_file=smirks_file,
            save_folder=prefiltered_rxn_folder,
            file_name=prefiltered_file_name,
            rxn_list=rxn_list
        )

    with gzip.open(os.path.join(prefiltered_rxn_folder, prefiltered_file_name), "r") as f:
        rxns = json.load(f)
    
    # Function that generates candidate molecules with the pre-loaded reactions and filters them
    candidate_seeds = sample_products(
        rxns=rxns, 
        n_seeds=n_seeds,
        prior=prior,
        syntheseus_oracle=syntheseus_oracle
    )
    
    print(candidate_seeds)

    return candidate_seeds
