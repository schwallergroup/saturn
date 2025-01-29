"""
Helper functions for Physico-chemical property filters and retro solvable SMILES.
"""
from typing import List, Dict, Union
import os
import yaml
import json
import subprocess
import tempfile
import shutil
import numpy as np

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from utils.chemistry_utils import canonicalize_smiles

# Constrain molecular weight
def within_molecular_weight_range(mol: Mol) -> bool:
    """Returns whether the molecular weight is within the specified range."""
    return 150 < CalcExactMolWt(mol) < 200

def within_small_molecule_size(mol: Mol) -> bool:
    """Returns whether the molecular weight > 300."""
    return CalcExactMolWt(mol) > 300

def more_than_five_heavy_atoms(mol: Mol) -> bool:
    """Returns whether the molecule has more than 5 heavy atoms."""
    return len(mol.GetAtoms()) >= 5

# Exclude SMILES with charges
def is_charged(mol: Mol) -> bool:
    """Returns whether any atom has a charge."""
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())
    
# Exclude SMILES with aliphatic chains longer than 3
SMARTS_CHAINS = [Chem.MolFromSmarts("-".join(["[CR0H2]"]*i)) for i in range(1, 11)]
def longest_aliphatic_c_chain(mol: Mol) -> int:
    """Returns the length of the longest aliphatic chain."""
    length = 0
    for chain in SMARTS_CHAINS:
        if mol.HasSubstructMatch(chain):
            length += 1
        else:
            break
    return length

# 5 <= ring size <= 6
def passes_ring_filter(mol: Mol) -> bool:
    """
    Whether to keep the extracted substructure or not based on ring properties.
    Returns True if:
        1. There are no rings

        or if and only if:

        1. No bicyclic rings
        2. Smallest ring size >= 5 and largest ring size <= 6
    """
    ring_info = mol.GetRingInfo()

    ring_sizes = [len(ring) for ring in ring_info.AtomRings()]

    # If there are no rings, return True
    if len(ring_sizes) == 0:
        return True

    # Check for bicylic rings
    elif len(ring_sizes) > 1:
        bicyclic_rule = Chem.MolFromSmarts("[R2]")
        if mol.HasSubstructMatch(bicyclic_rule):
            return False

    # If the largest ring size is greater than 6, return False
    if max(0, *ring_sizes) > 6:
        return False
    else:
        # If there are 3- and 4-membered rings, return False
        for size in [3, 4]:
            if size in ring_sizes:
                return False
        return True

def building_block_passes_property_filter(mol: Union[Mol, str]) -> bool:
    """
    Check if the building block passess all the property filters.
    """
    # TODO: Add more filters or remove some
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    if (within_molecular_weight_range(mol) and 
        not is_charged(mol) and
        longest_aliphatic_c_chain(mol) < 3 and 
        more_than_five_heavy_atoms(mol) and
        passes_ring_filter(mol)):
        return True
    
    return False

def enumerated_mol_passes_property_filter(mol: Mol) -> bool:
    """
    Check if the enumerated molecule passes all the property filters.
    """
    return (within_small_molecule_size(mol) and 
            not is_charged(mol) and
            longest_aliphatic_c_chain(mol) < 3 and 
            passes_ring_filter(mol))

def are_solvable_by_retro(
    smiles: List[str], 
    config: Dict[str, str]
) -> List[str]:
    """
    Take a list of SMILES, run retrosynthesis model and return only the solvable ones.
    """
    # 1. Make a temporary directory to store the SMILES and output results
    temp_dir = tempfile.mkdtemp()

    # 2. Write the SMILES to the temporary directory
    with open(os.path.join(temp_dir, "smiles.smi"), "w") as f:
        # Syntheseus does not accept empty lines
        for idx, s in enumerate(smiles):
            if idx < len(smiles) - 1:
                f.write(f"{s}\n")
            else:
                f.write(f"{s}")

    # 3. Write the config.yml to the temporary directory
    write_config(temp_dir, config)

    # 4. Run Syntheseus
    output = subprocess.run([
        "conda",
        "run",
        "-n",
        config["syntheseus_env_name"],
        "syntheseus",
        "search",
        "--config", os.path.join(temp_dir, "config.yml")
    ], capture_output=True)

    # 5. Parse the output
    is_solved = np.zeros(len(smiles))

    try:
        # Syntheseus output is tagged by the reaction model name
        reaction_model = parse_model_name(config["reaction_model"])
        output_results_dir = [folder for folder in os.listdir(os.path.join(temp_dir)) if reaction_model in folder][0]

        # If there was more than 1 SMILES, Syntheseus outputs the results for each SMILES in a separate folder
        if len(smiles) > 1:
            output_files = [file for file in os.listdir(os.path.join(temp_dir, output_results_dir)) if not file.endswith(".json")]
            # *Important* to sort by ascending integer order so the molecules to output mapping is correct
            output_files = sorted(output_files, key=lambda x: int(x))

        # Otherwise, Syntheseus outputs the results directly in the output_results_dir folder
        else:
            output_files = [os.path.join(temp_dir, output_results_dir)]

        # Loop through the results for each query SMILES and extract the number of model calls 
        # required to solve. This is the number of reaction steps 
        for idx, mol_results in enumerate(output_files):
            # Load the JSON file
            with open(os.path.join(temp_dir, output_results_dir, mol_results, "stats.json"), "r") as f:
                stats = json.load(f)
            # Extract the number of steps
            num_rxn_steps = stats["soln_time_rxn_model_calls"]
            is_solved[idx] = 1 if num_rxn_steps != np.inf else 0

    except Exception as e:
        print(e)
        pass

    # 6. Delete the temporary directory and Syntheseus output
    shutil.rmtree(temp_dir)

    solvable_smiles = [s for s, solved in zip(smiles, is_solved) if solved]

    return solvable_smiles

def write_config(
    dir_path: str, 
    run_config: Dict[str, str]
) -> None:
    """
    Syntheseus can take as input a yaml file for easy execution. Write this yaml file.
    """
    # TODO: Can expose more Syntheseus parameters to the user in the future
    config = {
        "inventory_smiles_file": run_config["building_blocks_file"],
        "search_targets_file": os.path.join(dir_path, "smiles.smi"),
        "model_class": parse_model_name(run_config["reaction_model"]),
        "time_limit_s": run_config["time_limit_s"],
        # Only return 1 result for now - this implies that if 1 solution is found, Syntheseus stops
        # NOTE: In retrosynthesis, it can be very useful to return multiple routes (unless a model's top-1 accuracy is 100%) 
        #       but we do this for simplicity at the moment, i.e., so long as *a* route is found, consider the molecule solved
        "num_top_results": 1,
        "results_dir": dir_path,
        # Saves storage memory
        "save_graph": False
    }

    # Write the data to the YAML file
    with open(os.path.join(dir_path, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def parse_model_name(model_name: str) -> str:
    """
    Syntheseus expects proper capitalization of the model names. Parse user input and return the correct model name.
    """
    # NOTE: Assumes the user is not providing their own trained model.
    #       The default behaviour in Syntheseus is to download a trained model by the authors.
    #       These models are stored in .cache/torch/syntheseus by default.
    assert model_name is not None, "Please provide the reaction model name."
    if model_name in ["localretro", "LocalRetro"]:
        return "LocalRetro"
    elif model_name in ["retroknn", "RetroKNN"]:
        return "RetroKNN"
    elif model_name in ["rootaligned", "RootAligned"]:
        return "RootAligned"
    elif model_name in ["graph2edits", "Graph2Edits", "graph2edit", "Graph2Edit"]:
        return "Graph2Edits"
    elif model_name in ["megan", "MEGAN"]:
        return "MEGAN"
    else:
        # TODO: Support all the models in Syntheseus 
        raise ValueError(f"Model name {model_name} not recognized or not supported yet.")
