import os
import sys
import logging
import argparse
from rdkit import Chem

sys.path.append("../")

from utils.utils import set_seed_everywhere
from beam_enumeration.beam_enumeration import BeamEnumeration
from models.generator import Generator

parser = argparse.ArgumentParser(description="Run manual Beam Enumeration.")
parser.add_argument(
    "--model_path", 
    type=str,
    required=True,
    help="Path to the checkpointed model."
)
parser.add_argument(
    "--substructures_output_path", 
    type=str,
    required=True,
    help="Path to the substructures output file."
)

args = parser.parse_args()

# Set the seed
set_seed_everywhere(
    seed=0, 
    device="cuda"
)

# Load the model
checkpointed_model = Generator.load_from_file(args.model_path, device="cuda")

# Run Beam Enumeration
beam_enumeration = BeamEnumeration()
beam_enumeration.pool_update(agent=checkpointed_model)

# Save the substructures
smiles = [Chem.MolToSmiles(mol) for mol in beam_enumeration.pool]
with open(args.substructures_output_path, "w") as f:
    for smile in smiles:
        f.write(f"{smile}\n")
