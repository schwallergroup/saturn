import os
import sys
import logging
import argparse
from rdkit import Chem

sys.path.append("../../")

from utils.utils import set_seed_everywhere
from beam_enumeration.beam_enumeration import BeamEnumeration
from large_beam_enumeration import LargeBeamEnumeration
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
parser.add_argument(
    "--substructure_type",
    type=str,
    required=True,
    help="Type of substructures to extract - 'structure' or 'scaffold'"
)
parser.add_argument(
    "--extract_large",
    action="store_true",
    help="Extract large substructures."
)
parser.add_argument(
    "--chemistry_filter",
    action="store_true",
    help="Apply chemistry filters."
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
if args.extract_large:
    beam_enumeration = LargeBeamEnumeration(
        substructure_type=args.substructure_type,
        substructure_min_size=25
    )
else:
    beam_enumeration = BeamEnumeration(
        substructure_type=args.substructure_type,
    )
beam_enumeration.pool_update(agent=checkpointed_model)

# NOTE: Substructure extraction does not take into the raw counts 
#       TODO: Test this later
if args.chemistry_filter:
    # Apply chemistry filters
    #   - No unpaired electrons
    #   - Largest ring <= 6

    def size_largest_ring(smiles: str) -> int:
        """
        Returns the size of the largest ring in the molecule
        """
        mol = Chem.MolFromSmiles(smiles)
        ring_info = mol.GetRingInfo()

        return max([0] + [len(ring) for ring in ring_info.AtomRings()])

    def has_unpaired_electrons(smiles: str) -> bool:
        """
        Returns whether any atom has unpaired electrons - catches radical and charged atoms
        """
        mol = Chem.MolFromSmiles(smiles)
        return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())

    entire_pool = beam_enumeration.entire_pool
    pool = set()
    for smiles in entire_pool.keys():
        if size_largest_ring(smiles) <= 6 and not has_unpaired_electrons(smiles):
            # Flatten stereochemistry and then canonacalize
            processed_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
            pool.add(processed_smiles)
            if len(pool) == 4:
                break
    smiles = pool
else:
    smiles = [Chem.MolToSmiles(mol) for mol in beam_enumeration.pool]

# Save the substructures
with open(args.substructures_output_path, "w") as f:
    for s in smiles:
        f.write(f"{s}\n")
