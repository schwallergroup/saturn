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
    "--size",
    type=int,
    required=True,
    help="Minimum size of substructures to extract."
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
        substructure_min_size=args.size
    )
else:
    beam_enumeration = BeamEnumeration(
        substructure_type=args.substructure_type,
    )
beam_enumeration.pool_update(agent=checkpointed_model)

# NOTE: Substructure extraction does not take into the raw frequency counts 
#       TODO: Ablate this in the future
if args.chemistry_filter:
    # Apply chemistry filters (inductive bias)
    #   - No unpaired electrons
    #   - Smallest ring size >= 5 and largest ring size <= 6 and no bicyclic rings

    def ring_filter(smiles: str) -> bool:
        """
        Whether to keep the extracted substructure or not based on ring properties.
        Returns True if:
            1. There are no rings

            or if and only if:

            1. No bicyclic rings
            2. Smallest ring size >= 5 and largest ring size <= 6
        """
        mol = Chem.MolFromSmiles(smiles)
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

    def has_unpaired_electrons(smiles: str) -> bool:
        """
        Returns whether any atom has unpaired electrons - catches radicals and some charged atoms.
        """
        mol = Chem.MolFromSmiles(smiles)
        return any(atom.GetNumRadicalElectrons() > 0 for atom in mol.GetAtoms())

    def is_charged(smiles: str) -> bool:
        """
        Returns whether any atom has a charge.
        """
        mol = Chem.MolFromSmiles(smiles)
        return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())

    entire_pool = beam_enumeration.entire_pool
    pool = set()
    for smiles in entire_pool.keys():
        if ring_filter(smiles) and not has_unpaired_electrons(smiles) and not is_charged(smiles):
            # Flatten stereochemistry and then canonicalize
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
