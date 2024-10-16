import re
from typing import Set
import sys
import numpy as np
from rdkit import Chem
from rxnmapper import RXNMapper

def run_rxnmapper(
    generated_smiles_path: str, 
    enforced_structures_smiles_path: str
) -> np.array:
    """
    For every generated molecule, find the maximum fraction of atoms that can be mapped to the enforced structures.
    """
    def extract_atom_labels(mapped_rxn: str) -> Set[str]:
        """
        Extract the atom labels from the mapped reaction.
        """
        return set(re.findall(r":(\d+)", mapped_rxn))

    # Read the generated and enforced structures from the input files
    generated_smiles, enforced_structures_smiles = [], []
    with open(generated_smiles_path, "r") as f:
        generated_smiles = [line.strip() for line in f.readlines()]
    with open(enforced_structures_smiles_path, "r") as f:
        enforced_structures_smiles = [line.strip() for line in f.readlines()]

    # Run rxnmapper for each generated SMILES against each enforced SMILES
    out = []
    rxn_mapper = RXNMapper()
    for gen_smiles in generated_smiles:
        max_fraction_mapped_atoms = 0
        for enforced_smiles in enforced_structures_smiles:
            enforced_mol = Chem.MolFromSmiles(enforced_smiles)
            pseudo_rxn = f"{enforced_smiles}>>{gen_smiles}"
            mapped_result = rxn_mapper.get_attention_guided_atom_maps([pseudo_rxn])[0]["mapped_rxn"]

            precusor, product = mapped_result.split(">>")[0], mapped_result.split(">>")[1]
            precusor_labels, product_labels = extract_atom_labels(precusor), extract_atom_labels(product)
            overlapping_labels = precusor_labels.intersection(product_labels)

            max_fraction_mapped_atoms = max(max_fraction_mapped_atoms, (len(overlapping_labels) / enforced_mol.GetNumAtoms()))

        out.append(max_fraction_mapped_atoms)

    return np.array(out)

if __name__ == "__main__":
    generated_smiles_path = sys.argv[1] 
    enforced_structures_smiles_path = sys.argv[2]
    results = run_rxnmapper(
        generated_smiles_path=generated_smiles_path, 
        enforced_structures_smiles_path=enforced_structures_smiles_path
    )
    print(results)  # Capture output
