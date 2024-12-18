"""
Helper functions for Physico-chemical property filters.
"""
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt


# Constrain molecular weight
def within_molecular_weight_range(mol: Mol) -> bool:
    """Returns whether the molecular weight is within the specified range."""
    return 150 < CalcExactMolWt(mol) < 200
    
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
        return False

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
