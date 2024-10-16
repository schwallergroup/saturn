from typing import List
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import RenumberAtoms
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string based on RDKit convention.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)

def canonicalize_smiles_batch(smiles_batch: np.array) -> List[str]:
    """
    Canonicalize a batch of SMILES strings based on RDKit convention.
    """
    return [canonicalize_smiles(smiles) for smiles in smiles_batch]

def randomize_smiles(smiles: str) -> str:
    """
    Shuffle atom numbering to generate a randomized SMILES string.
    Returns original SMILES string on RDKit error.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        new_atom_order = list(range(mol.GetNumHeavyAtoms()))
        random.shuffle(new_atom_order)
        random_mol = RenumberAtoms(mol, newOrder=new_atom_order)
        # TODO: Have option to control stereochemistry
        return Chem.MolToSmiles(random_mol, canonical=False, isomericSmiles=True)
    except Exception:
        return smiles

def randomize_smiles_batch(smiles_batch: np.ndarray[str], prior) -> np.ndarray[str]:
    """
    Randomize a batch of SMILES strings.
    Directly return if smiles_batch is empty.
    """
    if len(smiles_batch) > 0:
        randomized_smiles_batch = np.vectorize(randomize_smiles)(smiles_batch)
        return np.vectorize(can_be_encoded)(smiles_batch, randomized_smiles_batch, prior)
    else:
        return smiles_batch

def can_be_encoded(original_smiles: str, randomized_smiles: str, prior) -> str:
    """
    Check if a SMILES string can be encoded by the Vocabulary.
    """
    try:
        # There may be tokens in the randomized SMILES that are not in the Vocabulary
        # Check if the randomized SMILES can be encoded
        tokens = prior.tokenizer.tokenize(randomized_smiles)
        seq = prior.vocabulary.encode(tokens)
        return randomized_smiles
    except KeyError:
        return original_smiles
    
def get_bemis_murcko_scaffold(smiles: str) -> str:
    """
    Get the Bemis-Murcko scaffold: https://pubs.acs.org/doi/10.1021/jm9602928 of a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        try:
            scaffold = GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
        except Exception:
            return ""
    else:
        return ""

def construct_morgan_fingerprint(smiles: str, radius: int = 2, nBits: int = 1024):
    mol = Chem.MolFromSmiles(smiles)
    return GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)

def construct_morgan_fingerprints_batch(smiles_batch: np.ndarray[str]):
    fps = [construct_morgan_fingerprint(smiles) for smiles in smiles_batch]
    return fps

def construct_morgan_fingerprints_batch_from_file(file_path: str) -> List[np.ndarray[int]]:
    with open(file_path, "r") as f:
        smiles_batch = f.readlines()
    return construct_morgan_fingerprints_batch(smiles_batch)
