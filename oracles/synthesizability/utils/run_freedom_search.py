"""
Given an input SMILES, run search in Chemspace's Freedom 4.0 space.
"""
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdSynthonSpaceSearch



def freedom_search(
    smiles: str,
    random_seed: int,
    max_hits: int
) -> bool:
    """
    Perform an exact match search for a given SMILES string and retrieve FreedomSpace ID and SMILES.
    Returns the first exact match found.
    Checks the first 1000 results to save time - works for most cases.
    If exact match not found - performs a deep search.

    Args:
        smiles (str): Input SMILES string for the query molecule.

    Returns:
        bool: whether input SMILES was found in Freedom or not
    """
    # Random seed is fixed in both cases, in second case we use it 
    params = rdSynthonSpaceSearch.SynthonSpaceSearchParams()
    params.randomSeed = random_seed

    params_extended = rdSynthonSpaceSearch.SynthonSpaceSearchParams()
    params_extended.maxHits = max_hits
    params_extended.randomSeed = random_seed

    try:
        query_mol = Chem.MolFromSmiles(smiles)
        canonical_smiles = Chem.MolToSmiles(query_mol, canonical=True)

        results = synthonspace.SubstructureSearch(
            query_mol,
            params
        )

        for mol in results.GetHitMolecules():
            # If hit
            if Chem.MolToSmiles(mol, canonical=True) == canonical_smiles:
                return 1

        if len(results.GetHitMolecules()) < 1000:
            return 0

        else:
            # Making a deep search
            results = synthonspace.SubstructureSearch(
                query_mol, 
                params_extended
            )
            
            for mol in results.GetHitMolecules():
                if Chem.MolToSmiles(mol, canonical=True) == canonical_smiles:
                    return 1
            
            return 0
        
    except Exception:
        return 0

if __name__ == "__main__":
    smiles_list = sys.argv[1]
    synthons_file = sys.argv[2]
    random_seed = int(sys.argv[3])
    max_hits = int(sys.argv[4])

    smiles_list = smiles_list.split(",")
    
    # Create synthon object
    synthonspace = rdSynthonSpaceSearch.SynthonSpace()
    synthonspace.ReadDBFile(synthons_file)

    output = np.zeros(len(smiles_list))

    for idx, smiles in enumerate(smiles_list):
        try:
            # Perform search
            output[idx] = freedom_search(
                smiles=smiles, 
                random_seed=random_seed, 
                max_hits=max_hits
            )
        except Exception:
            output[idx] = 0

    print(output.tolist())
