from typing import Dict, Union
import sys

from rxn_insight.reaction import Reaction

def info_from_rxn_smiles(rxn_smiles: str) -> Dict[str, Union[str, int]]:
    """
    Runs Rxn-INSIGHT on the reaction SMILES and returns reaction information.

    Reference: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00834-z
    """
    rxn = Reaction(rxn_smiles)
    return rxn.get_reaction_info()

if __name__ == "__main__":
    rxn_smiles = sys.argv[1]  # The reaction SMILES string is passed as an argument
    rxn_info = info_from_rxn_smiles(rxn_smiles)
    print(rxn_info)  # Capture result
