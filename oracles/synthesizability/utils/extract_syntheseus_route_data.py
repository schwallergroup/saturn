from typing import Dict, Union
import pickle
import json
import sys

def extract_data(file_path: str,) -> Dict[str, Union[str, int]]:
    """
    Extract Mols, Reactions, and relevant information.

    Extracted node information includes:
    - depth
    - is_mol
    - mol_smiles
    - is_rxn
    - rxn_smiles
    - rxn_class (dummy value added)
    - rxn_name (dummy value added)
    - is_purchasable

    """
    # Load the pickle file
    with open(file_path, "rb") as f:
        route = pickle.load(f)  # set of syntheseus nodes

    syntheseus_route_data = {}
    # Sort nodes by depth
    sorted_nodes = sorted(enumerate(route), key=lambda x: x[1].depth)
    for idx, (_, node) in enumerate(sorted_nodes):
        syntheseus_route_data[f"node_{idx+1}"] = {
            "depth": node.depth,
            "is_mol": hasattr(node, "mol"),
            "mol_smiles": getattr(node, "mol", None).smiles if hasattr(node, "mol") else None,
            "is_rxn": hasattr(node, "reaction"),
            "rxn_smiles": str(getattr(node, "reaction", None)) if hasattr(node, "reaction") else None,
            "rxn_class": None,  # Dummy value
            "rxn_name": None,  # Dummy value
            "is_purchasable": node.mol.metadata["is_purchasable"] if hasattr(node, "mol") else None
        }

    # Convert the data to JSON format
    return json.dumps(syntheseus_route_data)

if __name__ == "__main__":
    pickle_file = sys.argv[1]  # The path to the pickle file is passed as an argument
    data = extract_data(pickle_file)
    print(data)  # Capture result
