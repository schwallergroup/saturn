from typing import Dict, Union, Set, Any
import pickle
import json
import sys

def extract_data(file_path: str, data_type: str) -> Dict[str, Union[str, int]]:
    # Load the pickle file
    with open(file_path, "rb") as f:
        route = pickle.load(f)  # set of syntheseus nodes

    if data_type == "mol":
        data = extract_mol_data(route)
    elif data_type == "rxn":
        data = extract_rxn_data(route)
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    # Convert the data to JSON format
    return json.dumps(data)
    
def extract_mol_data(route: Set[Any]) -> Dict[str, Union[str, int]]:
    # Syntheseus Mol nodes have the attribute "mol"
    syntheseus_route_data = {}
    for idx, node in enumerate(route):
        if hasattr(node, "mol"):
            syntheseus_route_data[idx] = {
                "smiles": node.mol.smiles,
                "depth": node.depth
            }

    return syntheseus_route_data

def extract_rxn_data(route: Set[Any]) -> Dict[str, Union[str, int]]:
    # Syntheseus Reaction nodes have the attribute "reaction"
    syntheseus_route_data = {}
    for idx, node in enumerate(route):
        if hasattr(node, "reaction"):
            rxn_smiles = str(node.reaction)
            syntheseus_route_data[idx] = {
                "rxn_smiles": rxn_smiles,
                "depth": node.depth
            }

    return syntheseus_route_data

if __name__ == "__main__":
    pickle_file = sys.argv[1]  # The path to the pickle file is passed as an argument
    data_type = sys.argv[2]  # The type of data to extract is passed as an argument
    data = extract_data(
        file_path=pickle_file,
        data_type=data_type
    )
    print(data)  # Capture result
