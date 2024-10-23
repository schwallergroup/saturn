from typing import Dict, Union, List, Any
import pickle
import json
import sys

def extract_data(file_path: str, data_type: str) -> Dict[str, Union[str, int]]:
    # Load the pickle file
    with open(file_path, "rb") as f:
        route = pickle.load(f)

    if data_type == "mol":
        return mol_data_from_pickle(route)
    elif data_type == "rxn":
        return rxn_data_from_pickle(route)
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    
def mol_data_from_pickle(route: List[Any]) -> Dict[str, Union[str, int]]:
    # For every entry, extract the Mol nodes' SMILES and depth
    syntheseus_route_data = {}
    for idx, node in enumerate(route):
        if hasattr(node, "mol"):
            syntheseus_route_data[idx] = {
                "smiles": node.mol.smiles,
                "depth": node.depth
            }

    # Convert the route data to JSON format
    return json.dumps(syntheseus_route_data)

def rxn_data_from_pickle(route: List[Any]) -> Dict[str, Union[str, int]]:
    # For every entry, extract the Reaction nodes' SMILES and depth
    syntheseus_route_data = {}
    for idx, node in enumerate(route):
        if hasattr(node, "reaction"):
            syntheseus_route_data[idx] = {
                "rxn_smiles": node.reaction,
                "depth": node.depth
            }

    # Convert the route data to JSON format
    return json.dumps(syntheseus_route_data)

if __name__ == "__main__":
    pickle_file = sys.argv[1]  # The path to the pickle file is passed as an argument
    data_type = sys.argv[2]  # The type of data to extract is passed as an argument
    data = extract_data(
        file_path=pickle_file,
        data_type=data_type
    )
    print(data)  # Capture result
