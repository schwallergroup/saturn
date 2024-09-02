from typing import Dict, Union
import pickle
import json
import sys

def data_from_pickle(file_path: str) -> Dict[str, Union[str, int]]:
    # Load the pickle file
    with open(file_path, "rb") as f:
        route = pickle.load(f)

    # For every entry, extract the nodes' SMILES and depth
    syntheseus_route_data = {}
    for idx, node in enumerate(route):
        attributes = dir(node)
        if "mol" in attributes:
            syntheseus_route_data[idx] = {
                "smiles": node.mol.smiles,
                "depth": node.depth
            }

    # Convert the route data to JSON format
    return json.dumps(syntheseus_route_data)

if __name__ == "__main__":
    pickle_file = sys.argv[1]  # The path to the pickle file is passed as an argument
    data = data_from_pickle(pickle_file)
    print(data)  # Capture result
