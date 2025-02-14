from typing import List, Tuple
import sys
import subprocess


def run_namerxn(
    namerxn_binary_path: str,
    rxn_smiles_file_path: str
) -> List[Tuple[str, str]]:
    """
    Runs NameRXN on the reaction SMILES and returns reaction information.

    Reference: https://www.nextmovesoftware.com/namerxn.html
    """
    output = subprocess.run([
        namerxn_binary_path, 
        "-nomap", "-nocan",
        rxn_smiles_file_path
    ], capture_output=True)
    output = output.stdout.decode().strip().split('\n')

    # Extract reaction data into list
    namerxn_output = []
    for line in output:
        if line:  # Skip empty lines
            # Remove leading ": " if present
            line = line.lstrip(": ")
            namerxn_output.append(line)
            
    # Extract the middle part between brackets for each reaction
    rxn_info = []
    for rxn in namerxn_output:
        # Extract the reaction class
        rxn_class = rxn.split("[")[1].strip().replace("]", "").replace("\n", "")

        # Extract the reaction name
        rxn_name = rxn.split("[")[0].strip()
        rxn_name = " ".join(rxn_name.split()[1:]).replace("\n", "")

        rxn_info.append((rxn_class, rxn_name))

    return rxn_info

if __name__ == "__main__":
    namerxn_binary_path = sys.argv[1]  # Path to NameRXN executable
    rxn_smiles_file_path = sys.argv[2]  # Path to file containing reaction SMILES
    rxn_info = run_namerxn(
        namerxn_binary_path=namerxn_binary_path,
        rxn_smiles_file_path=rxn_smiles_file_path
    )
    print(rxn_info)  # Capture result
