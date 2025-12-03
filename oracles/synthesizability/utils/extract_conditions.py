import json
import tempfile
from typing import List, Dict
from pathlib import Path
import os
import subprocess
import sys

def quarc_condition_extraction(smiles_file: str,
                               namerxn_binary: str,
                               quarc_path: str,
                               quarc_env: str,
                               model: str = "ffn") -> Dict:
    """Extract conditions for given SMILES using QUARC.
    """

    assert model in ["gnn", "ffn"], f"{model} is not valid, please, provide valid inference pipeline"

    # 1. Write SMILES and extract NameRxn classes and mapping
    temporary_dir = tempfile.mkdtemp()
    output_smiles_path = os.path.join(temporary_dir, "out.smi")

    # Call namerxn
    subprocess.run(
        [f"{namerxn_binary}",
        f"{smiles_file}",
        f"{output_smiles_path}"
        ]
    )

    rxns = []

    # Extract output
    with open(output_smiles_path, "r") as f:
        for rxn in f.readlines():
            outs = rxn.strip().split(" ")
            rxns.append((outs[0], outs[1]))


    # 2. Write QUARC input and execute it

    # Move to QUARC directory
    os.chdir(quarc_path)

    input_quarc = os.path.join(quarc_path, "input.json")
    output_quarc = os.path.join(quarc_path, "output.json")

    with open(input_quarc, "w") as f:
        data = [{"rxn_smiles": rxn[0], "rxn_class": rxn[1], "doc_id": i} for i, rxn in enumerate(rxns)]
        json.dump(data, f, indent=4)
    
    # Run predictions
    subprocess.run([
        "conda",
        "run",
        "-n",
        f"{quarc_env}",
        "python",
        "scripts/inference.py",
        "--config-path",
        f"configs/{model}_pipeline.yaml",
        "--input",
        f"{input_quarc}",
        "--output",
        f"{output_quarc}",
        "--top-k",
        "1"
    ])

    # 3. Extract conditions from QUARC output
    with open(output_quarc, "r") as f:
        conditions = json.load(f)
    
    conditions = [cond["predictions"][0] for cond in conditions]

    return conditions


if __name__ == "__main__":
    
    NAMERXN = "/data/namerxn"

    namerxn_binary = sys.argv[1]
    quarc_script = sys.argv[2]
    quarc_env = sys.argv[3]
    smiles_file = sys.argv[4]

    conds = quarc_condition_extraction(smiles_file,
                               namerxn_binary,
                               quarc_script,
                               quarc_env)

    print(conds)