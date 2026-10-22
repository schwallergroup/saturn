from typing import List, Dict
import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path


def quarc_condition_extraction(
    smiles_file: str,
    namerxn_binary: str,
    quarc_path: str,
    quarc_env: str,
    top_k: int = 1,
    model: str = "ffn"
    # FIXME: add the expected type to output Dict
) -> List[List[Dict]]:
    """
    Extract conditions for given SMILES using QUARC.

    Reference: https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc04957a

    Args:
        smiles_file: Path to the file containing the reaction SMILES (one per line).
        namerxn_binary: Path to the NameRXN executable used to classify the reactions.
        quarc_path: Path to the local QUARC repository.
        quarc_env: Name of the Conda environment with QUARC installed.
        top_k: Number of top-ranked predicted condition sets to return per reaction.
            Defaults to 1 (top-1 only), recovering the previous behaviour.
        model: QUARC inference pipeline to use ("ffn" or "gnn").

    Returns:
        A list with one entry per input reaction. Each entry is the list of that reaction's
        top-`top_k` predicted condition dicts (each with "agents" and "temperature" keys),
        ranked from most to least confident.
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
    
    os.chdir(quarc_path)
    input_quarc = os.path.join(temporary_dir, "input.json")
    output_quarc = os.path.join(temporary_dir, "output.json")

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
        f"{top_k}"
    ])

    # 3. Extract conditions from QUARC output
    with open(output_quarc, "r") as f:
        conditions = json.load(f)

    # Keep the top-`top_k` ranked predictions for each reaction. QUARC may return fewer
    # than `top_k` predictions for a given reaction, in which case all available are kept.
    conditions = [cond["predictions"][:top_k] for cond in conditions]

    return conditions

if __name__ == "__main__":

    NAMERXN = "/data/namerxn"

    namerxn_binary = sys.argv[1]
    quarc_script = sys.argv[2]
    quarc_env = sys.argv[3]
    smiles_file = sys.argv[4]
    # Optional 5th argument: number of top-N predicted conditions per reaction (default 1)
    top_k = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    conds = quarc_condition_extraction(smiles_file,
                               namerxn_binary,
                               quarc_script,
                               quarc_env,
                               top_k=top_k)

    print(conds)
