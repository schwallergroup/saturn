from typing import Tuple
import os
import subprocess
import tempfile
import shutil
import pandas as pd
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol


class AiZynthFinder(OracleComponent):
    """
    Wrapper around AiZynthFinder which is a retrosynthesis software.

    References:
    1. https://pubs.rsc.org/en/content/articlelanding/2020/sc/c9sc04944d
    2. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00472-1
    3. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00860-x
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # AiZynthFinder environment path
        self.env_name = self.parameters.specific_parameters.get("env_name", None)
        assert self.env_name is not None, "Please provide the Conda environment name with AiZynthFinder installed."
        # Path to AiZynthFinder configuration file
        self.config_path = self.parameters.specific_parameters.get("config_path", None)
        assert self.config_path is not None, "Please provide the path to an AiZynthFinder configuration file."
        # Whether to optimize for path length
        self.optimize_path_length = self.parameters.specific_parameters.get("optimize_path_length", False)
        # Download default AiZynthFinder models and stock databases with specified environment
        self._download_public_data()

    def __call__(
        self, 
        mols: np.ndarray[Mol]
    ) -> Tuple[np.ndarray[bool], np.ndarray[int]]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles)
    
    def _compute_property(
        self, 
        smiles: np.ndarray[str]
    ) -> Tuple[np.ndarray[bool], np.ndarray[int]]:
        """
        Execute AiZynthFinder on the SMILES batch.
        # TODO: parallelize
        """
        # 1. Make a temporary file to store the SMILES
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, "smiles.smi"), "w") as f:
            for smile in smiles:
                f.write(f"{smile}\n")

        # 2. Run AiZynthFinder
        subprocess.run([
            "conda",
            "run",
            "-n",
            self.env_name,
            "aizynthcli",
            "--config",
            self.config_path,
            "--smiles",
            os.path.join(temp_dir, "smiles.smi")
        ])

        # 3. Parse the output
        df = pd.read_json("output.json.gz", orient="table")
        solved = [int(solved) for solved in df["is_solved"]]
        # If solved, extract the number_of_steps - otherwise, set to 9999
        # This is safe because one would always want to minimize the number of steps
        steps = [steps if solved else 9999 for steps, solved in zip(df["number_of_steps"], solved)]

        # 4. Delete the temporary folder and AiZynthFinder output
        shutil.rmtree(temp_dir)
        os.remove("output.json.gz")

        return np.array(solved) if not self.optimize_path_length else np.array(steps)

    def _download_public_data(self):
        """
        Download default AiZynthFinder models and stock databases.
        """
        # Check if the data is already downloaded
        if os.path.exists(os.path.join(os.path.dirname(self.config_path), "zinc_stock.hdf5")):
            return
    
        subprocess.run([
            "conda",
            "run",
            "-n",
            self.env_name,
            "download_public_data",
            os.path.dirname(self.config_path)
        ])
        
