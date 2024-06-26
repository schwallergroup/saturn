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
import uuid  # For generating temporary file names
from concurrent.futures import ThreadPoolExecutor


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
        # Whether to parallelize AiZynthFinder execution
        self.parallelize = self.parameters.specific_parameters.get("parallelize", True) # Defaults to True
        self.max_workers = self.parameters.specific_parameters.get("max_workers", 4)  # Default to 4 workers
        # Download default AiZynthFinder models and stock databases
        self._download_public_data()

    def __call__(
        self, 
        mols: np.ndarray[Mol]
    ) -> np.ndarray[int]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._parallelized_compute_property(smiles) if self.parallelize else self._compute_property(smiles)
    
    def _parallelized_compute_property(
        self, 
        smiles: np.ndarray[str]
    ) -> np.ndarray[int]:
        """
        Thread Parallelized execution of AiZynthFinder on the SMILES batch.
        """
        # 1. Chunk the SMILES into 4 batches
        smiles_chunks = np.array_split(smiles, self.max_workers)
        # Ensure "str" type
        smiles_chunks = [np.array(chunk.tolist()) for chunk in smiles_chunks]  # List[List[str]]

        # 2. Multi-threaded execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._compute_property, smiles_chunks)
            results = list(results)
            # Flatten the results
            output = np.array([])
            for result in results:
                output = np.concatenate((output, result))

        return output
    
    def _compute_property(
        self, 
        smiles: np.ndarray[str]
    ) -> np.ndarray[int]:
        """
        Execute AiZynthFinder on the SMILES batch.
        """
        # 1. Make a temporary file to store the SMILES
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, f"output_{uuid.uuid4().hex}.json.gz")
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
            os.path.join(temp_dir, "smiles.smi"),
            "--output",
            output_file
        ])

        # 3. Parse the output
        df = pd.read_json(output_file, orient="table")
        solved = [int(solved) for solved in df["is_solved"]]
        # If solved, extract the number_of_steps - otherwise, set to 9999
        # This is safe because one would always want to minimize the number of steps
        steps = [steps if solved else 9999 for steps, solved in zip(df["number_of_steps"], solved)]

        # 4. Delete the temporary folder and AiZynthFinder output
        shutil.rmtree(temp_dir)

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
        
