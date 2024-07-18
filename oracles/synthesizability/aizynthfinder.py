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

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Store the current AiZynthFinder output for saving
        self.aizynth_output = pd.DataFrame()

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[int]:
        # Reset output storage
        self.aizynth_output = pd.DataFrame()
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._parallelized_compute_property(smiles, oracle_calls) if self.parallelize else self._compute_property(smiles, oracle_calls)
    
    def _parallelized_compute_property(
        self, 
        smiles: np.ndarray[str],
        oracle_calls: int
    ) -> np.ndarray[int]:
        """
        Thread Parallelized execution of AiZynthFinder on the SMILES batch.
        """
        # 1. Chunk the SMILES into max_worker batches
        smiles_chunks = np.array_split(smiles, self.max_workers)
        # Ensure "str" type
        smiles_chunks = [np.array(chunk.tolist()) for chunk in smiles_chunks]  # List[np.ndarray[str]]

        # 2. Multi-threaded execution
        workers = self.max_workers if len(smiles) > self.max_workers else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(self._compute_property, smiles_chunks, [oracle_calls] * len(smiles_chunks))
            results = list(results)
            # Flatten the results
            output = np.array([])
            for result in results:
                output = np.concatenate((output, result))

        # 3. Save the output
        self.aizynth_output.to_csv(os.path.join(self.output_dir, f"aizynth_output_{oracle_calls}.csv"), index=False)

        return output
    
    def _compute_property(
        self, 
        smiles: np.ndarray[str],
        oracle_calls: int
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
        output = subprocess.run([
            "conda",
            "run",
            "-n",
            self.env_name,
            "aizynthcli",
            "--config", self.config_path,
            "--smiles", os.path.join(temp_dir, "smiles.smi"),
            "--output", output_file
        ], capture_output=True)

        # 3. Parse the output
        try:
            df = pd.read_json(output_file, orient="table")
            is_solved = np.array(df["is_solved"]).astype(int)
            # If solved, extract the number_of_steps - otherwise, set to 99
            # This is safe because one would always want to minimize the number of steps
            steps = [steps if solved else 99 for steps, solved in zip(df["number_of_steps"], df["is_solved"])]
            # Concatenate the output
            self.aizynth_output = pd.concat([self.aizynth_output, df], ignore_index=True)
            # In case AiZynthFinder is not being parallelized, directly save output
            if not self.parallelize:
                assert len(self.aizynth_output) == len(smiles), "AiZynthFinder output length mismatch."
                self.aizynth_output.to_csv(os.path.join(self.output_dir, f"aizynth_output_{oracle_calls}.csv"), index=False)
        except Exception as e:
            print(f"Error in parsing AiZynthFinder output: {e}")
            is_solved = np.zeros(len(smiles))
            steps = np.zeros(len(smiles))
            steps.fill(99)

        # 4. Delete the temporary folder and AiZynthFinder output
        shutil.rmtree(temp_dir)

        # 5. Prepare and/or return the output
        if self.optimize_path_length:
            return np.array(steps)
        else:
            # Even if the path length is not being optimized, the output is still the number of steps so that this information is tracked 
            # HACK: Path length is only meaningful if a route is solved. If not solved, set the path length = -99 
            #       to work with the "binary" Reward Shaping function which sets reward = 1 if path >= 1, 0 otherwise
            return np.array([steps if solved else -99 for steps, solved in zip(steps, is_solved)])

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
