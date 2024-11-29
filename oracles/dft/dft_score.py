import os
import time
import subprocess
import tempfile
import shutil
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters



class DFTScore(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.env_name = self.parameters.specific_parameters.get("env_name", None)
        assert self.env_name is not None, "Please provide the Conda environment name with DFTScore installed."

        self.property = self.parameters.specific_parameters.get("property", None)
        assert self.property is not None, "Please provide the DFT property to be calculated."

        self.time_limit = self.parameters.specific_parameters.get("time_limit", 60)
        assert self.time_limit is not None and self.time_limit >= 30, "The time limit must be >= 30 minutes."

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_geometry_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the directory to save the DFT geometries."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        """
        Execute DFTScore on the SMILES batch.
        """
        # 1. Get SMILES
        smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]

        # 2. Create temporary directories for each SMILES
        temp_dirs = [tempfile.mkdtemp() for _ in smiles]

        # 3. Run DFTScore for each SMILES *simultaneously*
        processes = []
        for s, temp_dir in zip(smiles, temp_dirs):
            process = subprocess.Popen([
                "conda",
                "run",
                "-n",
                self.env_name,
                "dft_score",
                "--smiles", str(s),
                "--dir", str(temp_dir),
                "--task", str(self.property),
                "--time_limit", str(self.time_limit),
                "--queue", "slurm"
            ])
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.wait()

        # 4. Wait for all Slurm jobs to complete, with a maximum wait time of 1 hour
        # FIXME: Might be redundant since .wait() above should block until completion?
        start_time = time.time()
        max_wait_time = 3600
        all_completed = False

        while time.time() - start_time < max_wait_time:
            all_completed = all(os.path.exists(os.path.join(temp_dir, self.property)) for temp_dir in temp_dirs)
            if all_completed:
                break
            time.sleep(30)  # Wait for 30 seconds before checking again

        # 5. Collect results
        # TODO: Copy over final geometries and save
        results = []
        for temp_dir in temp_dirs:
            if os.path.exists(os.path.join(temp_dir, self.property)):
                with open(os.path.join(temp_dir, self.property), "r") as f:
                    results.append(float(f.read().strip()))
            else:
                results.append(float("inf"))

        # 6. Delete the temporary directories
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir)

        return np.array(results, dtype=np.float32)
