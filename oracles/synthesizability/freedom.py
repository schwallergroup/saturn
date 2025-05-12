import numpy as np
import subprocess
import ast
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol



class Freedom(OracleComponent):
    """
    Freedom search score based on: https://greglandrum.github.io/rdkit-blog/posts/2024-12-03-introducing-synthon-search.html
    Checks if the molecules are contained in Chemspace's Freedom 4.0 space or not.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # Freedom environment name
        self.env_name = self.parameters.specific_parameters.get("freedom_environment", None)
        assert self.env_name is not None, "Please provide the conda environment to run Freedom search"

        # Freedom extraction script
        self.freedom_script = self.parameters.specific_parameters.get("freedom_script", None)
        assert self.freedom_script is not None, "Please provide the path to the Freedom script"

        # Path to the freedom .spc file
        self.freedom_file_path = self.parameters.specific_parameters.get("freedom_file_path", None)
        assert self.freedom_file_path is not None, "Please provide the path to the Freedom .spc file"

        # Random seed for search
        self.random_seed = self.parameters.specific_parameters.get("random_seed", 33)
        
        # Maximum number of hits
        self.max_hits = self.parameters.specific_parameters.get("max_hits", 5000)

    def __call__(
        self, 
        mols: np.ndarray[Mol]
    ) -> np.ndarray[int]:

        smiles = np.vectorize(Chem.MolToSmiles)(mols)

        return self._compute_property(smiles)

    def _compute_property(
        self, 
        smiles: np.ndarray[str]
    ) -> np.ndarray[int]:
        """
        Execute Freedom search on the SMILES batch.
        """

        smiles_string = ",".join(list(smiles))

        try:
            output = subprocess.run([
                "conda",
                "run",
                "-n",
                self.env_name,
                "python",
                self.freedom_script,
                smiles_string,
                self.freedom_file_path,
                str(self.random_seed),
                str(self.max_hits)
            ], 
            capture_output=True, 
            text=True)

            is_in_freedom = ast.literal_eval(output.stdout)

            return np.array(is_in_freedom)
        
        except Exception:
            return np.zeros(len(smiles))
