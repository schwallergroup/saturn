"""
Adapted from https://github.com/MolecularAI/reinvent-scoring/blob/main/reinvent_scoring/scoring/score_components/structural/dockstream.py.
"""
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol

class DockStream(OracleComponent):
    """
    DockStream is a wrapper around various ligand enumerators/3D conformation generators and docking algorithms.
    The interface can take as input SMILES strings and return docking scores.
    Based on: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00563-7
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.docking_configuration_path = parameters.specific_parameters["configuration_path"]
        self.docker_script_path = parameters.specific_parameters["docker_script_path"]
        self.environment_path = parameters.specific_parameters["environment_path"]

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles)
    
    def _compute_property(self, smiles: np.ndarray[str]) -> np.ndarray[float]:
        """
        Run DockStream and return the docking scores.
        """
        command = self._create_command(smiles)
        try:
            return 123
        except Exception:
            return 0.0
        
    def _create_command(self, smiles: np.ndarray[str]) -> str:
        """
        Create the CLI command to run DockStream.
        """
        # pass entire batch to DockStream
        concatenated_smiles = '"' + ";".join(smiles) + '"'
        command = " ".join([
            self.environment_path,
            self.docker_script_path,
            "-conf", self._configuration_path,
            #"-output_prefix", self._get_step_string(step),
            "-smiles", concatenated_smiles,
            "-print_scores", 
            "-debug"
        ])
        return command

