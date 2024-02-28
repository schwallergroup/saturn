"""
Adapted from https://github.com/MolecularAI/reinvent-scoring/blob/main/reinvent_scoring/scoring/score_components/structural/dockstream.py.
"""
import io
import subprocess
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol

class DockStream(OracleComponent):
    """
    DockStream is a wrapper around various ligand enumerators/3D conformation generators and docking algorithms.
    The interface can take as input SMILES strings and return docking scores.
    Based on: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00563-7.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.docking_configuration_path = parameters.specific_parameters["configuration_path"]
        self.docker_script_path = parameters.specific_parameters["docker_script_path"]
        self.environment_path = parameters.specific_parameters["environment_path"]

    def __call__(self, mols: np.ndarray[Mol], oracle_calls: int) -> np.ndarray[float]:
        # FIXME: Bad practice as the function signature is not the same as the parent class abstract method
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles, oracle_calls)
    
    def _compute_property(self, smiles: np.ndarray[str], oracle_calls: int) -> np.ndarray[float]:
        """
        Run DockStream and return the docking scores.
        """
        command = self._create_command(smiles, oracle_calls)
        dockstream_results = self._get_docking_scores(command, len(smiles))
        docking_scores = []
        for result in dockstream_results:
            try:
                docking_scores.append(float(result))
            except ValueError:
                docking_scores.append(0.0)

        return np.array(docking_scores)
        
    def _create_command(self, smiles: np.ndarray[str], oracle_calls: int) -> str:
        """
        Create the CLI command to run DockStream.
        """
        # pass entire batch to DockStream - parallelization is handled by DockStream
        concatenated_smiles = '"' + ";".join(smiles) + '"'
        command = " ".join([
            self.environment_path,
            self.docker_script_path,
            "-conf", self.docking_configuration_path,
            # tags output poses and scores with the oracle calls so far
            "-output_prefix", f"\"oracle_calls_{oracle_calls}_\"",
            "-smiles", concatenated_smiles,
            "-print_scores",
            # DockStream can log DEBUG information such as ligand preparation and/or docking fails
            "-debug"
        ])
        return command
    
    def _get_docking_scores(self, command: str, num_scores: int) -> np.ndarray[str]:
        """
        Execute and return the DockStream docking scores output.
        """
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as proc:
            wrapped_proc_in = io.TextIOWrapper(proc.stdin, "utf-8")
            wrapped_proc_out = io.TextIOWrapper(proc.stdout, "utf-8")
            result = [str(wrapped_proc_out.readline()).strip() for idx in range(num_scores)]
            wrapped_proc_in.close()
            wrapped_proc_out.close()
            proc.wait()
            proc.terminate()
        return result
