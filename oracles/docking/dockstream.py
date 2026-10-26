"""
Adapted from https://github.com/MolecularAI/reinvent-scoring/blob/main/reinvent_scoring/scoring/score_components/structural/dockstream.py.
"""
import io
import os
import subprocess
import warnings
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol

# OpenBabel environment variables that must NOT leak from the Saturn conda
# environment into the DockStream subprocess. DockStream ships its own OpenBabel
# installation; if these point at Saturn's OpenBabel, DockStream loads
# incompatible plugins, which silently breaks ligand conversion and docking.
# See the reproducibility note in oracles/docking/README-dockstream.md.
_OPENBABEL_ENV_VARS_TO_STRIP = ("BABEL_LIBDIR", "BABEL_DATADIR")

# Fraction of a batch that is allowed to fail docking before the whole run is
# aborted. A small number of failures (e.g. an un-embeddable ligand) is normal,
# but a large fraction almost always indicates a broken docking setup (e.g. the
# OpenBabel leakage above) rather than genuinely bad molecules. Failing loudly
# prevents a run from silently optimizing against meaningless (0.0) scores.
_MAX_DOCKING_FAILURE_FRACTION = 0.10

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

    @staticmethod
    def _build_subprocess_env() -> dict:
        """
        Build the environment variable mapping passed to the DockStream subprocess.

        Returns a copy of the current process environment with the OpenBabel
        variables in ``_OPENBABEL_ENV_VARS_TO_STRIP`` removed. This isolates
        DockStream's OpenBabel installation from Saturn's, preventing
        incompatible OpenBabel plugins (inherited via BABEL_LIBDIR /
        BABEL_DATADIR) from corrupting ligand conversion and docking.

        Returns:
            dict: A sanitized copy of ``os.environ`` for the subprocess.
        """
        subprocess_env = os.environ.copy()
        for var in _OPENBABEL_ENV_VARS_TO_STRIP:
            subprocess_env.pop(var, None)
        return subprocess_env

    def __call__(self, mols: np.ndarray[Mol], oracle_calls: int) -> np.ndarray[float]:
        # FIXME: Bad practice as the function signature is not the same as the parent class abstract method
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles, oracle_calls)
    
    def _compute_property(self, smiles: np.ndarray[str], oracle_calls: int) -> np.ndarray[float]:
        """
        Run DockStream and return the docking scores.

        Any DockStream output that cannot be parsed as a float is treated as a
        failed docking calculation and assigned a score of 0.0 (the neutral
        value for the downstream reward transformation). The number of such
        failures is tracked: if it reaches ``_MAX_DOCKING_FAILURE_FRACTION`` of
        the batch, a RuntimeError is raised to terminate the run rather than
        continuing to optimize against meaningless scores. This makes systematic
        docking failures (e.g. a broken DockStream/OpenBabel setup) fail loudly
        instead of silently degrading the optimization.

        Args:
            smiles: The batch of SMILES strings to dock.
            oracle_calls: Number of oracle calls so far (used to tag outputs).

        Returns:
            np.ndarray[float]: The docking score per molecule (0.0 for failures).

        Raises:
            RuntimeError: If the fraction of failed dockings in the batch is at
                least ``_MAX_DOCKING_FAILURE_FRACTION``.
        """
        command = self._create_command(smiles, oracle_calls)
        dockstream_results = self._get_docking_scores(command, len(smiles))
        docking_scores = []
        num_failed = 0
        for result in dockstream_results:
            try:
                docking_scores.append(float(result))
            except ValueError:
                docking_scores.append(0.0)
                num_failed += 1

        self._check_failure_rate(num_failed, len(smiles))

        return np.array(docking_scores)

    def _check_failure_rate(self, num_failed: int, batch_size: int) -> None:
        """
        Guard against systematic docking failures within a batch.

        Emits a warning whenever any docking failed, and raises a RuntimeError
        if the failure fraction reaches ``_MAX_DOCKING_FAILURE_FRACTION``. A
        high failure rate is almost always a configuration/environment problem
        (e.g. OpenBabel plugin leakage into the DockStream subprocess) rather
        than genuinely undockable molecules, so the run is aborted to surface
        the problem instead of returning meaningless 0.0 scores.

        Args:
            num_failed: Number of molecules in the batch that failed docking.
            batch_size: Total number of molecules in the batch.

        Raises:
            RuntimeError: If ``num_failed / batch_size`` is at least
                ``_MAX_DOCKING_FAILURE_FRACTION``.
        """
        if num_failed == 0 or batch_size == 0:
            return

        failure_fraction = num_failed / batch_size
        if failure_fraction >= _MAX_DOCKING_FAILURE_FRACTION:
            raise RuntimeError(
                f"DockStream failed to produce valid docking scores for "
                f"{num_failed}/{batch_size} molecules "
                f"({failure_fraction:.0%}), which is at or above the "
                f"{_MAX_DOCKING_FAILURE_FRACTION:.0%} tolerance. This usually "
                f"indicates a broken docking setup rather than undockable "
                f"molecules. A common cause is the Saturn environment's "
                f"OpenBabel variables ({', '.join(_OPENBABEL_ENV_VARS_TO_STRIP)}) "
                f"leaking into the DockStream subprocess. Verify the DockStream "
                f"installation and configuration before rerunning."
            )

        warnings.warn(
            f"DockStream failed to produce valid docking scores for "
            f"{num_failed}/{batch_size} molecules ({failure_fraction:.0%}); "
            f"these were assigned a score of 0.0.",
            RuntimeWarning,
        )
        
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

        The subprocess is launched with an explicitly sanitized environment
        (see ``_build_subprocess_env``) so that Saturn's OpenBabel configuration
        does not interfere with DockStream's own OpenBabel installation.
        """
        subprocess_env = self._build_subprocess_env()
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=subprocess_env) as proc:
            wrapped_proc_in = io.TextIOWrapper(proc.stdin, "utf-8")
            wrapped_proc_out = io.TextIOWrapper(proc.stdout, "utf-8")
            result = [str(wrapped_proc_out.readline()).strip() for idx in range(num_scores)]
            wrapped_proc_in.close()
            wrapped_proc_out.close()
            proc.wait()
            proc.terminate()
        return result
