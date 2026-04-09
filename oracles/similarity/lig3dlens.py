"""
3D ligand shape and electrostatic similarity oracle using lig3dlens-align.

Computes similarity between generated molecules and a set of reference ligands
by running the lig3dlens-align tool as a subprocess. For each generated molecule,
the maximum similarity score across all reference ligands is returned.

Reference: https://github.com/healx/lig3dlens/
"""
import os
import subprocess
import tempfile
import shutil
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters


VALID_SIMILARITY_TYPES = ["esp", "shape", "combo"]


class Lig3DLens(OracleComponent):
    """
    Computes 3D shape and/or electrostatic similarity between generated molecules
    and a set of reference ligands using the lig3dlens-align tool.

    The similarity_type parameter controls which score is returned:
        - "esp": electrostatic similarity (ESPSim property)
        - "shape": 3D shape similarity (ShapeSim property)
        - "combo": average of ESPSim and ShapeSim

    For each generated molecule, the maximum score across all reference ligands is returned.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # Conda environment with lig3dlens installed
        self.conda_env_name = self.specific_parameters.get("conda_env_name", None)
        assert self.conda_env_name is not None, \
            "Please provide the Conda environment name with lig3dlens installed."

        # Reference ligands (.smi file)
        self.dataset_path = self.specific_parameters.get("dataset_path", None)
        assert self.dataset_path is not None, \
            "Please provide the path to the reference ligands .smi file."
        assert self.dataset_path.endswith(".smi"), \
            "Reference dataset must be a SMILES file (.smi)."
        self.reference_smiles = self._load_reference_smiles(self.dataset_path)
        assert len(self.reference_smiles) > 0, \
            "Reference SMILES file is empty."

        # Number of conformers for lig3dlens-align
        self.num_conformers = self.specific_parameters.get("num_conformers", 10)

        # Timeout in seconds for each lig3dlens-align subprocess call.
        # When the timeout expires, molecules computed so far are kept and the
        # rest receive a score of 0.0.
        self.timeout = self.specific_parameters.get("timeout", 3600)

        # Similarity type: "esp", "shape", or "combo"
        self.similarity_type = self.specific_parameters.get("similarity_type", "combo")
        assert self.similarity_type in VALID_SIMILARITY_TYPES, \
            f"similarity_type must be one of {VALID_SIMILARITY_TYPES}, got '{self.similarity_type}'."

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        """
        Compute 3D similarity for each molecule against every reference ligand
        and return the maximum score per molecule.

        Args:
            mols: Array of RDKit Mol objects (the generated batch).

        Returns:
            Array of float scores, one per input molecule. Each score is the
            maximum similarity across all reference ligands.
        """
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        num_mols = len(smiles)

        # Score matrix: rows = molecules, cols = reference ligands
        score_matrix = np.zeros((num_mols, len(self.reference_smiles)), dtype=np.float32)

        temp_dir = tempfile.mkdtemp()

        try:
            # Write the library file once (all generated molecules)
            # lig3dlens expects CSV format: header + "smiles,name" rows
            library_path = os.path.join(temp_dir, "library.smi")
            self._write_library_file(library_path, smiles)

            # Run lig3dlens-align for each reference ligand
            for ref_idx, ref_smi in enumerate(self.reference_smiles):
                ref_path = os.path.join(temp_dir, f"ref_{ref_idx}.smi")
                self._write_reference_file(ref_path, ref_smi)

                output_sdf_path = os.path.join(temp_dir, f"output_{ref_idx}.sdf")

                try:
                    subprocess.run(
                        [
                            "conda", "run", "-n", self.conda_env_name,
                            "lig3dlens-align",
                            "--ref", ref_path,
                            "--lib", library_path,
                            "--conf", str(self.num_conformers),
                            "--out", output_sdf_path,
                        ],
                        capture_output=True,
                        timeout=self.timeout,
                    )
                except subprocess.TimeoutExpired:
                    # Timeout: parse partial output below (computed molecules
                    # get their scores, the rest stay at 0.0)
                    pass
                except Exception:
                    # Other errors: skip this reference entirely
                    continue

                # Parse whatever output exists (full run or partial after timeout)
                if os.path.exists(output_sdf_path):
                    ref_scores = self._parse_output_sdf(output_sdf_path, num_mols)
                    score_matrix[:, ref_idx] = ref_scores

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Return maximum similarity across all reference ligands per molecule
        max_scores = np.max(score_matrix, axis=1)
        return max_scores

    @staticmethod
    def _load_reference_smiles(path: str) -> list:
        """Load reference SMILES from a .smi file, one per line.

        Handles optional name columns by taking only the first whitespace-delimited token.

        Args:
            path: Path to the .smi file.

        Returns:
            List of SMILES strings.
        """
        smiles = []
        with open(path, "r") as f:
            for line in f:
                smi = line.strip().split()[0] if line.strip() else ""
                if smi:
                    smiles.append(smi)
        return smiles

    @staticmethod
    def _write_library_file(path: str, smiles_list) -> None:
        """Write SMILES to a CSV file in the format expected by lig3dlens.

        lig3dlens uses SmilesMolSupplier with titleLine=True and delimiter=",",
        so the file must have a header row and comma-separated SMILES,name columns.
        Molecule names follow the pattern mol_0, mol_1, ... for index-based lookup.

        Args:
            path: Output file path.
            smiles_list: Iterable of SMILES strings.
        """
        with open(path, "w") as f:
            f.write("smiles,name\n")
            for idx, smi in enumerate(smiles_list):
                f.write(f"{smi},mol_{idx}\n")

    @staticmethod
    def _write_reference_file(path: str, smiles: str) -> None:
        """Write a single reference SMILES to a file for lig3dlens.

        lig3dlens reads the reference via readline().split(",")[0], so a plain
        SMILES string on the first line is sufficient.

        Args:
            path: Output file path.
            smiles: Reference SMILES string.
        """
        with open(path, "w") as f:
            f.write(f"{smiles}\n")

    def _parse_output_sdf(self, sdf_path: str, expected_count: int) -> np.ndarray:
        """Parse the lig3dlens-align output SD file and extract similarity scores.

        Extracts ESPSim and/or ShapeSim properties based on self.similarity_type.
        Uses the CPD_ID property (e.g. "mol_0", "mol_1") written by lig3dlens to
        map scores back to the correct molecule index. Molecules that are missing
        or lack the required properties get 0.0.

        Args:
            sdf_path: Path to the output SD file from lig3dlens-align.
            expected_count: Number of molecules expected in the output.

        Returns:
            Array of float scores of length expected_count.
        """
        scores = np.zeros(expected_count, dtype=np.float32)

        try:
            suppl = Chem.SDMolSupplier(sdf_path)
            for mol in suppl:
                if mol is None:
                    continue

                try:
                    # Extract the molecule index from CPD_ID (format: "mol_<idx>")
                    cpd_id = mol.GetProp("CPD_ID")
                    idx = int(cpd_id.split("_")[1])
                    if idx >= expected_count:
                        continue

                    if self.similarity_type == "esp":
                        scores[idx] = float(mol.GetProp("ESPSim"))
                    elif self.similarity_type == "shape":
                        scores[idx] = float(mol.GetProp("ShapeSim"))
                    else:  # combo
                        esp_score = float(mol.GetProp("ESPSim"))
                        shape_score = float(mol.GetProp("ShapeSim"))
                        scores[idx] = (esp_score + shape_score) / 2.0
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
        except Exception:
            pass

        return scores
