import os
import subprocess
import tempfile
import shutil
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import AllChem, Mol



class GNINA(OracleComponent):
    """
    Executes `gnina` (GPU-accelerated docking) which is a fork of smina which in turn is a fork of AutoDock Vina.

    References:
    1. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00522-2
    2. https://github.com/gnina/gnina
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # gnina binary
        self.binary = self.parameters.specific_parameters.get("binary", None)
        assert self.binary is not None, "Please provide the absolute path to the gnina binary."

        # Force-field for ligand energy minimization
        force_field_id = self.parameters.specific_parameters.get("force_field", "uff").lower()
        assert force_field_id in ["uff", "mmff94"], "force_field must be either 'uff' or 'mmff94'."
        self.force_field = AllChem.UFFOptimizeMolecule if force_field_id == "uff" else AllChem.MMFFOptimizeMolecule

        # Receptor path
        self.receptor = self.parameters.specific_parameters.get("receptor", None)
        assert self.receptor is not None and self.receptor.endswith(".pdbqt"), "Please provide the path to the receptor PDBQT file."

        # Reference ligand path
        self.reference_ligand = self.parameters.specific_parameters.get("reference_ligand", None)
        assert self.reference_ligand is not None and self.reference_ligand.endswith(".pdb"), "Please provide the path to the reference ligand PDB file."

        # `gnina` parameters
        self.exhaustiveness = self.parameters.specific_parameters.get("exhaustiveness", 8)  # Default to 8
        self.flexdist = self.parameters.specific_parameters.get("flexdist", 0)  # Allow flexible residues up to `flexdist` Angstroms from ligand. Default to 0

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        return self._compute_property(mols, oracle_calls)
    
    def _compute_property(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        """
        Execute `gnina` as a subprocess.
        """
        # 1. Make temporary files to store the input and output
        temp_input_sdf_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()

        # 2. Protonate Mols
        mols = [Chem.AddHs(mol) for mol in mols]

        # 3. Generate 1 (lowest energy) conformer using RDKit ETKDG and force-field (UFF or MMFF94) minimize
        for idx, mol in enumerate(mols):
        # Skip molecules that fail to embed
            try:
                # Generate conformer with ETKDG
                AllChem.EmbedMolecule(mol, ETversion=2, randomSeed=0)
                # Minimize conformer
                self.force_field(mol)
            except Exception:
                continue
            # Write out the minimized conformer in SDF format
            sdf_file = os.path.join(temp_input_sdf_dir, f"ligand_{idx+1}.sdf")
            writer = Chem.SDWriter(sdf_file)
            writer.write(mol)
            writer.flush()
            writer.close()

        # 4. Execute `gnina` and extract the raw docking scores
        # TODO: Could be paralellized but GPU docking should be fast enough as large libraries are not expected
        # Change directory to the `gnina` binary directory
        current_dir = os.getcwd()
        os.chdir(os.path.dirname(self.binary))

        sdf_files = os.listdir(temp_input_sdf_dir)
        docking_scores = np.zeros(len(mols))
        for sdf in sdf_files:
            # Index for docking score array
            ligand_idx = int(sdf.split("_")[1].split(".")[0])
            try:   
                # NOTE: `gnina` by default keeps hydrogens which is important for `Hbind` interaction mapper
                output = subprocess.run([
                    "./gnina", 
                    "-r", self.receptor, 
                    "-l", os.path.join(temp_input_sdf_dir, sdf), 
                    # Reference ligand for auto-box definition
                    "--autobox_ligand", self.reference_ligand,
                    # Add 4 Ångstroms to the autobox size
                    "--autobox_add", "4",
                    "--flexdist_ligand", self.reference_ligand,
                    # Allow flexible residues up to `flexdist` Ångstroms from ligand
                    "--flexdist", str(self.flexdist),
                    "-o", os.path.join(temp_output_dir, f"docked_pose_{ligand_idx}.sdf"), 
                    # Output only 1 pose
                    "--num_modes", "1",
                    "--exhaustiveness", str(self.exhaustiveness),
                    # Set seed to 0
                    "--seed", str(0)
                ], capture_output=True)
 
                # Open the output SDF file and extract the minimized affinity using RDKit
                suppl = Chem.SDMolSupplier(os.path.join(temp_output_dir, f"docked_pose_{ligand_idx}.sdf"))
                mol = next(suppl)
                if mol is not None and mol.HasProp("minimizedAffinity"):
                    affinity = float(mol.GetProp("minimizedAffinity"))
                    docking_scores[ligand_idx-1] = affinity
                else:
                    docking_scores[ligand_idx-1] = 0.0  # Set to 0.0 if affinity not found or molecule is None

            # In case of docking fails
            except Exception:
                docking_scores[ligand_idx-1] = 0.0  # Set to 0.0 if affinity not found or molecule is None

        # 5. Change back to the original directory
        os.chdir(current_dir)

        # 6. Copy and save the docking output
        # Remove the temporary files
        os.remove(os.path.join(temp_output_dir, "receptor.pdb"))
        for file in os.listdir(temp_output_dir):
            if file.startswith("temp_ligand") and file.endswith(".mol2"):
                os.remove(os.path.join(temp_output_dir, file))
        subprocess.run([
            "cp", 
            "-r", 
            temp_output_dir, 
            os.path.join(self.output_dir, f"results_{oracle_calls}")
        ])

        # 7. Delete the temporary folders
        shutil.rmtree(temp_input_sdf_dir)
        shutil.rmtree(temp_output_dir)

        return docking_scores
