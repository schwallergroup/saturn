from typing import Tuple
import os
import subprocess
import tempfile
import shutil
import json
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from oracles.docking.dataclass import ConstrainedDockingParameters
from oracles.docking.utils.constrained_docking_utils import extract_hbind_interactions, match_interactions, dense_reward_multiplier, sdf2smiles



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

        # Ligand embedding parameter - use ETKDG: https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654
        self.ETKDG = AllChem.ETKDG()

        # gnina parameters 
        self.exhaustiveness = self.parameters.specific_parameters.get("exhaustiveness", 8)  # Default to 8
        self.flexdist = self.parameters.specific_parameters.get("flexdist", 0)  # Allow flexible residues up to `flexdist` Angstroms from ligand. Default to 0

        # Constrained docking parameters
        self.constrained_docking_parameters = ConstrainedDockingParameters(**self.parameters.specific_parameters.get("constrained_docking", {}))
        self.constrained_generated_smiles = dict()

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
    ) -> Tuple[np.ndarray[float], np.ndarray[float]]:
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
                AllChem.EmbedMolecule(mol, self.ETKDG)
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

        # 3. Execute `gnina` and extract the raw docking scores
        # TODO: Could be paralellized but GPU docking should be fast enough as large libraries are not expected
        # Change directory to the `gnina` binary directory
        current_dir = os.getcwd()
        os.chdir(os.path.dirname(self.binary))

        # Sort the SDF files by ascending order to ensure the proper docking scores are assigned to the ligands
        sdf_files = os.listdir(temp_input_sdf_dir)
        sdf_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        docking_scores = np.zeros(len(sdf_files))
        for idx, sdf in enumerate(sdf_files):
            # NOTE: `gnina` by default keeps hydrogens which is important for `Hbind` interaction mapper
            output = subprocess.run([
                "./gnina", 
                "-r", self.receptor, 
                "-l", os.path.join(temp_input_sdf_dir, sdf), 
                "--autobox_ligand", self.reference_ligand,
                "--autobox_add", "4",
                "--flexdist_ligand", self.reference_ligand,
                "--flexdist", str(self.flexdist),
                "-o", os.path.join(temp_output_dir, f"docked_pose_{idx+1}.sdf"), 
                "--num_modes", "1",
                "--exhaustiveness", str(self.exhaustiveness),
                "--seed", str(0)
            ], capture_output=True)

            try:    
                # Open the output SDF file and extract the minimized affinity using RDKit
                suppl = Chem.SDMolSupplier(os.path.join(temp_output_dir, f"docked_pose_{idx+1}.sdf"))
                mol = next(suppl)
                if mol is not None and mol.HasProp("minimizedAffinity"):
                    affinity = float(mol.GetProp("minimizedAffinity"))
                    docking_scores[idx] = affinity
                else:
                    docking_scores[idx] = 0.0  # Set to 0.0 if affinity not found or molecule is None
            # In case of docking fails
            except Exception:
                docking_scores[idx] = 0.0  # Set to 0.0 if affinity not found or molecule is None

        # 4. Check for enforced interactions
        # `Hbind` expects the receptor as .pdb so convert from .pdbqt to .pdb
        output = subprocess.run([
            "obabel",
            "-ipdbqt", self.receptor,
            "-O", os.path.join(temp_output_dir, "receptor.pdb")
        ], capture_output=True)
        assert output.returncode == 0, "Failed to convert receptor from .pdbqt to .pdb."

        reward_multiplier = np.zeros(len(sdf_files))
        if self.constrained_docking_parameters.enforce_interactions:
            # Tracker
            self.constrained_generated_smiles[oracle_calls] = []

            # Change directory to the `Hbind` binary directory
            os.chdir(self.constrained_docking_parameters.hbind_binary)

            
            for idx, sdf in enumerate(sdf_files):
                try:
                    # Obabel: Convert docked pose SDF to mol2
                    output = subprocess.run([
                        "obabel",
                        "-isdf", os.path.join(temp_output_dir, f"docked_pose_{idx+1}.sdf"),
                        "-O", os.path.join(temp_output_dir, f"temp_ligand_{idx+1}.mol2")
                    ], capture_output=True)
                    if not os.path.exists(os.path.join(temp_output_dir, f"temp_ligand_{idx+1}.mol2")):
                        reward_multiplier[idx] = 0.0
                        continue

                    # Hbind: Interaction table
                    output = subprocess.run([
                        "./bin/hbind",
                        "-p", os.path.join(temp_output_dir, "receptor.pdb"),
                        "-l", os.path.join(temp_output_dir, f"temp_ligand_{idx+1}.mol2"),
                        # Include salt-bridges
                        "-s",
                        # Print summary table
                        "-t"
                    ], capture_output=True)

                    # Check for errors
                    if output.returncode != 0:
                        reward_multiplier[idx] = 0.0
                        continue
                        
                    output = output.stdout.decode("utf-8")
                    interactions = extract_hbind_interactions(output, "hbond")  # List[Tuple[str, float]

                    # Binary constrained-docking reward
                    if not self.constrained_docking_parameters.use_dense_reward:
                        enforced_residues_found = match_interactions(
                            interactions=interactions, 
                            enforced_residues=self.constrained_docking_parameters.enforced_residues, 
                            interaction_type="hbond"
                        )
                        reward_multiplier[idx] = 1.0 if len(enforced_residues_found) == len(self.constrained_docking_parameters.enforced_residues) else 0.0
                    # Dense constrained-docking reward
                    else:
                        reward_multiplier[idx] = dense_reward_multiplier(
                            interactions=interactions, 
                            enforced_residues=self.constrained_docking_parameters.enforced_residues,
                            interaction_type="hbond"
                        )

                    if reward_multiplier[idx] == 1.0:
                        self.constrained_generated_smiles[oracle_calls].append(sdf2smiles(os.path.join(temp_output_dir, f"docked_pose_{idx+1}.sdf")))

                except Exception:
                    reward_multiplier[idx] = 0.0

            with open(os.path.join(self.output_dir, "constrained_generated_smiles.json"), "w") as f:
                json.dump(self.constrained_generated_smiles, f, indent=4)

        # 5. Change back to the original directory
        os.chdir(current_dir)

        # 6. Copy and save the docking output
        # Remove the temporary files
        os.remove(os.path.join(temp_output_dir, "receptor.pdb"))
        for sdf in sdf_files:
            os.remove(os.path.join(temp_output_dir, f"temp_ligand_{sdf.split('_')[1].split('.')[0]}.mol2"))
        subprocess.run([
            "cp", 
            "-r", 
            temp_output_dir, 
            os.path.join(self.output_dir, f"results_{oracle_calls}")
        ])

        # 7. Delete the temporary folders
        shutil.rmtree(temp_input_sdf_dir)
        shutil.rmtree(temp_output_dir)

        return docking_scores, reward_multiplier
